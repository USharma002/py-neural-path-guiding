"""Path guiding integrator with MIS between BSDF and neural guiding."""
from __future__ import annotations

from typing import List, Tuple, Optional

import drjit as dr
import mitsuba as mi
import torch

mi.set_variant("cuda_ad_rgb")
DEVICE = "cuda"

from guiding.config import get_logger
from networks.nrc import NeuralRadianceCache
from guiding.system import PathGuidingSystem
from rendering.surface_intersection_record import SurfaceInteractionRecord
from sensor.spherical import GTSphericalCamera

logger = get_logger("integrator")


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """Compute MIS weight using the power heuristic (beta=2)."""
    a2 = dr.square(pdf_a)
    b2 = dr.square(pdf_b)
    w  = dr.select(pdf_a > 0, a2 / (a2 + b2), 0.0)
    return dr.select(dr.isfinite(w), w, 0.0)

class PathGuidingIntegrator(mi.SamplingIntegrator):
    """Path tracing integrator with neural path guiding and MIS."""
    def __init__(self: mi.SamplingIntegrator, props: mi.Properties) -> None:
        super().__init__(props)
        
        self.max_depth: int = props.get('max_depth', 5)
        self.rr_depth: int = props.get('rr_depth', 5)
        self.num_rays: int = 0

        # BSDF sampling fraction for MIS (0.5 = equal weight)
        self.bsdfSamplingFraction: float = 0.5
        self.iteration: int = 0

        self.use_nee = props.get('use_nee', False)
        
        self.surfaceInteractionRecord: Optional[SurfaceInteractionRecord] = None
        
        # For variance calculation
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)

        # Neural Radiance Cache
        self.nrc_system = NeuralRadianceCache(device=DEVICE)
        self.nrc_query_prob: float = props.get('nrc_query_prob', 0.8)

        # Path Guiding System (uses USE_NIS flag internally)
        self.guiding_system = PathGuidingSystem(device=DEVICE)
        
        # Change: Use Python bool to allow conditional skipping of Torch ops
        self.guiding: bool = False 

    def setup(
        self,
        scene: mi.Scene,
        num_rays: int,
        bbox_min: mi.Vector3f,
        bbox_max: mi.Vector3f,
        bsdfSamplingFraction: float = 0.5,
        isStoreNEERadiance: bool = False
    ) -> None:
        """Initialize the integrator for rendering."""
        self.num_rays = num_rays
        self.array_size = self.num_rays * self.max_depth

        self.bsdfSamplingFraction = bsdfSamplingFraction
        self.isStoreNEERadiance = isStoreNEERadiance

        self.scene = scene

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.guiding_system.set_scene(scene)
        self.nrc_system.set_scene(scene)

        self.guiding_system.bbox_max = bbox_max.torch().to("cuda")
        self.guiding_system.bbox_min = bbox_min.torch().to("cuda")
        
        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)
        self._data_scattered = False

        self.sumL = dr.zeros(mi.Spectrum, self.num_rays)
        self.sumL2 = dr.zeros(mi.Spectrum, self.num_rays)

        self._roughness_cache = torch.ones((self.num_rays, 1), device=DEVICE, dtype=torch.float32)

    def _get_roughness(self, n: int) -> torch.Tensor:
        """Return a slice of the pre-allocated roughness tensor."""
        if self._roughness_cache.shape[0] < n:
            self._roughness_cache = torch.ones((n, 1), device=DEVICE, dtype=torch.float32)
        return self._roughness_cache[:n]

    @dr.wrap(source='drjit', target='torch')
    def sample_guided_direction(self, position: torch.Tensor, wi: torch.Tensor, 
                                normal: torch.Tensor) -> Tuple[mi.Vector3f, mi.Float]:
        """Query the guiding network for a sampling direction."""
        with torch.no_grad():
            roughness = self._get_roughness(position.shape[1])
            wo_pred, pdf_val = self.guiding_system.sample_guided_direction(
                position.T, wi.T, roughness
            )
            
            pdf_val = pdf_val.squeeze()
            return mi.Vector3f(wo_pred.T), mi.Float(pdf_val)

    @dr.wrap(source='drjit', target='torch')
    def guiding_pdf(self, position: torch.Tensor, wi: torch.Tensor, 
                    normal: torch.Tensor, wo: torch.Tensor) -> mi.Float:
        """Query the guiding network for the PDF of a direction."""
        with torch.no_grad():
            roughness = self._get_roughness(position.shape[1])
            pdf_val = self.guiding_system.pdf(
                position.T, wi.T, wo.T, roughness
            )
            return mi.Float(pdf_val.squeeze())

    def set_guiding(self, guiding_active: bool = False) -> None:
        """Enable or disable path guiding."""
        if guiding_active:
            self.bsdfSamplingFraction = 0.5
        else:
            self.bsdfSamplingFraction = 1.0
        # Change: Set Python bool directly
        self.guiding = bool(guiding_active)

    def set_iteration(self: mi.SamplingIntegrator, iteration: int) -> None:
        self.iteration = iteration

    def resetRayPathData(self: mi.SamplingIntegrator) -> None:
        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape=self.array_size)
        self._data_scattered = False

    def process_incoming_radiance(self: mi.SamplingIntegrator, Lfinal: mi.Spectrum) -> None:
        """Calculate incoming radiance by propagating backwards."""
        rec = self.surfaceInteractionRecord
        array_idx = dr.arange(mi.UInt32, self.array_size)
        
        rec.radiance = rec.emittedRadiance * 0
        rec.product = rec.emittedRadiance * 0
        
        # Last bounce
        d = self.max_depth - 1
        depth_mask = rec.active & (array_idx % self.max_depth == d)
        
        if dr.any(depth_mask):
            Le_d = dr.gather(mi.Spectrum, rec.emittedRadiance, array_idx, depth_mask)
            Lnee_d = dr.gather(mi.Spectrum, rec.radiance_nee, array_idx, depth_mask)
            Lr_d = Lnee_d
            Lo_d = Le_d + Lr_d
            dr.scatter(rec.radiance, Lo_d, array_idx, depth_mask)
            dr.scatter(rec.product, Lo_d, array_idx, depth_mask)
        
        ray_ids = array_idx // self.max_depth
        # Propagate backward
        for d in reversed(range(self.max_depth - 1)):
            depth_mask = rec.active & (array_idx % self.max_depth == d)
            if not dr.any(depth_mask): continue
            
            Le_d = dr.gather(mi.Spectrum, rec.emittedRadiance, array_idx, depth_mask)
            Lnee_d = dr.gather(mi.Spectrum, rec.radiance_nee, array_idx, depth_mask)
            bsdf_d = dr.gather(mi.Spectrum, rec.bsdf, array_idx, depth_mask)
            
            next_vertex_idx = ray_ids * self.max_depth + (d + 1)
            Lo_next = dr.gather(mi.Spectrum, rec.product, next_vertex_idx, depth_mask)
            
            Lr_d = Lnee_d + bsdf_d * Lo_next
            Lo_d = Le_d + Lr_d
            
            dr.scatter(rec.radiance, Lo_d, array_idx, depth_mask)
            dr.scatter(rec.product, Lo_d, array_idx, depth_mask)
    
    def scatter_data_into_buffer(self: mi.SamplingIntegrator) -> None:
        if getattr(self, '_data_scattered', False):
            return
            
        rec = self.surfaceInteractionRecord
        isActive = rec.active

        rec.radiance[dr.isnan(rec.radiance)] = 0
        rec.radiance_nee[dr.isnan(rec.radiance_nee)] = 0
        
        radiance_zero = mi.luminance(rec.radiance) == 0
        woPdf_zero = rec.woPdf == 0
        woPdf_nan = dr.isnan(rec.woPdf)
        activeIndex = dr.compress(isActive & ~radiance_zero & ~woPdf_zero & ~woPdf_nan)

        if dr.width(activeIndex) == 0:
            self._data_scattered = True
            return
        
        rec.position = dr.gather(type(rec.position), rec.position, activeIndex)
        rec.normal = dr.gather(type(rec.normal), rec.normal, activeIndex)
        rec.wo = dr.gather(type(rec.wo), rec.wo, activeIndex)
        rec.wo_world = dr.gather(type(rec.wo_world), rec.wo_world, activeIndex)
        rec.wi = dr.gather(type(rec.wi), rec.wi, activeIndex)
        rec.radiance = dr.gather(type(rec.radiance), rec.radiance, activeIndex)
        rec.product = dr.gather(type(rec.product), rec.product, activeIndex)
        rec.woPdf = dr.gather(type(rec.woPdf), rec.woPdf, activeIndex)
        rec.bsdfPdf = dr.gather(type(rec.bsdfPdf), rec.bsdfPdf, activeIndex)
        rec.isDelta = dr.gather(type(rec.isDelta), rec.isDelta, activeIndex)
        rec.guidePdf = dr.gather(type(rec.guidePdf), rec.guidePdf, activeIndex)
        rec.sampleSource = dr.gather(type(rec.sampleSource), rec.sampleSource, activeIndex)
        rec.bsdfFraction = dr.gather(type(rec.bsdfFraction), rec.bsdfFraction, activeIndex)
        rec.guidingActive = dr.gather(type(rec.guidingActive), rec.guidingActive, activeIndex)
        rec.depth = dr.gather(type(rec.depth), rec.depth, activeIndex)

        if self.isStoreNEERadiance:
            rec.radiance_nee = dr.gather(type(rec.radiance_nee), rec.radiance_nee, activeIndex)
            rec.direction_nee = dr.gather(type(rec.direction_nee), rec.direction_nee, activeIndex)

        self._data_scattered = True

    @dr.syntax
    def sample(
        self: mi.SamplingIntegrator, 
        scene: mi.Scene, 
        sampler: mi.Sampler,
        ray_: mi.RayDifferential3f, 
        medium: mi.Medium = None, 
        active: bool = True,
        aovs: mi.Float = None,
    ) -> Tuple[mi.Color3f, bool, List[float]]:
        
        bsdf_ctx = mi.BSDFContext()
        ray = mi.Ray3f(ray_)
        depth = mi.UInt32(0)
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)
        η = mi.Float(1)
        active = mi.Bool(active)

        self.resetRayPathData()

        ray_index = dr.arange(mi.UInt32, dr.width(ray))
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while active:
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=depth == 0)
            bsdf = si.bsdf(ray)

            # 1. Direct Emission
            ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)
            raw_Le = ds_direct.emitter.eval(si)

            if self.use_nee:
                mis_bsdf_light = mis_weight(prev_bsdf_pdf, emitter_pdf)
            else:
                mis_bsdf_light = mi.Float(1.0)
            
            Le = β * mis_bsdf_light * raw_Le
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # 2. BSDF Sampling
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)
            bsdf_pdf = bsdf_sample.pdf
            bsdf_value = bsdf_weight * bsdf_pdf
            woPdf = mi.Float(bsdf_pdf)
            wo_local = mi.Vector3f(bsdf_sample.wo)
            wo_world = si.to_world(wo_local)
            wi_world = -ray.d
            wi_local = si.to_local(wi_world)
            delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # 3. Emitter Sampling (NEE)
            is_smooth = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            active_em = active_next & is_smooth & self.use_nee
            Lr_dir = mi.Spectrum(0)
            
            if active_em:
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
                active_em &= ds.pdf != 0.0
                wo_em = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_em, active_em)

                # --- Change: Conditional Torch Call ---
                # Only calculate guiding PDF if the Python flag is TRUE
                guiding_pdf_em = mi.Float(0.0)
                if self.guiding:
                    guiding_pdf_em = self.guiding_pdf(si.p, wi_local, si.n, wo_em)
                    guiding_pdf_em = dr.select(active_em, guiding_pdf_em, 0.0)
                
                # Combine PDFs
                pdf_hemisphere_em = bsdf_pdf_em
                if self.guiding:
                    pdf_hemisphere_em = (self.bsdfSamplingFraction * bsdf_pdf_em) + ((1 - self.bsdfSamplingFraction) * guiding_pdf_em)
                
                mis_em = dr.select(ds.delta, mi.Float(1.0), mis_weight(ds.pdf, pdf_hemisphere_em))
                Lr_dir = dr.select(active_em, β * mis_em * bsdf_value_em * em_weight, 0.0)


            # 4. BSDF or Guiding Sampling
            inputs_valid_for_guiding = dr.all(dr.isfinite(si.p))
            
            # --- Change: Use Python bool for guiding switch ---
            # If self.guiding is False, this entire block logic simplifies
            guiding_enabled_jit = mi.Bool(self.guiding) 
            do_guiding_bsdf_mis = active_next & ~delta & (self.iteration >= 0) & inputs_valid_for_guiding & guiding_enabled_jit

            active_sample_guiding_mis = sampler.next_1d(active_next) > self.bsdfSamplingFraction
            active_sample_guiding_mis &= do_guiding_bsdf_mis
            
            active_sample_bsdf_mis = do_guiding_bsdf_mis & ~active_sample_guiding_mis

            active_sample_guiding_mis &= active_next
            active_sample_bsdf_mis &= active_next

            # Initialize variables
            guided_pdf = mi.Float(0.0)
            combined_guided_pdf = mi.Float(0.0)

            # --- Change: Python conditional block for Torch ops ---
            if self.guiding:
                # This block will NOT be compiled into the kernel if self.guiding is False
                # allowing LoopRecord=True to work in unguided mode.
                
                # 1. Sample Direction
                g_dir, g_pdf = self.sample_guided_direction(si.p, wi_local, si.n)
                
                # Apply guiding sample
                wo_world[active_sample_guiding_mis] = si.to_world(g_dir)
                wo_local[active_sample_guiding_mis] = g_dir
                bsdf_value_g, bsdf_pdf_g = bsdf.eval_pdf(bsdf_ctx, si, wo_local, active_sample_guiding_mis)
                bsdf_value[active_sample_guiding_mis] = bsdf_value_g
                bsdf_pdf[active_sample_guiding_mis] = bsdf_pdf_g
                guided_pdf = g_pdf

                # 2. Evaluate PDF for MIS (if we picked BSDF sample)
                guided_pdf_for_bsdf_sample = self.guiding_pdf(si.p, wi_local, si.n, wo_local)
                
                combined_guided_pdf = dr.select(
                    active_sample_bsdf_mis,
                    guided_pdf_for_bsdf_sample,
                    guided_pdf
                )
                
                # Update woPdf and weights
                woPdf[active_next & do_guiding_bsdf_mis] = (self.bsdfSamplingFraction * bsdf_pdf) + (1 - self.bsdfSamplingFraction) * combined_guided_pdf
                bsdf_weight[active_next & do_guiding_bsdf_mis] = bsdf_value / woPdf

            # --- Bookkeeping ---
            bsdf_frac = mi.Float(self.bsdfSamplingFraction)
            guide_pdf_store = dr.select(do_guiding_bsdf_mis, combined_guided_pdf, mi.Float(0.0))
            sample_source_store = dr.select(active_sample_guiding_mis, mi.UInt32(1), mi.UInt32(0))

            L = L + Le + Lr_dir
            
            globalIndex = (ray_index * self.max_depth) + depth
            storeFlag = active & si.is_valid()

            dr.scatter(self.surfaceInteractionRecord.position, value= mi.Vector3f(si.p), index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wi, value= wi_local, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wo, value= wo_local, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wo_world, value= wo_world, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.normal, value= mi.Vector3f(si.n), index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.active, value= storeFlag, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.bsdf, value= bsdf_weight, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.throughputRadiance, value= L, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.emittedRadiance, value= Le / β, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.woPdf, value= woPdf, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.bsdfPdf, value= bsdf_pdf, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.isDelta, value= delta, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.guidePdf, value=guide_pdf_store, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.sampleSource, value=sample_source_store, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.bsdfFraction, value=bsdf_frac, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.guidingActive, value=guiding_enabled_jit, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.depth, value=mi.UInt32(depth), index=globalIndex, active=storeFlag)

            wo_pred = wo_world
            ray = si.spawn_ray(wo_pred)
            η *= bsdf_sample.eta
            β *= bsdf_weight
            
            prev_si = si
            prev_bsdf_pdf = woPdf
            prev_bsdf_delta = delta

            β_max = dr.max(β)
            active_next &= β_max != 0
            rr_prob = dr.minimum(β_max * η**2, .95)
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue
            
            depth[si.is_valid()] += 1
            active = active_next

        dr.schedule(L)
        sampler.schedule_state()
        dr.schedule(self.surfaceInteractionRecord)

        Lfinal = mi.Spectrum(L)
        self.process_incoming_radiance(Lfinal)

        self.sumL += L
        self.sumL2 += dr.square(L)
        
        N = self.iteration + 1
        mean = self.sumL / N
        mean_sq = self.sumL2 / N
        variance = dr.maximum(0.0, mean_sq - dr.square(mean))
        
        aov_list = [variance.x, variance.y, variance.z]
        return (L, depth!= 0, aov_list)


# PathGuidingIntegrator.sample = sample
mi.register_integrator("path_guiding_integrator", lambda props: PathGuidingIntegrator(props))