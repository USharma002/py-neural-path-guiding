"""Path guiding integrator with MIS between BSDF and neural guiding."""
from __future__ import annotations

from typing import List, Tuple, Optional

import drjit as dr
import mitsuba as mi
import torch

mi.set_variant("cuda_ad_rgb")

from guiding.config import get_logger
from networks.nrc import NeuralRadianceCache
from guiding.system import PathGuidingSystem
from rendering.surface_intersection_record import SurfaceInteractionRecord
from sensor.spherical import GTSphericalCamera

logger = get_logger("integrator")
DEVICE = "cuda"


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """Compute MIS weight using the power heuristic.
    
    Args:
        pdf_a: PDF of the first sampling strategy
        pdf_b: PDF of the second sampling strategy
        
    Returns:
        MIS weight for strategy A
    """
    a2 = dr.sqr(pdf_a)
    result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)
    result[dr.isnan(result)] = 0
    return result


class PathGuidingIntegrator(mi.SamplingIntegrator):
    """Path tracer with neural path guiding using vMF mixtures.
    
    Implements MIS between BSDF sampling and neural guiding distribution,
    combined with Next Event Estimation (NEE).
    
    Args:
        props: Mitsuba properties dictionary
    """

    def __init__(self: mi.SamplingIntegrator, props: mi.Properties) -> None:
        super().__init__(props)
        
        self.max_depth: int = props.get('max_depth', 5)
        self.rr_depth: int = props.get('rr_depth', 5)
        self.num_rays: int = 0

        # BSDF sampling fraction for MIS (0.5 = equal weight)
        self.bsdfSamplingFraction: Optional[float] = None
        self.iteration: int = 0
        
        self.surfaceInteractionRecord: Optional[SurfaceInteractionRecord] = None
        
        # For variance calculation
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)

        # Neural Radiance Cache
        self.nrc_system = NeuralRadianceCache(device=DEVICE)
        self.nrc_query_prob: float = props.get('nrc_query_prob', 0.8)

        # Path Guiding System (uses USE_NIS flag internally)
        self.guiding_system = PathGuidingSystem(device=DEVICE)
        self.guiding = mi.Bool(False)

    def setup(
        self,
        num_rays: int,
        bbox_min: mi.Vector3f,
        bbox_max: mi.Vector3f,
        bsdfSamplingFraction: float = 0.5,
        isStoreNEERadiance: bool = False
    ) -> None:
        """Initialize the integrator for rendering.
        
        Args:
            num_rays: Total number of rays per frame
            bbox_min: Scene bounding box minimum
            bbox_max: Scene bounding box maximum  
            bsdfSamplingFraction: Fraction of samples from BSDF (vs guiding)
            isStoreNEERadiance: Whether to store NEE radiance for training
        """

        self.num_rays = num_rays
        self.array_size = self.num_rays * self.max_depth

        self.bsdfSamplingFraction = bsdfSamplingFraction
        self.isStoreNEERadiance = isStoreNEERadiance

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

        self.guiding_system.bbox_max = bbox_max.torch().to("cuda")
        self.guiding_system.bbox_min = bbox_min.torch().to("cuda")

        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)
        self._data_scattered = False  # Track if scatter_data_into_buffer has been called

    @dr.wrap_ad(source='drjit', target='torch')
    def sample_guided_direction(
        self, 
        position: torch.Tensor, 
        wi: torch.Tensor, 
        normal: torch.Tensor
    ) -> Tuple[mi.Vector3f, mi.Float]:
        """Sample a direction from the guiding distribution.
        
        Args:
            position: World positions
            wi: Incoming directions
            normal: Surface normals (unused, kept for API compatibility)
            
        Returns:
            Tuple of (sampled direction, PDF value)
        """
        with torch.no_grad():
            roughness = torch.ones((position.shape[0], 1), device=DEVICE)
            wo_pred, pdf_val = self.guiding_system.sample_guided_direction(position, wi, roughness)
            return mi.Vector3f(wo_pred), mi.Float(pdf_val)

    @dr.wrap_ad(source='drjit', target='torch')
    def query_nrc(
        self, 
        positions: torch.Tensor, 
        normals: torch.Tensor, 
        view_dirs: torch.Tensor
    ) -> mi.Spectrum:
        """Query the Neural Radiance Cache."""
        with torch.no_grad():
            roughness = torch.ones((positions.shape[0], 1), device=DEVICE)
            L_nrc = self.nrc_system.query(positions, normals, view_dirs, roughness).float()
            return mi.Spectrum(L_nrc)
            


    @dr.wrap_ad(source='drjit', target='torch')
    def guiding_pdf(
        self, 
        position: torch.Tensor, 
        wi: torch.Tensor, 
        normal: torch.Tensor, 
        wo: torch.Tensor
    ) -> mi.Float:
        """Evaluate the guiding distribution PDF.
        
        Args:
            position: World positions
            wi: Incoming directions
            normal: Surface normals (unused, kept for API compatibility)
            wo: Directions to evaluate
            
        Returns:
            PDF values
        """
        with torch.no_grad():
            roughness = torch.ones((position.shape[0], 1), device=DEVICE)
            pdf_val = self.guiding_system.pdf(position, wi, wo, roughness)
            return mi.Float(pdf_val)

    def set_guiding(self, guiding_active: bool = False) -> None:
        """Enable or disable path guiding.
        
        Args:
            guiding_active: Whether to use neural path guiding
        """
        if guiding_active:
            self.bsdfSamplingFraction = 0.5
        else:
            self.bsdfSamplingFraction = 1.0
        self.guiding = mi.Bool(guiding_active)

    def set_iteration(self: mi.SamplingIntegrator, iteration: int) -> None:
        """Set the current rendering iteration number."""
        self.iteration = iteration

    def resetRayPathData(self: mi.SamplingIntegrator) -> None:
        """Reset the surface interaction record buffer."""
        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape=self.array_size)
        self._data_scattered = False

    def sample(
        self: mi.SamplingIntegrator, 
        scene: mi.Scene, 
        sampler: mi.Sampler,
        ray_: mi.RayDifferential3f, 
        medium: mi.Medium = None, 
        active: bool = True,
        aovs: mi.Float = None,
    ) -> Tuple[mi.Color3f, bool, List[float]]:
        """Sample radiance along a ray using path tracing with optional guiding.
        
        Args:
            scene: The scene to render
            sampler: Random number sampler
            ray_: Primary ray to trace
            medium: Participating medium (unused)
            active: Active lanes mask
            aovs: Arbitrary output variables (unused)
            
        Returns:
            Tuple of (radiance, valid mask, AOV list)
        """

        # BSDF Context for Path Tracing
        bsdf_ctx = mi.BSDFContext()

        # Copy input ray to avoid mutating caller's state
        ray = mi.Ray3f(ray_)

        depth = mi.UInt32(0)       # Current path depth
        L = mi.Spectrum(0)         # Radiance accumulator
        β = mi.Spectrum(1)         # Path throughput weight
        η = mi.Float(1)            # Index of refraction
        active = mi.Bool(active)   # Active SIMD lanes

        self.resetRayPathData()

        # Index for storing per-ray data
        ray_index = dr.arange(mi.UInt32, dr.width(ray))
        
		# Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Guiding", 
            state=lambda: (
                sampler, ray, depth, L, β, η, active,
                prev_si, prev_bsdf_pdf, prev_bsdf_delta
            )
        )
        
        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            
            # Compute a surface interaction with given ray
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------
            
            # Compute MIS weight for emitter sample from previous bounce
            ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)
            mis = mis_weight(
                prev_bsdf_pdf,
                emitter_pdf
            )

            raw_Le = ds_direct.emitter.eval(si)

            # Le = β * raw_Le
            Le = β * mis * raw_Le

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()


            # ------------------ Detached BSDF sampling -------------------
            
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            bsdf_pdf = bsdf_sample.pdf
            bsdf_value = bsdf_weight * bsdf_pdf # including cos(theta)
            woPdf = mi.Float( bsdf_pdf )
            wo_local = mi.Vector3f( bsdf_sample.wo )
            wo_world = si.to_world( wo_local )

            wi_world = -ray.d # the past/view direction (as outgoing for light)
            wi_local = si.to_local(wi_world)

            #  If we sampled a delta component, then we have a 0 probability
            #  of sampling that direction via guiding
            delta = mi.has_flag( bsdf_sample.sampled_type, mi.BSDFFlags.Delta )

            # ---------------------- Emitter sampling ----------------------

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            # Evaluate BSDF * cos(theta) differentiably for the emitter sample
            wo_em = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_em, active_em)

            # Calculate full hemisphere PDF for MIS with NEE
            guiding_pdf_em = self.guiding_pdf(
                si.p.torch(),
                wi_local.torch(),
                si.n.torch(),
                wo_em.torch()
            )

            # Combine BSDF and Guiding PDFs for the full hemisphere PDF
            pdf_hemisphere_em = (self.bsdfSamplingFraction * bsdf_pdf_em) + ((1 - self.bsdfSamplingFraction) * guiding_pdf_em)
            pdf_hemisphere_em = dr.select(self.guiding, pdf_hemisphere_em, bsdf_pdf_em)

            # Calculate MIS weight using the full hemisphere PDF
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, pdf_hemisphere_em))

            Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # NOTE: Neural Radiance Cache (NRC) integration is disabled
            # Uncomment and complete the following to enable NRC:
            # query_nrc_mask = (depth > 2) & si.is_valid() & ~delta & active
            # use_cache = query_nrc_mask & (sampler.next_1d() < self.nrc_query_prob)
            # L_cache = self.query_nrc(si.p.torch(), si.n.torch(), wo_world.torch())
            # L += dr.select(use_cache, β * bsdf_val * L_cache, mi.Spectrum(0.))
            # active_next &= ~use_cache

            # ------------------- BSDF or Guiding Sampling -----------------

            inputs_valid_for_guiding = dr.all(dr.isfinite(si.p))
            
            # Check if surface is suitable for MIS (not delta) and guiding is active
            do_guiding_bsdf_mis = active_next & ~delta & (self.iteration >= 0) & inputs_valid_for_guiding & self.guiding

            if dr.any(do_guiding_bsdf_mis):
                # Probabilistically split paths between guiding and BSDF sampling
                active_sample_guiding_mis = sampler.next_1d(active_next) > self.bsdfSamplingFraction
                active_sample_guiding_mis &= do_guiding_bsdf_mis

                # Combined PDF: p(wo) = bsdfFraction * p_bsdf(wo) + (1-bsdfFraction) * p_guide(wo)
                active_sample_bsdf_without_mis = ~do_guiding_bsdf_mis & ~active_sample_guiding_mis
                active_sample_bsdf_mis = do_guiding_bsdf_mis & ~active_sample_guiding_mis

                active_sample_guiding_mis &= active_next
                active_sample_bsdf_mis &= active_next
                active_sample_bsdf_without_mis &= active_next

                # Sample from guiding distribution
                guided_dir, guided_pdf = self.sample_guided_direction(si.p.torch(), wi_local.torch(), si.n.torch())
                wo_world[active_sample_guiding_mis] = si.to_world(guided_dir)
                wo_local[active_sample_guiding_mis] = guided_dir
                bsdf_value[active_sample_guiding_mis], bsdf_pdf[active_sample_guiding_mis] = bsdf.eval_pdf(bsdf_ctx, si, wo_local, active_sample_guiding_mis)

                # For BSDF-sampled paths, still need guiding PDF for MIS
                guided_pdf_for_bsdf_sample = self.guiding_pdf(
                    si.p.torch(), wi_local.torch(), si.n.torch(), wo_local.torch()
                )

                combined_guided_pdf = dr.select(
                    active_sample_bsdf_mis,
                    guided_pdf_for_bsdf_sample,
                    guided_pdf
                )

                # Compute combined PDF and BSDF weight
                woPdf[active_next & do_guiding_bsdf_mis] = (self.bsdfSamplingFraction * bsdf_pdf) + (1 - self.bsdfSamplingFraction) * combined_guided_pdf
                bsdf_weight[active_next & do_guiding_bsdf_mis] = bsdf_value / woPdf

            # ---- Update loop variables based on current interaction -----
            L = L + Le + Lr_dir
            
            globalIndex = (ray_index * self.max_depth) + depth
            storeFlag = active & si.is_valid() # don't use nrc for training data

            dr.scatter(self.surfaceInteractionRecord.position, value= si.p, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wi, value= wi_local, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wo, value= wo_local, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wo_world, value= wo_world, index= globalIndex, active= storeFlag)

            dr.scatter(self.surfaceInteractionRecord.normal, value= si.n, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.active, value= storeFlag, index= globalIndex, active= storeFlag)

            dr.scatter(self.surfaceInteractionRecord.bsdf, value= bsdf_weight, index= globalIndex, active= storeFlag)
            # dr.scatter(self.surfaceInteractionRecord.throughputBsdf, value= β, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.throughputRadiance, value= L, index= globalIndex, active= storeFlag)
            
            # isStoreNEERadiance = self.isStoreNEERadiance & storeFlag
            # dr.scatter(self.surfaceInteractionRecord.radiance_nee, value= Lr_dir / β, index= globalIndex, active= isStoreNEERadiance)
            # dr.scatter(self.surfaceInteractionRecord.direction_nee, value= dirToCanonical(ds.d), index= globalIndex, active= isStoreNEERadiance)
            dr.scatter(self.surfaceInteractionRecord.emittedRadiance, value= Le / β, index= globalIndex, active= storeFlag)

            dr.scatter(self.surfaceInteractionRecord.woPdf, value= woPdf, index= globalIndex, active= storeFlag)
            dr.scatter(self.surfaceInteractionRecord.bsdfPdf, value= bsdf_pdf, index= globalIndex, active= storeFlag)

            dr.scatter(self.surfaceInteractionRecord.isDelta, value= delta, index= globalIndex, active= storeFlag)

            wo_pred = wo_world

            ray = si.spawn_ray(wo_pred)
            η *= bsdf_sample.eta
            β *= bsdf_weight
            
            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = woPdf
            prev_bsdf_delta = delta

            # -------------------- Stopping criterion ---------------------
            
            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue
            
            depth[si.is_valid()] += 1
            active = active_next


        dr.schedule(L)
        sampler.schedule_state()
        dr.schedule(self.surfaceInteractionRecord)

        Lfinal = mi.Spectrum( L )
        self.process_incoming_radiance( Lfinal )

        # self.scatter_data_into_buffer()

        return (L, dr.neq(depth, 0), [])

    def process_incoming_radiance(self: mi.SamplingIntegrator, Lfinal: mi.Spectrum) -> None:
        """Calculate incoming radiance by propagating backwards from path endpoints.
        
        Args:
            Lfinal: Final accumulated radiance (unused, kept for API)
        """

        rec = self.surfaceInteractionRecord
        array_idx = dr.arange(mi.UInt32, self.array_size)

        rec.radiance = rec.emittedRadiance * 0
        rec.product = rec.emittedRadiance * 0

        # We must iterate backwards from the second-to-last bounce to the first.
        for d in range(self.max_depth - 1, -1, -1):
            # Create a mask for all active vertices at the current depth 'd'
            depth_mask = rec.active & dr.eq(array_idx % self.max_depth, d)
            if not dr.any(depth_mask):
                continue

            if d == self.max_depth - 1:
                Lo_from_next_bounce = rec.emittedRadiance * 0
            else:
                next_vertex_idx = array_idx + 1
                Lo_from_next_bounce = dr.gather(mi.Spectrum, rec.product, next_vertex_idx, depth_mask)             

            Le_d = dr.gather(mi.Spectrum, rec.emittedRadiance, array_idx, depth_mask)
            Lnee_d = dr.gather(mi.Spectrum, rec.radiance_nee, array_idx, depth_mask)
            bsdf_d = dr.gather(mi.Spectrum, rec.bsdf, array_idx, depth_mask)

            Li_d = Lnee_d + Lo_from_next_bounce
            # Li_d = Lo_from_next_bounce

            dr.scatter(rec.radiance, Li_d, array_idx, depth_mask)

            Lo_d = Le_d + bsdf_d * Li_d
            dr.scatter(rec.product, Lo_d, array_idx, depth_mask)

    def scatter_data_into_buffer(self: mi.SamplingIntegrator) -> None:
        """Filter and compact surface interaction data for training.
        
        Removes inactive interactions, NaN values, and zero-radiance samples.
        This method is idempotent - calling it multiple times has no additional effect.
        """
        # Skip if already scattered this frame
        if getattr(self, '_data_scattered', False):
            return
            
        rec = self.surfaceInteractionRecord
        isActive = rec.active

        # Filter NaN radiance values
        rec.radiance[dr.isnan(rec.radiance)] = 0
        rec.radiance_nee[dr.isnan(rec.radiance_nee)] = 0
        
        # Filter zero radiance samples
        radiance_zero = dr.eq(mi.luminance(rec.radiance), 0)
        radiance_nee_zero = dr.eq(mi.luminance(rec.radiance_nee), 0)
        bothRadianceZero = radiance_zero & radiance_nee_zero

        # Filter invalid PDF values
        woPdf_zero = dr.eq(rec.woPdf, 0)
        woPdf_nan = dr.isnan(rec.woPdf)
        
        activeIndex = dr.compress(isActive & ~bothRadianceZero & ~woPdf_zero & ~woPdf_nan)

        if dr.width(activeIndex) == 0:
            self._data_scattered = True
            return
        
        # Compact the data arrays
        rec.position = dr.gather(type(rec.position), rec.position, activeIndex)
        rec.normal = dr.gather(type(rec.normal), rec.normal, activeIndex)
        rec.wo = dr.gather(type(rec.wo), rec.wo, activeIndex)
        rec.wo_world = dr.gather(type(rec.wo_world), rec.wo_world, activeIndex)
        rec.wi = dr.gather(type(rec.wi), rec.wi, activeIndex)
        rec.radiance = dr.gather(type(rec.radiance), rec.radiance, activeIndex)

        if self.isStoreNEERadiance:
            rec.radiance_nee = dr.gather(type(rec.radiance_nee), rec.radiance_nee, activeIndex)
            rec.direction_nee = dr.gather(type(rec.direction_nee), rec.direction_nee, activeIndex)
        
        rec.product = dr.gather(type(rec.product), rec.product, activeIndex)
        rec.woPdf = dr.gather(type(rec.woPdf), rec.woPdf, activeIndex)
        rec.bsdfPdf = dr.gather(type(rec.bsdfPdf), rec.bsdfPdf, activeIndex)
        rec.isDelta = dr.gather(type(rec.isDelta), rec.isDelta, activeIndex)
        
        # Mark as scattered so we don't do it again this frame
        self._data_scattered = True

# Register new integrator
mi.register_integrator("path_guiding_integrator", lambda props: PathGuidingIntegrator(props))


if __name__ == "__main__":
    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, False)

    import matplotlib.pyplot as plt
    logger.info("Initializing Integrator")
    integrator = mi.load_dict({
        'type': 'path_guiding_integrator',
        'max_depth': 5
    })

    logger.info("Loading Scene")
    scene = mi.cornell_box()
    scene['sensor']['sampler']['sample_count'] = 1
    scene['sensor']['film']['width'] = 400
    scene['sensor']['film']['height'] = 400

    num_rays = scene['sensor']['film']['width'] * scene['sensor']['film']['height'] * 1
    scene = mi.load_dict(scene)

    integrator.setup(
        num_rays=num_rays,
        bbox_min=scene.bbox().min,
        bbox_max=scene.bbox().max
    )

    for i in range(100):
        img = mi.render(scene, integrator=integrator)
        integrator.scatter_data_into_buffer()
        loss = integrator.nrc_system.train_step(integrator)
        logger.info(f"Loss: {loss}")

    logger.info("Rendering")
    img = mi.render(scene, integrator=integrator)

    plt.imshow(img ** (1. / 2.2))
    plt.axis("off")
    plt.show()