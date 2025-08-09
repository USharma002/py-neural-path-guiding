from __future__ import annotations # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")

from path_guiding_system import PathGuidingSystem
from surface_intersection_record import SurfaceInteractionRecord
from nrc import NeuralRadianceCache
from typing import Tuple

import torch

from math_utils import *

device = "cuda"

# wo : outgoing path direction (where the light will come from)
# wi : incoming path direction (where light will go after coming)

epsilon = 0.00001
def mis_weight(pdf_a, pdf_b):
    """
        Compute the Multiple Importance Sampling (MIS) weight given the densities
        of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)
    result[ dr.isnan( result ) ] = 0
    return result


class PathGuidingIntegrator(mi.SamplingIntegrator):
    """Simple path tracer with MIS + NEE."""

    def __init__(self: mi.SamplingIntegrator, props: mi.Properties):
        super().__init__(props)
        
        self.max_depth = props.get('max_depth', 5)
        self.rr_depth = props.get('rr_depth', 5)
        self.num_rays = 0

        """
            When guiding, we perform MIS with the balance heuristic between the guiding
            distribution and the BSDF, combined with probabilistically choosing one of the
            two sampling methods. This factor controls how often the BSDF is sampled
            vs. how often the guiding distribution is sampled.
            Default = 0.5 (50%)
        """
        self.bsdfSamplingFraction = None
        self.iteration = 0
        
        self.surfaceInteractionRecord: SurfaceInteractionRecord = None
        
        # For variance calculation
        self.sumL = mi.Spectrum(0)
        self.sumL2 = mi.Spectrum(0)

        self.nrc_system = NeuralRadianceCache(device="cuda")
        self.nrc_query_prob = props.get('nrc_query_prob', 0.8) # 80% chance to query NRC after 1st bounce

        self.guiding_system = PathGuidingSystem()
        self.guiding = mi.Bool(False)

    def setup(
        self,
        num_rays: int,
        bbox_min: mi.Vector3f,
        bbox_max: mi.Vector3f,
        bsdfSamplingFraction: float = 0.5,
        isStoreNEERadiance: bool = False
    ) -> None:

        self.num_rays = num_rays
        self.array_size = self.num_rays * self.max_depth

        self.bsdfSamplingFraction = bsdfSamplingFraction
        self.isStoreNEERadiance = isStoreNEERadiance

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

        self.guiding_system.bbox_max = bbox_max.torch().to("cuda")
        self.guiding_system.bbox_min = bbox_min.torch().to("cuda")

        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)

    @dr.wrap_ad(source='drjit', target='torch')
    def sample_guided_direction(self, position, wi, normal):
        # position[position.isnan()] = 0.0
        # wi[wi.isnan()] = 0.0
        # normal[normal.isnan()] = 0.0

        with torch.no_grad():
            roughness = torch.ones((position.shape[0], 1), device=device)
            wo_pred, pdf_val = self.guiding_system.sample_guided_direction(position, wi, roughness)

            return mi.Vector3f( wo_pred ), mi.Float( pdf_val )

    @dr.wrap_ad(source='drjit', target='torch')
    def query_nrc(self, positions, normals, view_dirs):
        """Wrapper to query the NRC with correct arguments."""
        with torch.no_grad():
            roughness = torch.ones((positions.shape[0], 1), device=device)
            L_nrc = self.nrc_system.query(positions, normals, view_dirs, roughness).float()

            return mi.Spectrum( L_nrc )
            


    @dr.wrap_ad(source='drjit', target='torch')
    def guiding_pdf(self, position, wi, normal, wo):
        # position[position.isnan()] = 0.0
        # wi[wi.isnan()] = 0.0
        # normal[normal.isnan()] = 0.0
        # wo[wo.isnan()] = 0.0

        with torch.no_grad():
            roughness = torch.ones((position.shape[0], 1), device=device)
            pdf_val = self.guiding_system.pdf(position, wi, roughness, wo)
            return mi.Float( pdf_val )

    def set_guiding(self, guiding_active: bool = False):
        if guiding_active:
            self.bsdfSamplingFraction = 0.5
        else:
            self.bsdfSamplingFraction = 1.0

        self.guiding = mi.Bool(guiding_active)

    def set_iteration(self: mi.SamplingIntegrator, iteration: int):
        """Sets the current rendering iteration number."""
        self.iteration = iteration


    def resetRayPathData(self: mi.SamplingIntegrator):
        self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)

    def sample(
        self: mi.SamplingIntegrator, 
        scene: mi.Scene, 
        sampler: mi.Sampler,
        ray_: mi.RayDifferential3f, 
        medium: mi.Medium = None, 
        active: bool = True,
        aovs: mi.Float = None,
    ) -> Tuple[mi.Color3f, bool, List[float]]:

        # BSDF Context for Path Tracing
        bsdf_ctx = mi.BSDFContext()

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray_)

        depth = mi.UInt32(0)      # Depth of current vertex
        L = mi.Spectrum(0)        # Radiance accumulator
        β = mi.Spectrum(1)        # Path throughput weight
        η = mi.Float(1)           # Index of refraction
        active = mi.Bool(active)  # Active SIMD lanes

        self.resetRayPathData()

        # index to store the data for rays
        ray_index = dr.arange( mi.UInt32, dr.width(ray) )
        
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

            # --- START of THE FIX ---
            # Calculate the full hemisphere sampling PDF for the emitter's direction
            # This is where we make NEE "aware" of the combined PDF

            # First, get the guiding PDF for the direction sampled by NEE
            guiding_pdf_em = self.guiding_pdf(
                si.p.torch(),           # Current position
                wi_local.torch(),       # Incoming direction
                si.n.torch(),           # Surface normal
                wo_em.torch()           # Direction towards the light
            )

            # Combine BSDF and Guiding PDFs to get the full hemisphere PDF
            # (This must only be done where guiding is active, otherwise the PDF is just the BSDF PDF)
            pdf_hemisphere_em = (self.bsdfSamplingFraction * bsdf_pdf_em) + ((1 - self.bsdfSamplingFraction) * guiding_pdf_em)
            pdf_hemisphere_em = dr.select(self.guiding, pdf_hemisphere_em, bsdf_pdf_em)

            # Now, calculate the correct MIS weight using the full hemisphere PDF
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, pdf_hemisphere_em))
            # --- END of THE FIX ---

            Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ----------------------- Neural Radiance Cache ----------------
            # L_cache = mi.Spectrum(0)

            # query_nrc_mask = (depth > 2) & si.is_valid() & ~delta & active
            # inputs_valid_for_caching = dr.all(dr.isfinite(si.p))

            # use_cache = query_nrc_mask & (sampler.next_1d() < self.nrc_query_prob) & inputs_valid_for_caching

            # predicted_radiance_t = self.query_nrc(si.p.torch(), si.n.torch(), wo_world.torch())
            # L_cache = mi.Spectrum(predicted_radiance_t)

            # bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, si, wo_local, use_cache)

            # L += dr.select(use_cache, β * bsdf_val * L_cache, mi.Spectrum(0.))

            # active_next &= ~use_cache

            # ------------------ BSDF or Guiding ? --------------------------

            inputs_valid_for_guiding = dr.all(dr.isfinite(si.p))
            #  Check if the surface is suitable for MIS (not a perfect delta reflection/refraction).
            #  Check if guiding is even active for this iteration.
            #  Create a master mask 'do_guiding_bsdf_mis' for all paths where MIS will be performed.
            do_guiding_bsdf_mis = active_next & ~delta & (self.iteration >= 0) & inputs_valid_for_guiding & self.guiding

            if dr.any(do_guiding_bsdf_mis):
                #  Probabilistically split the MIS-enabled paths into two groups.
                #  'active_sample_guiding_mis': Paths that will generate their direction from the guide.
                #  'active_sample_bsdf_mis': Paths that will use the direction from the BSDF sample taken earlier.
                active_sample_guiding_mis = sampler.next_1d(active_next) > self.bsdfSamplingFraction
                active_sample_guiding_mis &= do_guiding_bsdf_mis

                """
                pdf(result of sampling) = pdf(sample generated from sd tree sampling) * pdf(do sd tree sampling) + pdf(sample generated from bsdf sampling) * (1-pdf(do sd tree sampling))
                """
                active_sample_bsdf_without_mis = ~do_guiding_bsdf_mis & ~active_sample_guiding_mis
                active_sample_bsdf_mis = do_guiding_bsdf_mis & ~active_sample_guiding_mis

                active_sample_guiding_mis &= active_next
                active_sample_bsdf_mis &= active_next
                active_sample_bsdf_without_mis &= active_next

                # IF sampling with Guiding technique MIS
                guided_dir, guided_pdf = self.sample_guided_direction(si.p.torch(), wi_local.torch(), si.n.torch())
                wo_world[active_sample_guiding_mis] = si.to_world(guided_dir)
                wo_local[active_sample_guiding_mis] = guided_dir
                bsdf_value[active_sample_guiding_mis], bsdf_pdf[active_sample_guiding_mis] = bsdf.eval_pdf(bsdf_ctx, si, wo_local, active_sample_guiding_mis)

                # If we instead sampled from the BSDF, we still need to know Guiding PDF for MIS
                guided_pdf_for_bsdf_sample = self.guiding_pdf(
                    si.p.torch(), wi_local.torch(), si.n.torch(), wo_local.torch()
                )

                # dr.scatter(guided_pdf, guided_pdf_for_bsdf_sample, ray_index, active_sample_bsdf_mis)
                combined_guided_pdf = dr.select(
                    active_sample_bsdf_mis, # The condition mask
                    guided_pdf_for_bsdf_sample,   # Value if true
                    guided_pdf # Value if false
                )

                # Finally, compute pdf and BSDF weight
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

            #   The two is only difference if we sample material together with the tree: if(sample.x >= bsdfSamplingFraction) woPdf = bsdfPdf * bsdfSamplingFraction;
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

          
      
    def process_incoming_radiance(self: mi.SamplingIntegrator, Lfinal : mi.Spectrum) -> None:
        """
        Calculate incoming radiance at each vertex by propagating radiance
        backwards from the end of each path, correctly including NEE.
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
        """
            Scatter surface interaction data into buffer.
            This also filter out invalid surface interactions before scattering.
        """

        # Remove surfaceInteractionRecord that are inactive. This modifies the array size.
        isActive = self.surfaceInteractionRecord.active

        # Filter NaN radiance
        self.surfaceInteractionRecord.radiance[ dr.isnan(self.surfaceInteractionRecord.radiance) ] = 0
        self.surfaceInteractionRecord.radiance_nee[ dr.isnan(self.surfaceInteractionRecord.radiance_nee) ] = 0
        
        # Check if both radiance is zero then also filter out
        radiance_zero = dr.eq( mi.luminance(self.surfaceInteractionRecord.radiance), 0 )
        radiance_nee_zero = dr.eq( mi.luminance( self.surfaceInteractionRecord.radiance_nee ), 0 )
        bothRadianceZero = radiance_zero & radiance_nee_zero

        # Remove woPdf = 0 or NaN
        woPdf_zero = dr.eq( self.surfaceInteractionRecord.woPdf, 0 )
        woPdf_nan = dr.isnan( self.surfaceInteractionRecord.woPdf )
        
        activeIndex = dr.compress(isActive & ~bothRadianceZero & ~woPdf_zero & ~woPdf_nan)


        # If there is no active element then exit
        if( dr.width( activeIndex ) == 0 ):
            return
            
        self.surfaceInteractionRecord.position = dr.gather( type(self.surfaceInteractionRecord.position), self.surfaceInteractionRecord.position, activeIndex )
        self.surfaceInteractionRecord.normal = dr.gather( type(self.surfaceInteractionRecord.normal), self.surfaceInteractionRecord.normal, activeIndex )
        
        self.surfaceInteractionRecord.wo = dr.gather( type(self.surfaceInteractionRecord.wo), self.surfaceInteractionRecord.wo, activeIndex )
        self.surfaceInteractionRecord.wo_world = dr.gather( type(self.surfaceInteractionRecord.wo_world), self.surfaceInteractionRecord.wo_world, activeIndex )
        self.surfaceInteractionRecord.wi = dr.gather( type(self.surfaceInteractionRecord.wi), self.surfaceInteractionRecord.wi, activeIndex )
        self.surfaceInteractionRecord.radiance = dr.gather( type(self.surfaceInteractionRecord.radiance), self.surfaceInteractionRecord.radiance, activeIndex )

        if self.isStoreNEERadiance:
            self.surfaceInteractionRecord.radiance_nee = dr.gather( type(self.surfaceInteractionRecord.radiance_nee), self.surfaceInteractionRecord.radiance_nee, activeIndex )
            self.surfaceInteractionRecord.direction_nee = dr.gather( type(self.surfaceInteractionRecord.direction_nee), self.surfaceInteractionRecord.direction_nee, activeIndex )
        
        self.surfaceInteractionRecord.product = dr.gather( type(self.surfaceInteractionRecord.product), self.surfaceInteractionRecord.product, activeIndex )
        self.surfaceInteractionRecord.woPdf = dr.gather( type(self.surfaceInteractionRecord.woPdf), self.surfaceInteractionRecord.woPdf, activeIndex )
        self.surfaceInteractionRecord.bsdfPdf = dr.gather( type(self.surfaceInteractionRecord.bsdfPdf), self.surfaceInteractionRecord.bsdfPdf, activeIndex )
        self.surfaceInteractionRecord.isDelta = dr.gather( type(self.surfaceInteractionRecord.isDelta), self.surfaceInteractionRecord.isDelta, activeIndex )

# Register new integrator
mi.register_integrator("path_guiding_integrator", lambda props: PathGuidingIntegrator(props))


if __name__ == "__main__":
    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, False)

    import matplotlib.pyplot as plt
    print("Initializing Integrator")
    integrator = mi.load_dict({
        'type': 'path_guiding_integrator',
        'max_depth': 5
    })

    print("Loading Scene")
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
        print(f"Loss: {loss}")

    print("Rendering")
    img = mi.render(scene, integrator=integrator)

    plt.imshow(img ** (1. / 2.2))
    plt.axis("off")
    plt.show()