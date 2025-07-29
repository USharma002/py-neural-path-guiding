import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class SphericalCamera(mi.Sensor):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # 1. Wavelength sampling (standard)
        wavelengths, wav_weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f),
                                                          wavelength_sample, active)

        # 2. Define Ray Origin
        o = self.world_transform.translation()

        # 3. Correct the Pixel Coordinate Sampling
        film_size = mi.Vector2f(self.film().crop_size())
        half_pixel_offset = 0.5 / film_size
        corrected_position_sample = position_sample - half_pixel_offset

        # 4. Generate Ray Direction in a FIXED World-Aligned Frame
        d = mi.warp.square_to_uniform_sphere(corrected_position_sample)

        # Ensure directions are normalized, matching the C++ `normalize()` call (though it's
        # usually redundant for warp functions).
        d = dr.normalize(d)

        return mi.Ray3f(o, d, time, wavelengths), wav_weight

    def sample_ray_differential(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        ray, weight = self.sample_ray(time, wavelength_sample, position_sample, aperture_sample, active)
        return mi.RayDifferential3f(ray), weight

    def sample_direction(self, it, sample, active=True):
        # This function will not be correct, but it won't be used by a standard path tracer.
        return super().sample_direction(it, sample, active)


# Register the new, corrected sensor
mi.register_sensor("spherical", lambda props: SphericalCamera(props))

class GTSphericalCamera(SphericalCamera):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.initial_si = None
        self.product_bsdf = False
        self.product_cosine = False

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        ray, wav_weight = super().sample_ray(time, wavelength_sample,
                                             position_sample, aperture_sample, active)

        # Modify the starting weight if this is a GT render
        if self.initial_si is not None and dr.any(self.initial_si.is_valid()):
            si_active = active & self.initial_si.is_valid()
            if dr.any(si_active):
                si = self.initial_si
                bsdf_ctx = mi.BSDFContext()
                bsdf = si.bsdf()  # Get the BSDF at the intersection

                # The "outgoing" direction for our BSDF eval is towards the original camera.
                wo_world = -si.wi

                # The "incoming" direction for our BSDF eval is the new ray's direction.
                wi_world = ray.d

                # Get the BSDF from the context
                bsdf_to_eval = si.bsdf()
                
                wi_local = si.to_local(wi_world)  # incoming light direction
                wo_local = si.to_local(wo_world)  # outgoing direction (to camera)

                # si_for_eval.wi = wo_world

                if self.product_bsdf:
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wi_local, active=True)
                    wav_weight = dr.select(si_active, wav_weight * bsdf_val, wav_weight)

                if self.product_cosine:
                    cos_val = dr.abs(wi_local.z)
                    wav_weight = dr.select(si_active, wav_weight * cos_val, wav_weight)

        return ray, wav_weight

# Register this new, specialized camera
mi.register_sensor("gt_spherical_camera", lambda props: GTSphericalCamera(props))

def mis_weight(pdf_a, pdf_b):
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class Simple(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray_: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray_)
        depth = mi.UInt32(0)
        f = mi.Spectrum(1.)
        L = mi.Spectrum(0.)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, depth, f, L, active, prev_si))

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            bsdf: mi.BSDF = si.bsdf(ray)

            # Direct emission

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = f * ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # BSDF Sampling
            bsdf_smaple, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            # Update loop variables

            ray = si.spawn_ray(si.to_world(bsdf_smaple.wo))
            L = (L + Le)
            f *= bsdf_val

            prev_si = dr.detach(si, True)

            # Stopping criterion (russian roulettte)

            active_next &= dr.neq(dr.max(f), 0)

            rr_prop = dr.maximum(f.x, dr.maximum(f.y, f.z))
            rr_prop[depth < self.rr_depth] = 1.
            f *= dr.rcp(rr_prop)
            active_next &= (sampler.next_1d() < rr_prop)

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), [])


mi.register_integrator("simple", lambda props: Simple(props))

if __name__ == "__main__":
    scene = mi.cornell_box()
    scene['integrator']['type'] = 'simple'
    scene['integrator']['max_depth'] = 16
    scene['integrator']['rr_depth'] = 2
    scene['sensor']['sampler']['sample_count'] = 64
    scene['sensor']['film']['width'] = 1024
    scene['sensor']['film']['height'] = 1024
    scene = mi.load_dict(scene)

    hover_sensor = mi.load_dict({
        'type': 'gt_spherical_camera',
        # Set the camera's origin to the hit point
        'film': {'type': 'hdrfilm', 'width': 1024, 'height': 1024}
    })

    # Step 1: Generate a ray through the center of the film
    sensor_resolution = hover_sensor.film().crop_size()
    center_uv = mi.Point2f(0.5, 0.5)  # center of the image plane

    wavelength_sample = mi.Float(0.5)
    aperture_sample = mi.Point2f(0.5)

    # Get the ray from the custom camera
    ray, _ = hover_sensor.sample_ray(
        time=0.0,
        wavelength_sample=wavelength_sample,
        position_sample=center_uv,
        aperture_sample=aperture_sample,
        active=True
    )

    # Step 2: Trace the ray into the scene to find the intersection
    si = scene.ray_intersect(ray)

    # Optional: Check if the ray hit anything
    print("Simulated center ray hit valid surface:", si.is_valid())

    # Step 3: Set initial_si on the camera
    hover_sensor.initial_si = si
    hover_sensor.product_bsdf = True    # or False, depending on what you want
    hover_sensor.product_cosine = True  # or False, depending on what you want

    import matplotlib.pyplot as plt
    img = mi.render(scene, sensor=hover_sensor, spp=128)

    plt.imshow(img ** (1. / 2.2))
    plt.axis("off")
    plt.show()