import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class SphericalCamera(mi.Sensor):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f),
                                                          wavelength_sample, active)

        o = self.world_transform.translation()

        film_size = mi.Vector2f(self.film().crop_size())
        half_pixel_offset = 0.5 / film_size
        corrected_position_sample = position_sample - half_pixel_offset

        d = mi.warp.square_to_uniform_sphere(corrected_position_sample)

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

                if self.product_bsdf:
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wi_local, active=True)
                    wav_weight = dr.select(si_active, wav_weight * bsdf_val, wav_weight)

                if self.product_cosine:
                    cos_val = dr.abs(wi_local.z)
                    wav_weight = dr.select(si_active, wav_weight * cos_val, wav_weight)

        return ray, wav_weight

mi.register_sensor("gt_spherical_camera", lambda props: GTSphericalCamera(props))