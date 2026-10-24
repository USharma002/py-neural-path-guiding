"""Sensor that renders the full sphere of directions seen from a surface point (the hemisphere panel)."""
import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


class GTSphericalCamera(mi.Sensor):
    """Equal-area image of incident radiance at initial_si, optionally times the BSDF and/or cosine."""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.initial_si = None
        self.product_bsdf = False
        self.product_cosine = False

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        wavelengths, weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active)
        film_size = mi.Vector2f(self.film().crop_size())
        d = dr.normalize(mi.warp.square_to_uniform_sphere(position_sample - 0.5 / film_size))
        ray = mi.Ray3f(self.world_transform.translation(), d, time, wavelengths)

        si = self.initial_si
        if si is not None:
            wi_local = si.to_local(ray.d)
            valid = active & si.is_valid()
            if self.product_bsdf:
                weight = dr.select(valid, weight * si.bsdf().eval(mi.BSDFContext(), si, wi_local, True), weight)
            if self.product_cosine:
                weight = dr.select(valid, weight * dr.abs(wi_local.z), weight)
        return ray, weight

    def sample_ray_differential(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        ray, weight = self.sample_ray(time, wavelength_sample, position_sample, aperture_sample, active)
        return mi.RayDifferential3f(ray), weight


mi.register_sensor("gt_spherical_camera", lambda props: GTSphericalCamera(props))
