import mitsuba as mi
import drjit as dr

class SurfaceInteractionRecord:
    """ Custom data struct to hold the recordings together """
    DRJIT_STRUCT = {
        'position' : mi.Vector3f, # normalized to [-1, 1]^3?
        'wi' : mi.Vector3f, # [theta, phi] wi, where theta in [0, pi] and phi in [0, 2pi]
        'wo' : mi.Vector3f, # [theta, phi] wi, where theta in [0, pi] and phi in [0, 2pi]
    
        # Storation needed for calculation
        'bsdf' : mi.Color3f, # [optional] for learning 5-D product distribution L_i * f_r
        'normal': mi.Vector3f, # [auxiliary] surface (shading) normal where theta in [0, pi] and phi in [0, 2pi]
        'throughputBsdf' : mi.Color3f, # throughput at current radiance, used to calculate local radiance
        'throughputRadiance' : mi.Spectrum, # local radiance 
        'emittedRadiance' : mi.Spectrum, # Le emitted from light source
        
        'radiance' : mi.Float,
        'product' : mi.Spectrum, # radiance_spectrum * bsdf  (outgoing radiance)
        
        'woPdf' : mi.Float, # the combined pdf of the sampled direction
        'bsdfPdf' : mi.Float, # [optional] the pdf of the sampled bsdf, for learning selection
        'statisticalWeight' : mi.Float,
        'isDelta' : mi.Bool, # is this scatter event is sampled from a delta lobe?
        'active' : mi.Bool, # whether current lane is active or not
        'twoSided': mi.Bool, # is the surface two-sided?
        'miss': mi.Bool, # is this scatter event added for NRC alignment since it's a miss?
        
        # Opitonal inputs
        'radiance_nee' : mi.Color3f, # radiance by the NEE
        'direction_nee' : mi.Vector2f, # direction of the light for NEE
        'diffuse' : mi.Color3f, # [auxiliary] diffuse color of the surface
        'specular': mi.Color3f, # [auxiliary] specular color of the surface
    }

    def __init__(self) -> None:
        self.position = mi.Vector3f()
        self.wo = mi.Vector3f()
        self.wi = mi.Vector3f()
        self.normal = mi.Vector3f()

        self.emittedRadiance = mi.Spectrum() # Le
        self.throughputRadiance = mi.Spectrum() # L
        
        self.bsdf = mi.Color3f() # bsdf
        self.throughputBsdf = mi.Color3f() # thp
        self.normal = mi.Vector3f()
        
        self.product = mi.Spectrum() # bsdf * L
        
        self.woPdf = mi.Float()
        self.bsdfPdf = mi.Float()
        self.isDelta = mi.Bool()
        
        self.active = mi.Bool()