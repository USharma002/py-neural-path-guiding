"""Data structures for recording surface interactions during path tracing."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi


class SurfaceInteractionRecord:
    """Holds per-vertex data for training guiding / NIS.

    Added fields needed for NIS paper-style MIS-aware training:
    - guidePdf: p_guide(wo) evaluated at the actually-sampled wo
    - sampleSource: 0=BSDF sampled wo, 1=Guiding sampled wo
    - bsdfFraction: mixture coefficient used at sampling time
    - guidingActive: whether guiding was enabled for that vertex
    - depth: bounce index (optional but useful for analysis/bucketing)
    """

    DRJIT_STRUCT = {
        # Geometry / directions
        "position": mi.Vector3f,
        "normal": mi.Vector3f,
        "wi": mi.Vector3f,          # local
        "wo": mi.Vector3f,          # local
        "wo_world": mi.Vector3f,    # world

        # Shading / transport terms
        "bsdf": mi.Color3f,                 # usually BSDF weight term used in throughput update
        "throughputBsdf": mi.Color3f,       # optional
        "throughputRadiance": mi.Spectrum,  # optional
        "emittedRadiance": mi.Spectrum,     # optional

        # Targets
        "radiance": mi.Float,       # scalar (often luminance-like)
        "product": mi.Spectrum,     # spectrum target used by your training pipeline

        # PDFs / sampling bookkeeping
        "woPdf": mi.Float,          # combined mixture pdf used for estimator (your existing combined_pdf)
        "bsdfPdf": mi.Float,        # bsdf-only pdf
        "guidePdf": mi.Float,       # guide-only pdf at sampled wo  (NEW)
        "bsdfFraction": mi.Float,   # mixture weight used at sampling time (NEW)
        "sampleSource": mi.UInt32,  # 0=BSDF, 1=GUIDE (NEW)

        # Flags / misc
        "isDelta": mi.Bool,
        "guidingActive": mi.Bool,     
        "active": mi.Bool,

        # Optional extras already present in your struct
        "statisticalWeight": mi.Float,
        "twoSided": mi.Bool,
        "miss": mi.Bool,

        "radiance_nee": mi.Color3f,
        "direction_nee": mi.Vector2f,

        "diffuse": mi.Color3f,
        "specular": mi.Color3f,

        # Optional (NEW)
        "depth": mi.UInt32,
    }

    def __init__(self) -> None:
        # Geometry / directions
        self.position = mi.Vector3f(0.0)
        self.normal = mi.Vector3f(0.0)
        self.wi = mi.Vector3f(0.0)
        self.wo = mi.Vector3f(0.0)
        self.wo_world = mi.Vector3f(0.0)

        # Shading / transport
        self.bsdf = mi.Color3f(0.0)
        self.throughputBsdf = mi.Color3f(0.0)
        self.throughputRadiance = mi.Spectrum(0.0)
        self.emittedRadiance = mi.Spectrum(0.0)

        # Targets
        self.radiance = mi.Float(0.0)
        self.product = mi.Spectrum(0.0)

        # PDFs / bookkeeping
        self.woPdf = mi.Float(0.0)
        self.bsdfPdf = mi.Float(0.0)
        self.guidePdf = mi.Float(0.0)         
        self.bsdfFraction = mi.Float(1.0)        
        self.sampleSource = mi.UInt32(0)   

        # Flags / misc
        self.isDelta = mi.Bool(False)
        self.guidingActive = mi.Bool(False)     
        self.active = mi.Bool(False)

        self.statisticalWeight = mi.Float(1.0)
        self.twoSided = mi.Bool(False)
        self.miss = mi.Bool(False)

        self.radiance_nee = mi.Color3f(0.0)
        self.direction_nee = mi.Vector2f(0.0)

        self.diffuse = mi.Color3f(0.0)
        self.specular = mi.Color3f(0.0)

        self.depth = mi.UInt32(0)               
