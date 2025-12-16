"""Shared training data structures for neural path guiding systems."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class TrainingBatch:
    """A batch of training data shared between NRC and guiding systems.

    Note: Some fields may be empty tensors if the integrator did not store them.
    """
    # Core
    position: torch.Tensor          # (N, 3) world
    normal: torch.Tensor            # (N, 3) world
    wi: torch.Tensor                # (N, 3) local (as stored by integrator)
    wo: torch.Tensor                # (N, 3) local
    wo_world: torch.Tensor          # (N, 3) world (optional but useful)

    roughness: torch.Tensor         # (N, 1)

    # PDFs / sampling bookkeeping
    combined_pdf: torch.Tensor      # (N,) == woPdf in record (final mixture pdf used for estimator)
    bsdf_pdf: torch.Tensor          # (N,) bsdfPdf
    guide_pdf: torch.Tensor         # (N,) guiding pdf for wo under guide distribution at sampling time (store this!)
    sample_source: torch.Tensor     # (N,) int8: 0=bsdf, 1=guide (store this!)
    bsdf_fraction: torch.Tensor     # (N,) float: bsdfSamplingFraction at sampling time (store this!)

    is_delta: torch.Tensor          # (N,) bool/int

    # Values / targets
    radiance_scalar: torch.Tensor   # (N,)
    radiance_rgb: torch.Tensor      # (N, 3) (e.g., rec.product)

    # Optional extra signals (if you store them)
    throughput_radiance: torch.Tensor  # (N, 3) optional
    emitted_radiance: torch.Tensor     # (N, 3) optional
    bsdf_weight: torch.Tensor          # (N, 3) optional

    radiance_nee: torch.Tensor         # (N, 3) optional
    direction_nee: torch.Tensor        # (N, 2) or (N, 3) optional

    num_samples: int
    device: str = "cuda"

    @classmethod
    def empty(cls, device: str = "cuda") -> "TrainingBatch":
        z3 = torch.empty(0, 3, device=device)
        z1 = torch.empty(0, device=device)
        zN1 = torch.empty(0, 1, device=device)

        return cls(
            position=z3,
            normal=z3,
            wi=z3,
            wo=z3,
            wo_world=z3,

            roughness=zN1,

            combined_pdf=z1,
            bsdf_pdf=z1,
            guide_pdf=z1,
            sample_source=torch.empty(0, device=device, dtype=torch.int8),
            bsdf_fraction=z1,

            is_delta=torch.empty(0, device=device, dtype=torch.bool),

            radiance_scalar=z1,
            radiance_rgb=z3,

            throughput_radiance=z3,
            emitted_radiance=z3,
            bsdf_weight=z3,

            radiance_nee=z3,
            direction_nee=torch.empty(0, 2, device=device),

            num_samples=0,
            device=device
        )

    def is_valid(self) -> bool:
        return self.num_samples > 0


def _get_attr_torch(rec, name: str, device: str, transpose_if_matrix3: bool = True):
    """Best-effort extraction of rec.<name> to torch tensor on device."""
    if not hasattr(rec, name):
        return None
    t = getattr(rec, name)
    try:
        x = t.torch().to(device)
    except Exception:
        return None
    # Heuristic: (3, N) -> (N, 3)
    if transpose_if_matrix3 and x.ndim == 2 and x.shape[0] == 3:
        return x.t()
    return x


def prepare_shared_training_data(
    integrator,
    device: str = "cuda",
    max_samples: int = -1
) -> TrainingBatch:
    """Prepare training data using zero-copy views where possible."""
    import drjit as dr

    rec = integrator.surfaceInteractionRecord

    integrator.scatter_data_into_buffer()

    num_samples = dr.width(rec.position)
    if num_samples == 0:
        return TrainingBatch.empty(device)

    # Core geometry / directions
    pos = _get_attr_torch(rec, "position", device)
    normal = _get_attr_torch(rec, "normal", device)
    wi = _get_attr_torch(rec, "wi", device)
    wo = _get_attr_torch(rec, "wo", device)
    wo_world = _get_attr_torch(rec, "wo_world", device)

    # PDFs / flags
    combined_pdf = _get_attr_torch(rec, "woPdf", device, transpose_if_matrix3=False)
    bsdf_pdf = _get_attr_torch(rec, "bsdfPdf", device, transpose_if_matrix3=False)
    is_delta = _get_attr_torch(rec, "isDelta", device, transpose_if_matrix3=False)

    guide_pdf = _get_attr_torch(rec, "guidePdf", device, transpose_if_matrix3=False)
    sample_source = _get_attr_torch(rec, "sampleSource", device, transpose_if_matrix3=False)
    bsdf_fraction = _get_attr_torch(rec, "bsdfFraction", device, transpose_if_matrix3=False)

    # Optional (useful for matching C++ "guiding active" gating)
    guiding_active = _get_attr_torch(rec, "guidingActive", device, transpose_if_matrix3=False)
    depth = _get_attr_torch(rec, "depth", device, transpose_if_matrix3=False)
    # Fallbacks if missing
    if guide_pdf is None: guide_pdf = torch.empty(num_samples, device=device)
    if sample_source is None: sample_source = torch.empty(num_samples, device=device, dtype=torch.int8)
    if bsdf_fraction is None: bsdf_fraction = torch.empty(num_samples, device=device)

    # Targets
    radiance_rgb = _get_attr_torch(rec, "product", device)  # (N,3) if stored as (3,N)

    raw_rad = _get_attr_torch(rec, "radiance", device, transpose_if_matrix3=False)
    if raw_rad is None:
        radiance_scalar = torch.empty((num_samples,), device=device)
    elif raw_rad.ndim > 1:
        # if (3,N)
        radiance_scalar = (0.2126 * raw_rad[0] + 0.7152 * raw_rad[1] + 0.0722 * raw_rad[2])
    else:
        radiance_scalar = raw_rad

    # Optional extras
    throughput_radiance = _get_attr_torch(rec, "throughputRadiance", device)
    emitted_radiance = _get_attr_torch(rec, "emittedRadiance", device)
    bsdf_weight = _get_attr_torch(rec, "bsdf", device)

    radiance_nee = _get_attr_torch(rec, "radiance_nee", device)
    direction_nee = _get_attr_torch(rec, "direction_nee", device, transpose_if_matrix3=False)

    # If you don’t store these yet, they’ll just be empty and you can fill later. (was needed for NIS implementation)
    guide_pdf = torch.empty((num_samples,), device=device)
    sample_source = torch.empty((num_samples,), device=device, dtype=torch.int8)
    bsdf_fraction = torch.empty((num_samples,), device=device)

    # Roughness placeholder (or replace with a real roughness field you store)Dummy for now
    roughness = torch.ones((num_samples, 1), device=device)

    # Apply max_samples if requested (simple truncation)
    if max_samples is not None and max_samples > 0 and num_samples > max_samples:
        sl = slice(0, max_samples)
        pos, normal, wi, wo, wo_world = pos[sl], normal[sl], wi[sl], wo[sl], wo_world[sl]
        roughness = roughness[sl]
        combined_pdf, bsdf_pdf = combined_pdf[sl], bsdf_pdf[sl]
        is_delta = is_delta[sl]
        radiance_scalar = radiance_scalar[sl]
        radiance_rgb = radiance_rgb[sl]
        guide_pdf = guide_pdf[sl]
        sample_source = sample_source[sl]
        bsdf_fraction = bsdf_fraction[sl]
        if throughput_radiance is not None: throughput_radiance = throughput_radiance[sl]
        if emitted_radiance is not None: emitted_radiance = emitted_radiance[sl]
        if bsdf_weight is not None: bsdf_weight = bsdf_weight[sl]
        if radiance_nee is not None: radiance_nee = radiance_nee[sl]
        if direction_nee is not None and direction_nee.ndim > 0: direction_nee = direction_nee[sl]
        num_samples = max_samples

    # Fallbacks for missing optionals
    def _or_empty(x, shape_last):
        if x is None:
            return torch.empty((num_samples, shape_last), device=device)
        return x

    throughput_radiance = _or_empty(throughput_radiance, 3)
    emitted_radiance = _or_empty(emitted_radiance, 3)
    bsdf_weight = _or_empty(bsdf_weight, 3)
    radiance_nee = _or_empty(radiance_nee, 3)
    if direction_nee is None:
        direction_nee = torch.empty((num_samples, 2), device=device)

    return TrainingBatch(
        position=pos,
        normal=normal,
        wi=wi,
        wo=wo,
        wo_world=wo_world,

        roughness=roughness,

        combined_pdf=combined_pdf,
        bsdf_pdf=bsdf_pdf,
        guide_pdf=guide_pdf,
        sample_source=sample_source,
        bsdf_fraction=bsdf_fraction,

        is_delta=is_delta.to(torch.bool) if is_delta is not None else torch.empty((num_samples,), device=device, dtype=torch.bool),

        radiance_scalar=radiance_scalar,
        radiance_rgb=radiance_rgb,

        throughput_radiance=throughput_radiance,
        emitted_radiance=emitted_radiance,
        bsdf_weight=bsdf_weight,

        radiance_nee=radiance_nee,
        direction_nee=direction_nee,

        num_samples=num_samples,
        device=device
    )
