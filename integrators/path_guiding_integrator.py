"""Path tracer that mixes BSDF and learned guiding samples and records every vertex for training.

Each guiding method subclasses PathGuidingIntegrator to bind its distribution and,
optionally, to record extra per-vertex data and build its own training batch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import drjit as dr
import mitsuba as mi
import torch

mi.set_variant("cuda_ad_rgb")
DEVICE = "cuda"

from guiding.training_data import TrainingBatch
from networks.nrc import NeuralRadianceCache
import sensor.spherical  # noqa: F401  (registers the hemisphere sensor)


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """Power heuristic (beta = 2)."""
    a2 = dr.square(pdf_a)
    b2 = dr.square(pdf_b)
    w  = dr.select(pdf_a > 0, a2 / (a2 + b2), 0.0)
    return dr.select(dr.isfinite(w), w, 0.0)


def _to_torch(x) -> torch.Tensor:
    """Dr.Jit array -> torch; vectors come out as (3, N) and are transposed to (N, 3)."""
    t = x.torch()
    return t.t() if t.ndim == 2 else t


class PathGuidingIntegrator(mi.SamplingIntegrator):
    """Guiding is one-sample MIS: wo comes from the BSDF with probability bsdfSamplingFraction, else from the guide.

    Subclass hooks:
      distribution_class / config_class  the guiding method
      EXTRA_RECORD_FIELDS                 per-vertex values to record beyond RECORD_FIELDS;
                                          names are keys of the `vertex` dict built in sample()
      record_extras()                     scatter those values (override for custom per-vertex work)
      batch_class / batch_extras()        training batch type and the extra fields it needs
    """

    distribution_class: Optional[Type] = None
    config_class: Optional[Type] = None

    # Per-vertex record, indexed by ray_index * max_depth + depth
    RECORD_FIELDS: Dict[str, type] = {
        "position": mi.Vector3f,
        "normal": mi.Vector3f,
        "wi": mi.Vector3f,                  # local
        "wo": mi.Vector3f,                  # local
        "bsdf": mi.Color3f,                 # f * cos / woPdf (and 1 / rr_prob past rr_depth)
        "emittedRadiance": mi.Spectrum,     # MIS-weighted emission, as used by the path estimate
        "emittedRaw": mi.Spectrum,          # unweighted emission
        "radiance_nee": mi.Color3f,
        "product": mi.Spectrum,             # outgoing radiance toward the previous vertex (NRC target)
        "incident": mi.Spectrum,            # incident radiance L_i(x, wo) from the next vertex
        "woPdf": mi.Float,                  # MIS mixture pdf that generated wo
        "active": mi.Bool,
    }
    EXTRA_RECORD_FIELDS: Dict[str, type] = {}

    batch_class: Type[TrainingBatch] = TrainingBatch
    BSDF_SAMPLING_FRACTION: float = 0.5   # probability of sampling wo from the BSDF when guiding
    TRAINING_BUDGET_PERCENT: int = 20     # share of a render budget spent training (ground-truth export)

    def __init__(self, props: mi.Properties = mi.Properties(), nrc_system: Optional[NeuralRadianceCache] = None) -> None:
        super().__init__(props)

        self.max_depth: int = props.get('max_depth', 5)
        self.rr_depth: int = props.get('rr_depth', 5)
        self.num_rays: int = 0

        self.bsdfSamplingFraction: float = self.BSDF_SAMPLING_FRACTION
        self.guiding: bool = False  # Python bool so the guiding branch is skipped at trace time
        self.use_nee: bool = props.get('use_nee', False)

        self.nrc_system = nrc_system or NeuralRadianceCache(device=DEVICE)
        self.nrc_query_prob: float = props.get('nrc_query_prob', 0.8)
        self.use_nrc: bool = False
        self.nrc_min_depth: int = props.get('nrc_min_depth', 2)

        self.guide = self.distribution_class(self.config_class(device=DEVICE))

        self.record_type = type("SurfaceInteractionRecord", (),
                                {"DRJIT_STRUCT": {**self.RECORD_FIELDS, **self.EXTRA_RECORD_FIELDS}})
        self.scene = None
        self.surfaceInteractionRecord = None
        self._data_scattered = False

    def setup(self, scene: mi.Scene, num_rays: int) -> None:
        """Bind a scene and wavefront size; no-op if neither changed (the renderer calls this every frame)."""
        if scene is self.scene and num_rays == self.num_rays:
            return
        self.num_rays = num_rays
        self.array_size = self.num_rays * self.max_depth
        self.scene = scene

        self.guide.set_scene(scene)
        self.nrc_system.set_scene(scene)

        self.resetRayPathData()

    def set_guiding(self, guiding_active: bool = False) -> None:
        self.guiding = bool(guiding_active)

    def set_nrc(self, nrc_active: bool = False) -> None:
        """Terminate paths into the NRC (with probability nrc_query_prob past nrc_min_depth)."""
        self.use_nrc = bool(nrc_active)

    @staticmethod
    def _roughness(n: int) -> torch.Tensor:
        return torch.ones((n, 1), device=DEVICE)  # roughness is not recorded

    @dr.wrap(source='drjit', target='torch')
    def sample_guided_direction(self, position: torch.Tensor, wi: torch.Tensor,
                                normal: torch.Tensor) -> Tuple[mi.Vector3f, mi.Float]:
        with torch.no_grad():
            roughness = self._roughness(position.shape[1])
            wo_pred, pdf_val = self.guide.sample(position.T, wi.T, normal.T, roughness)
            return mi.Vector3f(wo_pred.T), mi.Float(pdf_val.squeeze())

    @dr.wrap(source='drjit', target='torch')
    def guiding_pdf(self, position: torch.Tensor, wi: torch.Tensor,
                    normal: torch.Tensor, wo: torch.Tensor) -> mi.Float:
        with torch.no_grad():
            roughness = self._roughness(position.shape[1])
            pdf_val = self.guide.pdf(position.T, wi.T, normal.T, roughness, wo.T)
            return mi.Float(pdf_val.squeeze())

    @dr.wrap(source='drjit', target='torch')
    def query_nrc(self, position: torch.Tensor, normal: torch.Tensor, view_dir: torch.Tensor, roughness: torch.Tensor) -> mi.Spectrum:
        with torch.no_grad():
            pred_radiance = self.nrc_system.query(position.T, normal.T, view_dir.T, roughness)
            return mi.Spectrum(pred_radiance.T)

    # --- Recording ---

    def resetRayPathData(self) -> None:
        self.surfaceInteractionRecord = dr.zeros(self.record_type, shape=self.array_size)
        self._data_scattered = False

    def record_extras(self, vertex: Dict[str, Any], index: mi.UInt32, active: mi.Bool) -> None:
        rec = self.surfaceInteractionRecord
        for name in self.EXTRA_RECORD_FIELDS:
            dr.scatter(getattr(rec, name), value=vertex[name], index=index, active=active)

    def process_incoming_radiance(self) -> None:
        """Walk each path backwards to fill rec.product (outgoing radiance) and rec.incident (L_i along wo)."""
        rec = self.surfaceInteractionRecord
        array_idx = dr.arange(mi.UInt32, self.array_size)

        rec.product = rec.emittedRadiance * 0
        rec.incident = rec.emittedRadiance * 0

        ray_ids = array_idx // self.max_depth
        for d in reversed(range(self.max_depth)):
            mask = rec.active & (array_idx % self.max_depth == d)
            if not dr.any(mask):
                continue
            Lo = dr.gather(mi.Spectrum, rec.emittedRadiance, array_idx, mask) + \
                dr.gather(mi.Spectrum, rec.radiance_nee, array_idx, mask)
            if d + 1 < self.max_depth:
                nxt = ray_ids * self.max_depth + (d + 1)
                Lo_next = dr.gather(mi.Spectrum, rec.product, nxt, mask)
                Lo += dr.gather(mi.Spectrum, rec.bsdf, array_idx, mask) * Lo_next
                # L_i along wo is the next vertex's outgoing radiance with its emission counted in full:
                # the MIS weight on emission only matters for the pixel estimate, where NEE covers the rest.
                L_i = Lo_next - dr.gather(mi.Spectrum, rec.emittedRadiance, nxt, mask) + \
                    dr.gather(mi.Spectrum, rec.emittedRaw, nxt, mask)
                dr.scatter(rec.incident, L_i, array_idx, mask)
            dr.scatter(rec.product, Lo, array_idx, mask)

    def scatter_data_into_buffer(self) -> None:
        """Compact the record to vertices with a valid pdf and non-zero outgoing or incident radiance."""
        if self._data_scattered:
            return

        rec = self.surfaceInteractionRecord
        rec.product[dr.isnan(rec.product)] = 0
        rec.incident[dr.isnan(rec.incident)] = 0
        rec.radiance_nee[dr.isnan(rec.radiance_nee)] = 0

        has_radiance = (mi.luminance(rec.product) != 0) | (mi.luminance(rec.incident) != 0)
        keep = rec.active & has_radiance & (rec.woPdf != 0) & ~dr.isnan(rec.woPdf)
        activeIndex = dr.compress(keep)

        for name in self.record_type.DRJIT_STRUCT:
            value = getattr(rec, name)
            setattr(rec, name, dr.gather(type(value), value, activeIndex))

        self._data_scattered = True

    # --- Training batch ---

    def batch_extras(self, field: Callable[[str], torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
        """Extra batch_class fields; `field(name)` reads a compacted record field as a tensor."""
        return {}

    def make_batch(self, max_samples: int = -1) -> TrainingBatch:
        self.scatter_data_into_buffer()
        rec = self.surfaceInteractionRecord

        n = dr.width(rec.position)
        if n == 0:
            return self.batch_class.empty()

        field = lambda name: _to_torch(getattr(rec, name))
        tensors = dict(
            position=field("position"),
            normal=field("normal"),
            wi=field("wi"),
            wo=field("wo"),
            roughness=self._roughness(n),
            combined_pdf=field("woPdf"),
            guide_weight_rgb=field("bsdf") * field("incident"),
            outgoing_rgb=field("product"),
            **self.batch_extras(field, n),
        )
        if 0 < max_samples < n:
            tensors = {k: v[:max_samples] for k, v in tensors.items()}
            n = max_samples
        return self.batch_class(**tensors, num_samples=n)

    def train_step(self, batch: TrainingBatch, train_guide: bool = True,
                   train_nrc: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """One optimization step on a batch; returns (guide loss, NRC loss), None where not trained."""
        nrc_loss = self.nrc_system.train_step_from_batch(batch) if train_nrc else None
        guide_loss = self.guide.train_step(batch) if train_guide else None
        return guide_loss, nrc_loss

    # --- Transport ---

    @dr.syntax
    def sample(
        self,
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

            wi_local = si.to_local(-ray.d)

            active_nrc = mi.Bool(False)
            if self.use_nrc:
                active_nrc = active & si.is_valid() & (depth >= self.nrc_min_depth) & (sampler.next_1d(active) < self.nrc_query_prob)
                nrc_rad = self.query_nrc(si.p, si.n, wi_local, self._roughness(dr.width(ray)))
                L += dr.select(active_nrc, β * nrc_rad, 0.0)
                active &= ~active_nrc

            # Emission hit by the previous bounce
            ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)
            raw_Le = ds_direct.emitter.eval(si)

            if self.use_nee:
                mis_bsdf_light = mis_weight(prev_bsdf_pdf, emitter_pdf)
            else:
                mis_bsdf_light = mi.Float(1.0)

            Le = dr.select(active, β * mis_bsdf_light * raw_Le, 0.0)
            active_next = active & (depth + 1 < self.max_depth) & si.is_valid()

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)
            bsdf_pdf = bsdf_sample.pdf
            bsdf_value = bsdf_weight * bsdf_pdf
            woPdf = mi.Float(bsdf_pdf)
            wo_local = mi.Vector3f(bsdf_sample.wo)
            wo_world = si.to_world(wo_local)
            delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            do_guiding_bsdf_mis = active_next & ~delta & dr.all(dr.isfinite(si.p)) & mi.Bool(self.guiding)

            # Next event estimation
            Lnee = mi.Color3f(0.0)

            is_smooth = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            active_em = active_next & is_smooth & self.use_nee
            Lr_dir = mi.Spectrum(0)

            # Branch on the Python flag, not the mask: a symbolic branch would force the
            # emitter/BSDF virtual calls into symbolic mode while vcall recording is off.
            if self.use_nee:
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
                active_em &= ds.pdf != 0.0
                wo_em = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_em, active_em)

                # Weight against the same mixture density used for BSDF/guide-sampled hits (woPdf)
                pdf_hemisphere_em = mi.Float(bsdf_pdf_em)
                if self.guiding:
                    guide_pdf_em = self.guiding_pdf(si.p, wi_local, si.n, wo_em)
                    pdf_hemisphere_em[do_guiding_bsdf_mis] = (
                        self.bsdfSamplingFraction * bsdf_pdf_em + (1 - self.bsdfSamplingFraction) * guide_pdf_em
                    )

                mis_em = dr.select(ds.delta, mi.Float(1.0), mis_weight(ds.pdf, pdf_hemisphere_em))
                Lr_dir = dr.select(active_em, β * mis_em * bsdf_value_em * em_weight, 0.0)

                Lnee = mis_em * bsdf_value_em * em_weight

            # One-sample MIS: take the guide's direction with probability 1 - bsdfSamplingFraction
            use_guide_sample = do_guiding_bsdf_mis & (sampler.next_1d(active_next) > self.bsdfSamplingFraction)
            guide_pdf = mi.Float(0.0)

            if self.guiding:
                g_dir, g_pdf = self.sample_guided_direction(si.p, wi_local, si.n)

                wo_world[use_guide_sample] = si.to_world(g_dir)
                wo_local[use_guide_sample] = g_dir
                bsdf_value_g, bsdf_pdf_g = bsdf.eval_pdf(bsdf_ctx, si, wo_local, use_guide_sample)
                bsdf_value[use_guide_sample] = bsdf_value_g
                bsdf_pdf[use_guide_sample] = bsdf_pdf_g

                guide_pdf = dr.select(use_guide_sample, g_pdf, self.guiding_pdf(si.p, wi_local, si.n, wo_local))

                woPdf[do_guiding_bsdf_mis] = self.bsdfSamplingFraction * bsdf_pdf + (1 - self.bsdfSamplingFraction) * guide_pdf
                bsdf_weight[do_guiding_bsdf_mis] = bsdf_value / woPdf

            L = L + Le + Lr_dir

            globalIndex = (ray_index * self.max_depth) + depth
            storeFlag = active & si.is_valid()

            dr.scatter(self.surfaceInteractionRecord.position, value=mi.Vector3f(si.p), index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wi, value=wi_local, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.wo, value=wo_local, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.normal, value=mi.Vector3f(si.n), index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.active, value=storeFlag, index=globalIndex, active=storeFlag)
            # mis * Le directly: Le / β would be 0/0 wherever a throughput channel is zero
            dr.scatter(self.surfaceInteractionRecord.emittedRadiance, value=mis_bsdf_light * raw_Le, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.emittedRaw, value=raw_Le, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.woPdf, value=woPdf, index=globalIndex, active=storeFlag)
            dr.scatter(self.surfaceInteractionRecord.radiance_nee, value=Lnee, index=globalIndex, active=storeFlag & active_em)

            if len(self.EXTRA_RECORD_FIELDS) > 0:
                self.record_extras(dict(
                    bsdfPdf=bsdf_pdf,
                    isDelta=delta,
                    guidePdf=dr.select(do_guiding_bsdf_mis, guide_pdf, mi.Float(0.0)),
                    sampleSource=dr.select(use_guide_sample, mi.UInt32(1), mi.UInt32(0)),
                    wo_world=wo_world,
                    throughputRadiance=L,
                    depth=mi.UInt32(depth),
                ), globalIndex, storeFlag)

            ray = si.spawn_ray(wo_world)
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

            # Recorded throughput must include the Russian-roulette compensation, like β
            rr_weight = dr.select(rr_active & (rr_prob > 0), dr.rcp(rr_prob), 1.0)
            dr.scatter(self.surfaceInteractionRecord.bsdf, value=bsdf_weight * rr_weight, index=globalIndex, active=storeFlag)

            depth[active & si.is_valid()] += 1
            active = active_next

        dr.schedule(L)
        sampler.schedule_state()
        dr.schedule(self.surfaceInteractionRecord)

        self.process_incoming_radiance()

        return (L, depth != 0, [])


def _create_default(props: mi.Properties) -> PathGuidingIntegrator:
    import guiding.registry as registry
    return registry.create_integrator(registry.get_default_method(), props)


mi.register_integrator("path_guiding_integrator", _create_default)
