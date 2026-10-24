"""Frame-by-frame wavefront renderer, so training can run between samples."""
from __future__ import annotations

from typing import Optional

import drjit as dr
import mitsuba as mi
import torch
from tqdm.auto import tqdm

mi.set_variant("cuda_ad_rgb")
dr.set_log_level(dr.LogLevel.Warn)

from integrators.path_guiding_integrator import PathGuidingIntegrator


class Renderer:
    def render_frame(self, scene: mi.Scene, integrator: mi.SamplingIntegrator) -> mi.Spectrum:
        """Trace one jittered sample per pixel and return the per-pixel radiance."""
        idx = dr.arange(mi.UInt32, self.num_rays)
        pixel = mi.Vector2f(idx % self.res_x, idx // self.res_x)
        film_pos = (pixel + self.sampler.next_2d()) / mi.Vector2f(self.res_x, self.res_y)
        ray, _ = self.sensor.sample_ray(0.0, self.sampler.next_1d(), film_pos, self.sampler.next_2d())
        radiance, valid, _ = integrator.sample(scene, self.sampler, ray)
        return dr.select(valid, radiance, 0.0)

    def render(
        self,
        scene: mi.Scene,
        sensor: Optional[mi.Sensor] = None,
        spp: Optional[int] = None,
        integrator: Optional[mi.SamplingIntegrator] = None,
        progress: bool = True,
        seed: int = 0,
    ) -> torch.Tensor:
        """Mean of `spp` frames as an (H, W, 3) tensor."""
        self.sensor = sensor or scene.sensors()[0]
        self.sampler = self.sensor.sampler()
        self.res_x, self.res_y = self.sensor.film().crop_size()
        self.num_rays = self.res_x * self.res_y
        integrator = integrator or scene.integrator()
        spp = spp or self.sampler.sample_count()

        guided = isinstance(integrator, PathGuidingIntegrator)
        if guided:
            integrator.setup(scene, self.num_rays)
            # The integrator calls into PyTorch inside the path loop, which loop/vcall recording cannot trace
            dr.set_flag(dr.JitFlag.LoopRecord, False)
            dr.set_flag(dr.JitFlag.VCallRecord, False)

        total = torch.zeros((self.num_rays, 3), device="cuda")
        for i in tqdm(range(spp), f"Rendering ({spp} spp)", leave=False) if progress else range(spp):
            self.sampler.seed(i + seed, self.num_rays)
            total += self.render_frame(scene, integrator).torch().T

        if guided:
            dr.set_flag(dr.JitFlag.LoopRecord, True)
            dr.set_flag(dr.JitFlag.VCallRecord, True)
        return (total / spp).reshape((self.res_y, self.res_x, 3))
