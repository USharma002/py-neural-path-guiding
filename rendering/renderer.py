"""Custom renderer for neural path guiding with frame-by-frame control."""
from __future__ import annotations

import random
import time
from typing import Optional, Callable

import drjit as dr
import mitsuba as mi
import torch
from tqdm.auto import tqdm

from guiding.config import get_logger
from integrators.path_guiding_integrator import PathGuidingIntegrator

logger = get_logger("renderer")
mi.set_variant("cuda_ad_rgb")
dr.set_log_level(dr.LogLevel.Warn)

DEVICE = "cuda"


class Renderer:
	"""Custom renderer supporting frame-by-frame rendering with neural path guiding."""
	
	def __init__(self):
		# Cache for avoiding redundant setup
		self._cached_scene = None
		self._cached_sensor = None
		self._cached_num_rays = 0
		self._pixel_indices = None
	
	def render_frame(
		self, 
		scene: mi.Scene, 
		integrator: mi.SamplingIntegrator
	) -> mi.Spectrum:
		"""Render a single frame.
		
		Args:
			scene: The Mitsuba scene to render
			integrator: The integrator to use
			
		Returns:
			Radiance values for all pixels
		"""
		# Use cached pixel indices if available
		if self._pixel_indices is None or len(self._pixel_indices) != self.num_rays:
			self._pixel_indices = dr.arange(mi.UInt32, self.num_rays)
		
		x = self._pixel_indices % self.res_x
		y = self._pixel_indices // self.res_x
		pixel_pos_base = mi.Vector2f(x, y)
		pixel_pos_jittered = (pixel_pos_base + self.sampler.next_2d()) / mi.Vector2f(self.res_x, self.res_y)
		ray, _ = self.sensor.sample_ray(
			0.0, self.sampler.next_1d(), pixel_pos_jittered, self.sampler.next_2d()
		)
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
		progress_setter: Optional[Callable[[int], None]] = None, 
		guiding: bool = False
	) -> torch.Tensor:
		"""Render the scene with the given parameters.
		
		Args:
			scene: The Mitsuba scene to render
			sensor: Optional sensor override
			spp: Samples per pixel
			integrator: Optional integrator override
			progress: Whether to show progress bar
			seed: Random seed for reproducibility
			progress_setter: Optional callback for progress updates
			guiding: Whether guiding is enabled
			
		Returns:
			Rendered image as a PyTorch tensor of shape (H, W, 3)
		"""
		self.sensor = scene.sensors()[0] if sensor is None else sensor
		self.film = self.sensor.film()
		self.sampler = self.sensor.sampler()

		film_size = self.film.crop_size()
		self.res_x, self.res_y = film_size
		self.num_rays = self.res_x * self.res_y

		if hasattr(integrator, 'setup'):
			integrator.setup(
				num_rays=self.num_rays,
				bbox_min=scene.bbox().min,
				bbox_max=scene.bbox().max
			)

		if spp is None:
			spp = self.sampler.sample_count()
		if integrator is None:
			integrator = scene.integrator()

		accumulated_radiance = torch.zeros((self.num_rays, 3), device=DEVICE)
		pbar = tqdm(range(spp), f"Rendering ({spp} spp)", leave=False) if progress else range(spp)
		
		# When guiding, we must disable loop recording.
		# When NOT guiding, we WANT loop recording for maximum performance.
		is_complex_integrator = isinstance(integrator, PathGuidingIntegrator)
		if is_complex_integrator:
			dr.set_flag(dr.JitFlag.LoopRecord, False)
			dr.set_flag(dr.JitFlag.VCallRecord, False)
		
		for i in pbar:
			if progress_setter:
				progress_setter(i * 100 // spp)
			
			if hasattr(integrator, 'set_iteration'):
				integrator.set_iteration(i + 1)

			self.sampler.seed(i + seed, self.num_rays)
			
			frame = self.render_frame(scene, integrator)
			
			accumulated_radiance += frame.torch()

		# Restore JIT flags to default state after rendering is complete.
		if is_complex_integrator:
			dr.set_flag(dr.JitFlag.LoopRecord, True)
			dr.set_flag(dr.JitFlag.VCallRecord, True)

		if spp > 0:
			final_image_torch = accumulated_radiance/ spp
		else:
			final_image_torch = accumulated_radiance
			
		final_image_reshaped = final_image_torch.reshape((self.res_y, self.res_x, 3))
		
		return final_image_reshaped.detach()


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	cbox = mi.cornell_box()
	cbox['integrator']['max_depth'] = 5
	cbox['integrator']['rr_depth'] = 5
	cbox['sensor']['sampler']['sample_count'] = 1
	cbox['sensor']['film']['width'] = 400
	cbox['sensor']['film']['height'] = 400

	scene = mi.load_dict(cbox)
	
	# We create one instance of the renderer to use.
	renderer_instance = Renderer()

	integrator = mi.load_dict({
		'type': 'path_guiding_integrator',
		'max_depth': cbox['integrator']['max_depth']
	})

	num_rays = cbox['sensor']['film']['width'] * cbox['sensor']['film']['height']
	logger.info(f"Setting up integrator for a wavefront of {num_rays} rays.")
	integrator.setup(
		num_rays=num_rays,
		bbox_min=scene.bbox().min,
		bbox_max=scene.bbox().max
	)

	# integrator.set_guiding( False )

	logger.info("Starting render process...")
	start_time = time.time()
	
	# This call will now be fast and stable, even with high SPP
	# Note: We pass guiding=False for this example to test the vanilla PT performance
	img = renderer_instance.render(scene, spp=cbox['sensor']['sampler']['sample_count'], integrator=integrator).cpu().numpy()
	
	end_time = time.time()
	logger.info(f"Total rendering time: {end_time - start_time:.2f} seconds.")

	if hasattr(integrator, 'nrc_system'):
		loss = integrator.nrc_system.train_step(integrator)
		logger.info(f"NRC Loss: {loss}")
	
	# plt.imshow(img ** (1. / 2.2))
	# plt.axis("off")
	# plt.show()