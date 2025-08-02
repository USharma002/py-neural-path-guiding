import mitsuba as mi
import drjit as dr
import torch
from tqdm.auto import tqdm
import time

from path_guiding_integrator import PathGuidingIntegrator

mi.set_variant("cuda_ad_rgb")
dr.set_log_level(dr.LogLevel.Warn)

device='cuda'

class Renderer:
	def render_frame(self, scene, integrator):
		# This method is already efficient. No changes needed.
		pixel_indices = dr.arange(mi.UInt32, self.num_rays)
		x = pixel_indices % self.res_x
		y = pixel_indices // self.res_x
		pixel_pos_base = mi.Vector2f(x, y)
		pixel_pos_jittered = (pixel_pos_base + self.sampler.next_2d()) / mi.Vector2f(self.res_x, self.res_y)
		ray, _ = self.sensor.sample_ray(
			0.0, self.sampler.next_1d(), pixel_pos_jittered, self.sampler.next_2d()
		)
		radiance, valid, _ = integrator.sample(scene, self.sampler, ray)
		return dr.select(valid, radiance, 0.0)

	def render(self, scene, sensor=None, spp=None, integrator=None, progress=True, seed=0, progress_setter=None, guiding=False):
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

		accumulated_radiance = torch.zeros((self.num_rays, 3), device=device)
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
	print(f"Setting up integrator for a wavefront of {num_rays} rays.")
	integrator.setup(
		num_rays=num_rays,
		bbox_min=scene.bbox().min,
		bbox_max=scene.bbox().max
	)

	# integrator.set_guiding( False )

	print("\nStarting render process...")
	start_time = time.time()
	
	# This call will now be fast and stable, even with high SPP
	# Note: We pass guiding=False for this example to test the vanilla PT performance
	img = renderer_instance.render(scene, spp=cbox['sensor']['sampler']['sample_count'], integrator=integrator).cpu().numpy()
	
	end_time = time.time()
	print(f"Total rendering time: {end_time - start_time:.2f} seconds.")

	plt.imshow(img ** (1. / 2.2))
	plt.axis("off")
	plt.show()