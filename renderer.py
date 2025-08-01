import mitsuba as mi
import drjit as dr
import torch

from tqdm.auto import tqdm
from path_guiding_integrator import PathGuidingIntegrator
import matplotlib.pyplot as plt
import time 
import random

device = "cuda"

mi.set_variant("cuda_ad_rgb")
dr.set_log_level(dr.LogLevel.Warn)

class Renderer:
	def __init__(self):
		self.guiding = False

	def render_frame(self, scene, integrator):
		# This method is perfect. No changes needed.
		pixel_indices = dr.arange(mi.UInt32, self.num_rays)
		x = pixel_indices % self.res_x
		y = pixel_indices // self.res_x
		pixel_pos_base = mi.Vector2f(x, y)
		pixel_pos_jittered = (pixel_pos_base + self.sampler.next_2d()) / mi.Vector2f(self.res_x, self.res_y)
		ray, _ = self.sensor.sample_ray(
			0.0, self.sampler.next_1d(), pixel_pos_jittered, self.sampler.next_2d()
		)
		radiance, valid, _ = integrator.sample(scene, self.sampler, ray)
		valid_radiance = dr.select(valid, radiance, 0.0)
		return valid_radiance

	def render(self, scene, sensor = None, spp = None, integrator = None, progress=True, seed = 0, progress_setter=None, guiding=False):
		self.sensor = scene.sensors()[0] if sensor is None else sensor
		self.film = self.sensor.film()
		self.sampler = self.sensor.sampler()

		film_size = self.film.crop_size()

		self.res_x, self.res_y = film_size
		self.num_rays = self.res_x * self.res_y

		if spp is None:
			spp = self.sampler.sample_count()
		if integrator is None:
			integrator = scene.integrator()

		accumulated_radiance = 0
		pbar = tqdm(range(spp), f"Rendering ({spp} spp)", leave=False) if progress else range(spp)
		integrator.set_iteration( 0 )
		if guiding:
			dr.set_flag(dr.JitFlag.LoopRecord, False)
			dr.set_flag(dr.JitFlag.VCallRecord, False)
		
		for i in pbar:
			if progress_setter:
				progress_setter( i * 100 // spp )
			
			if guiding:
				integrator.set_iteration(i + 1)
			else:
				integrator.set_iteration( 0 )

			self.sampler.seed(i + random.randint(0, 2**32) * seed, self.num_rays)
			frame = self.render_frame(scene, integrator)
			accumulated_radiance += frame.torch()			

		dr.set_flag(dr.JitFlag.LoopRecord, True)
		dr.set_flag(dr.JitFlag.VCallRecord, True)
		# Now, this part will be fast because the computation graph is small and efficient.
		final_image = accumulated_radiance / spp if spp > 0 else accumulated_radiance
		final_image_reshaped = final_image.reshape((self.res_y, self.res_x, 3))
		return final_image_reshaped.cpu().detach()


if __name__ == "__main__":
	cbox = mi.cornell_box()
	cbox['integrator']['max_depth'] = 5
	cbox['integrator']['rr_depth'] = 5
	cbox['sensor']['sampler']['sample_count'] = 100
	cbox['sensor']['film']['width'] = 400
	cbox['sensor']['film']['height'] = 400

	scene = mi.load_dict(cbox)
	
	mi.render = Renderer().render

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

	integrator.guided_direction = True
	
	print("\nStarting render process...")
	start_time = time.time()
	
	# This call will now be fast and stable, even with high SPP
	img = mi.render(scene, spp=cbox['sensor']['sampler']['sample_count'], integrator=integrator)
	
	end_time = time.time()
	print(f"Total rendering time: {end_time - start_time:.2f} seconds.")

	plt.imshow(img ** (1. / 2.2))
	plt.axis("off")
	plt.show()