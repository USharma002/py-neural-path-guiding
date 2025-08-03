# Standard Library Imports
import os
import io
import random
import sys
import time

# Third-Party Imports
import drjit as dr
import matplotlib.cm as cm
import mitsuba as mi
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from copy import deepcopy

# PyQt6 Imports
from PyQt6.QtCore import QTimer, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QColor, QKeyEvent
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget, QCheckBox, QDialog,
    QSpinBox, QFormLayout, QDialogButtonBox 
)

# Local Project Imports
from path_guiding_integrator import PathGuidingIntegrator
from renderer import Renderer
from spherical import GTSphericalCamera
from vmf_mixture import MixedSphericalGaussianDistribution
from visualization_helpers import (
    visualize_pdf_on_sphere_html, plot_non_batched_mixture,
    numpy_to_qpixmap, draw_marker_on_image
)

mi.set_variant('cuda_ad_rgb')
mi.set_log_level(mi.LogLevel.Warn)
device = "cuda"

custom_render = Renderer().render


class SaveGtDialog(QDialog):
    """A dialog to get resolution, SPP, and comparison type from the user."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Ground Truth Options")

        layout = QFormLayout(self)

        # --- Online/Offline Mode ---
        self.online_checkbox = QCheckBox("Equal Time (Online) Comparison")
        self.online_checkbox.toggled.connect(self.toggle_mode)
        layout.addRow(self.online_checkbox)

        # --- Offline Settings (SPP) ---
        self.spp_label = QLabel("Samples per Pixel (SPP):")
        self.spp_spin = QSpinBox()
        self.spp_spin.setRange(1, 65536)
        self.spp_spin.setValue(256)
        layout.addRow(self.spp_label, self.spp_spin)

        # --- Online Settings (Time) ---
        self.time_label = QLabel("Time Budget (seconds):")
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 3600)
        self.time_spin.setValue(60)
        layout.addRow(self.time_label, self.time_spin)
        
        # --- Common Settings (Resolution) ---
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 8192)
        self.width_spin.setValue(1024)
        layout.addRow("Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 8192)
        self.height_spin.setValue(1024)
        layout.addRow("Height:", self.height_spin)
        
        # --- Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initialize UI state
        self.toggle_mode(self.online_checkbox.isChecked())

    def toggle_mode(self, is_online):
        """Shows/hides settings based on the comparison type."""
        self.time_label.setVisible(is_online)
        self.time_spin.setVisible(is_online)
        self.spp_label.setVisible(not is_online)
        self.spp_spin.setVisible(not is_online)

    def getValues(self):
        """Returns the selected values from the spin boxes."""
        return {
            "online": self.online_checkbox.isChecked(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "spp": self.spp_spin.value(),
            "time": self.time_spin.value()
        }


class MitsubaViewer(QMainWindow):
    """
    An interactive PyQt application for visualizing Mitsuba 3 scenes,
    inspecting BSDFs, and training a path guiding neural network.
    """
    MAIN_RESOLUTION = (256, 256)
    HOVER_RESOLUTION = (256, 256)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mitsuba Path Guiding Viewer")

        # --- State Variables ---
        self.scene = None
        self.integrator = None
        self.guiding_system = None
        self.main_sensor = None
        self.hover_sensor = None
        self.original_main_image_np = None
        self.last_click_coords = None
        self.is_rendering = False

        # --- Initialization ---
        self._init_ui()
        self._init_renderer()
        self._connect_signals()

    # -------------------------------------------------------------------------
    # --- Initialization and Setup
    # -------------------------------------------------------------------------

    def _init_ui(self):
        """Creates and arranges all UI widgets."""
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # --- Top Controls ---
        top_controls_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Scene")
        self.train_button = QPushButton("Train 100 Steps")
        self.save_gt_button = QPushButton("Save GT Images...") 
        self.clear_button = QPushButton("Clear Selection (Esc)")
        top_controls_layout.addWidget(self.browse_button)
        top_controls_layout.addWidget(self.train_button)
        top_controls_layout.addWidget(self.save_gt_button)
        top_controls_layout.addWidget(self.clear_button) 
        self.main_layout.addLayout(top_controls_layout)

        # --- Checkboxes ---
        checkbox_layout = QHBoxLayout()
        self.cosine_checkbox = QCheckBox("Cosine Product")
        self.bsdf_checkbox = QCheckBox("BSDF Product")
        self.guiding_checkbox = QCheckBox("Use Path Guiding")
        self.realtime_checkbox = QCheckBox("Real-Time Render")
        self.train_realtime_checkbox = QCheckBox("Train in Real-Time")
        self.accumulate_checkbox = QCheckBox("Accumulate Frames")
        self.bsdf_checkbox.setChecked(True)
        for widget in [self.cosine_checkbox, self.bsdf_checkbox, self.guiding_checkbox, self.realtime_checkbox, self.train_realtime_checkbox, self.accumulate_checkbox]:
            checkbox_layout.addWidget(widget)
        self.main_layout.addLayout(checkbox_layout)

        # --- Main Visualization Area ---
        vis_layout = QHBoxLayout()
        self.main_view = QLabel("Click 'Browse' to load a scene")
        self.hover_view = QLabel("Click render to inspect point\n(Viridis PDF)")
        self.normalized_pdf_view = QLabel("Ground Truth (sRGB)")
        self.vmf_view = QLabel("Guiding Distribution")
        self.vmf_sphere_view = QWebEngineView()
        
        # Configure QLabels
        for view in [self.main_view, self.hover_view, self.normalized_pdf_view, self.vmf_view]:
            view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Configure QWebEngineView separately
        self.vmf_sphere_view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
        self.vmf_sphere_view.page().setBackgroundColor(QColor("black"))
        
        vis_layout.addWidget(self.main_view, stretch=1)
        vis_layout.addWidget(self.hover_view)
        vis_layout.addWidget(self.normalized_pdf_view)
        vis_layout.addWidget(self.vmf_view)
        vis_layout.addWidget(self.vmf_sphere_view)
        self.main_layout.addLayout(vis_layout)

        # Timer for real-time rendering
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(100)  # ms

    def _reset_accumulator(self):
        """Resets the accumulation buffer and SPP counter."""
        print("--- Accumulator Reset ---")
        h, w = self.MAIN_RESOLUTION[1], self.MAIN_RESOLUTION[0]
        # Initialize on the GPU for fast accumulation
        self.accumulated_image = np.zeros((h, w, 3))
        self.spp_counter = 0

    def _init_renderer(self):
        """Loads the default scene and sets up the Mitsuba integrator."""
        self.integrator = mi.load_dict({"type": "path_guiding_integrator"})
        self.guiding_system = self.integrator.guiding_system
        self.load_default_scene()

    def _connect_signals(self):
        """Connects all widget signals to their corresponding slots."""
        self.browse_button.clicked.connect(self.load_scene_from_file)
        self.train_button.clicked.connect(self.train_step_clicked)
        self.save_gt_button.clicked.connect(self.save_ground_truth_images)
        self.clear_button.clicked.connect(self.clear_selection)
        
        self.cosine_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.bsdf_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.guiding_checkbox.stateChanged.connect(lambda: self.re_render_main_image(spp=1))
        self.realtime_checkbox.stateChanged.connect(self.toggle_realtime_render)
        self.accumulate_checkbox.stateChanged.connect(self._reset_accumulator)


        self.render_timer.timeout.connect(self.realtime_render_loop)
        
        self.main_view.mousePressEvent = self.on_mouse_click

    # -------------------------------------------------------------------------
    # --- Core Logic & Rendering
    # -------------------------------------------------------------------------

    def _transform_light(self, scene_dict):
        """
        Applies a hard-coded transformation to the default Cornell Box light
        to flip its orientation and move it down slightly.
        """
        try:
            original_transform = np.array(scene_dict['light']['to_world'].matrix)
            original_position = original_transform[:3, 3]
            # Matrix to flip orientation around X-axis
            flip_x = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
            flipped_transform = flip_x @ original_transform
            # Move light down
            flipped_transform[:3, 3] = original_position - np.array([0, 0.2, 0])
            scene_dict['light']['to_world'] = mi.ScalarTransform4f(flipped_transform)
        except KeyError:
            print("Warning: Could not find 'light' in scene dictionary to transform.")
        return scene_dict

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key press events for the main window."""
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    def clear_selection(self):
        """Resets the UI after a point was selected."""
        if self.last_click_coords is None:
            return

        print("Clearing selected point.")
        self.last_click_coords = None

        # Redraw the main view without the marker
        self._update_main_view_with_marker()

        # Reset the detail views to their initial placeholder text
        self.hover_view.setText("Click render to inspect point\n(Viridis PDF)")
        self.normalized_pdf_view.setText("Ground Truth (sRGB)")
        self.vmf_view.setText("Guiding Distribution")
        self.vmf_sphere_view.setHtml("") # Clear the 3D sphere visualization

    def load_scene(self, scene_path=None):
        """Loads a scene from a file path or creates a default Cornell Box."""
        try:
            if scene_path:
                self.scene = mi.load_file(scene_path)
            else:
                cbox = mi.cornell_box()
                cbox['integrator']['max_depth'] = 5
                cbox['sensor']['film']['width'], cbox['sensor']['film']['height'] = self.MAIN_RESOLUTION
                self._transform_light(cbox)

                self.scene = mi.load_dict(cbox)

            self.main_sensor = self.scene.sensors()[0]
            sensor_params = mi.traverse(self.main_sensor)
            sensor_params["film.size"] = self.MAIN_RESOLUTION
            sensor_params.update()
            
            num_rays = self.MAIN_RESOLUTION[0] * self.MAIN_RESOLUTION[1]
            print(f"Setting up integrator for a wavefront of {num_rays} rays.")
            
            print(self.scene.bbox().min, self.scene.bbox().max)
            self.integrator.setup(
                num_rays=num_rays,
                bbox_min=self.scene.bbox().min,
                bbox_max=self.scene.bbox().max
            )
            self.re_render_main_image(spp=16)

        except Exception as e:
            print(f"Failed to load scene: {e}")
            self.main_view.setText(f"Failed to load scene:\n{e}")

    def load_default_scene(self):
        self.load_scene(scene_path=None)
        
    def load_scene_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mitsuba Scene", "", "XML Files (*.xml)")
        if file_path:
            self.load_scene(file_path)

    def _render_for_duration(self, sensor, duration, is_online_guided):
        """
        Renders and optionally trains for a fixed duration, correctly accumulating
        the results of every frame.
        """
        spp_counter = 0
        
        # Get resolution from the sensor's film
        film_size = sensor.film().size()
        width, height = film_size[0], film_size[1]
        
        # Initialize the accumulator on the GPU for fast addition
        accumulated_image = torch.zeros((height, width, 3), device=device)

        # In online mode, the guiding system starts fresh for a fair comparison
        if is_online_guided:
            self.guiding_system = self.integrator.guiding_system.__class__()
            self.integrator.guiding_system = self.guiding_system
        
        print(f"\nStarting {'online guided' if is_online_guided else 'unguided'} render for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:
            if is_online_guided:
                # --- Online Guided Path ---
                # 1. Gather unbiased training data (guiding OFF)
                # self.integrator.set_guiding(False)
                # # This render is *only* for data collection, its result is not accumulated.
                # custom_render(self.scene, sensor=sensor, spp=1, integrator=self.integrator,
                #               seed=random.randint(0, 100000), guiding=False, progress=False)
                
                # # 2. Train the network on that data
                # self.guiding_system.train_step(self.integrator)

                # 3. Render the frame we will actually see, using the updated guide
                self.integrator.set_guiding(True)

                # The custom_render function returns a reshaped PyTorch tensor, we need to flatten it
                frame = custom_render(self.scene, sensor=sensor, spp=1, integrator=self.integrator,
                                           seed=random.randint(0, 100000), guiding=True, progress=False)
                self.guiding_system.train_step(self.integrator)

            else:
                # --- Unguided Path ---
                self.integrator.set_guiding(False)
                frame = custom_render(self.scene, sensor=sensor, spp=1, integrator=self.integrator,
                                           seed=random.randint(0, 100000), guiding=False, progress=False)

            accumulated_image += frame
            spp_counter += 1
        
        if spp_counter > 0:
            final_image = accumulated_image / spp_counter
        else:
            final_image = accumulated_image

        print(f"Rendered {spp_counter} SPP in ~{duration} seconds.")

        return final_image.detach().cpu().numpy()

    def realtime_render_loop(self):
        """Performs one step of rendering and training."""
        if self.is_rendering: return
        self.is_rendering = True
        try:
            use_guiding = self.guiding_checkbox.isChecked()
            self.integrator.set_guiding(use_guiding)
            
            # Render one frame
            image_np = custom_render(self.scene, spp=1, integrator=self.integrator, 
                                     seed=random.randint(0, 100000), guiding=use_guiding, progress=False).cpu().numpy()
            
            # Train the network
            if self.train_realtime_checkbox.isChecked():
                loss = self.guiding_system.train_step(self.integrator)
                self.train_button.setText(f"Train 100 Steps (Loss: {loss:.2f})")
            
            # Update UI

            # --- Accumulation Logic ---
            if self.accumulate_checkbox.isChecked():
                # Add current frame to the buffer and increment counter
                self.accumulated_image += image_np
                self.spp_counter += 1
                
                # Calculate the average and convert to NumPy for display
                average_image = self.accumulated_image / self.spp_counter
                self.original_main_image_np = average_image
            else:
                # Just show the current frame
                self.original_main_image_np = image_np
            
            self._update_main_view_with_marker()

            # If a point is selected, update its visualizations in real-time
            if self.last_click_coords:
                si, _, _ = self._get_interaction_from_pos(QPoint(*self.last_click_coords))
                if dr.any(si.is_valid()):
                    self._update_vmf_visualizations(si)

        finally:
            self.is_rendering = False

    # -------------------------------------------------------------------------
    # --- Event Handlers (Slots)
    # -------------------------------------------------------------------------

    def on_mouse_click(self, event: QMouseEvent):
        """Handles mouse clicks on the main render view to inspect a point."""
        if self.is_rendering: return
        
        si, x, y = self._get_interaction_from_pos(event.pos())
        if not dr.any(si.is_valid()):
            self.hover_view.setText("Miss (No surface hit)")
            return

        self.last_click_coords = (x, y)
        self._update_main_view_with_marker()
        
        self.is_rendering = True
        self.hover_view.setText("Rendering GT...")
        self.normalized_pdf_view.setText("Rendering GT...")
        QApplication.processEvents()

        try:
            self._update_hemisphere_gt_view(si)
            self._update_vmf_visualizations(si)
        finally:
            self.is_rendering = False
    
    def train_step_clicked(self):
        """Manually triggers 100 training iterations."""
        print("Manual training run (100 steps) started.")
        self.integrator.set_guiding( False )

        for i in range(100):
            custom_render(self.scene, spp=1, integrator=self.integrator, progress=False, seed=random.randint(0, 100000), guiding=False)
            loss = self.guiding_system.train_step(self.integrator)
            if (i + 1) % 10 == 0:
                print(f"Step {i+1}/100, Loss: {loss}")
                self.train_button.setText(f"Train 100 Steps (Loss: {loss:.2f})")
        
        self.integrator.set_guiding(self.guiding_checkbox.isChecked())
        self.reprocess_last_click() # Update visualizations with new network state
        print("Manual training finished.")

    def re_render_main_image(self, spp=1):
        """Force a re-render of the main view, typically after changing a setting."""
        if self.is_rendering or self.scene is None: return
        
        # When re-rendering, always reset the accumulator for a fresh start
        self._reset_accumulator()

        self.is_rendering = True
        print(f"Re-rendering main image with path guiding: {self.guiding_checkbox.isChecked()}")
        try:
            use_guiding = self.guiding_checkbox.isChecked()
            self.integrator.set_guiding( use_guiding )
            image_np = custom_render(self.scene, spp=spp, integrator=self.integrator, seed=0, guiding=use_guiding).cpu().numpy()
            self.original_main_image_np = image_np
            self._update_main_view_with_marker()
        finally:
            self.is_rendering = False

    def reprocess_last_click(self):
        """Simulates a click at the last known coordinates to refresh views."""
        if self.last_click_coords:
            local_pos = QPointF(float(self.last_click_coords[0]), float(self.last_click_coords[1]))
            self.on_mouse_click(QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, local_pos,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
            ))

    def toggle_realtime_render(self, state):
        """Starts or stops the real-time rendering timer."""
        if state:
            self.render_timer.start()
        else:
            self.render_timer.stop()
            self.train_button.setText("Train 100 Steps") 

    def change_resolution(self, resolution):
        sensor_params = mi.traverse(self.main_sensor)
        sensor_params["film.size"] = resolution
        sensor_params.update()

    def save_ground_truth_images(self):
        """Opens a dialog to get render settings, then saves high-quality images."""
        if self.scene is None: return

        dialog = SaveGtDialog(self)
        if dialog.exec():
            settings = dialog.getValues()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.makedirs("renders", exist_ok=True)

            # Change resolution Temporarily
            self.change_resolution((settings['width'], settings['height']))
            
            # --- RENDER LOGIC ---
            if settings['online']:

                # --- Equal Time (Online) Comparison ---
                duration = settings['time']
                
                # Render unguided for the given duration
                print(f"Rendering unguided for {duration} seconds...")
                image_unguided = self._render_for_duration(self.main_sensor, duration, is_online_guided=False)
                
                # Render and train online for the same duration
                print(f"Rendering and training online for {duration} seconds...")
                image_guided = self._render_for_duration(self.main_sensor, duration, is_online_guided=True)

            else:
                # --- Final Quality (Offline) Comparison ---
                spp = settings['spp']
                
                # Render unguided at high SPP
                self.integrator.set_guiding(False)
                print(f"Rendering {spp} SPP unguided ground truth...")
                image_unguided = custom_render(self.scene, sensor=self.main_sensor, spp=spp, integrator=self.integrator, seed=42, guiding=False).cpu().numpy()
                
                # Render guided at high SPP
                self.integrator.set_guiding(True)
                print(f"Rendering {spp} SPP guided ground truth...")
                image_guided = custom_render(self.scene, sensor=self.main_sensor, spp=spp, integrator=self.integrator, seed=42, guiding = True).cpu().numpy()

            # --- Save both images ---
            for image, prefix in [(image_unguided, "unguided"), (image_guided, "guided")]:
                try:
                    mode = "online" if settings['online'] else "offline"
                    res = f"{settings['width']}x{settings['height']}"
                    
                    image_srgb = (np.clip(image, 0, 1) ** (1/2.2) * 255).astype(np.uint8)
                    img_pil = Image.fromarray(image_srgb)
                    
                    filename = f"{prefix}_{mode}_{res}_{timestamp}.png"
                    path = os.path.join("renders", filename)
                    img_pil.save(path)
                    print(f"Saved image to {path}")

                    bmp = mi.Bitmap(image)
                    exr_filename = f"{prefix}_{mode}_{res}_{timestamp}.exr"
                    exr_path = os.path.join("renders", exr_filename)
                    bmp.write(exr_path)
                    print(f"Saved HDR image to {exr_path}")
                    
                except Exception as e:
                    print(f"Error saving '{prefix}' image: {e}")

            self.change_resolution(self.MAIN_RESOLUTION)
        else:
            print("Save operation cancelled.")

    # -------------------------------------------------------------------------
    # --- UI & Visualization Update Helpers
    # -------------------------------------------------------------------------

    def _get_interaction_from_pos(self, q_pos: QPoint):
        """Converts a QPoint on the main view to a Mitsuba SurfaceInteraction."""
        x, y = q_pos.x(), q_pos.y()
        film_size = self.main_sensor.film().size()
        if not (0 <= x < film_size[0] and 0 <= y < film_size[1]): 
            return mi.SurfaceInteraction3f(), -1, -1

        u, v = (x + 0.5) / film_size[0], (y + 0.5) / film_size[1]
        ray, _ = self.main_sensor.sample_ray(0.0, 0.5, [u, v], [0.5, 0.5])
        return self.scene.ray_intersect(ray), x, y
        
    def _update_main_view_with_marker(self):
        """Redraws the main view, adding a marker if a point is selected."""
        if self.original_main_image_np is None: return
        
        image_to_show = self.original_main_image_np
        if self.last_click_coords:
            x, y = self.last_click_coords
            image_to_show = draw_marker_on_image(image_to_show.copy(), x, y)
        
        pixmap = numpy_to_qpixmap(image_to_show ** (1/2.2))
        self.main_view.setPixmap(pixmap)

    def _update_hemisphere_gt_view(self, si: mi.SurfaceInteraction3f):
        """Renders the ground truth hemispherical view for a given surface point."""
        if self.hover_sensor is None:
            self.hover_sensor = mi.load_dict({
                'type': 'gt_spherical_camera',
                'film': {'type': 'hdrfilm', 'width': self.HOVER_RESOLUTION[0], 'height': self.HOVER_RESOLUTION[1]}
            })

        dummy_ray = si.spawn_ray(si.n)
        self.hover_sensor.world_transform = mi.Transform4f.translate(dummy_ray.o)
        self.hover_sensor.initial_si = si
        self.hover_sensor.product_bsdf = self.bsdf_checkbox.isChecked()
        self.hover_sensor.product_cosine = self.cosine_checkbox.isChecked()

        gt_image_np = mi.render(self.scene, sensor=self.hover_sensor, spp=128).numpy()
        hover_image_np = np.transpose(gt_image_np, (1, 0, 2))

        self.normalized_pdf_view.setPixmap(numpy_to_qpixmap(hover_image_np ** (1/2.2)))

        luminance = np.mean(hover_image_np, axis=2)
        log_luminance = np.log1p(luminance)

        visualization_scale = 10.0
        scaled_luminance = log_luminance * visualization_scale

        colored_rgb = cm.viridis(scaled_luminance)[:, :, :3]
        self.hover_view.setPixmap(numpy_to_qpixmap(colored_rgb))

    def _update_vmf_visualizations(self, si: mi.SurfaceInteraction3f):
        """Updates the 2D and 3D visualizations of the guiding distribution."""
        ray = si.spawn_ray(dr.zeros(mi.Vector3f)) # dummy ray to get incident dir
        
        position = si.p.torch()
        wo = -ray.d.torch() # Incident direction
        roughness = torch.tensor([[1.0]], device=device) # Placeholder

        vmf_params = self.guiding_system.gnn(position, wo, roughness)
        guiding_dist = MixedSphericalGaussianDistribution(vmf_params, batch_idx=0)

        # Update 2D plot of the vMF mixture
        vmf_img_np = plot_non_batched_mixture(guiding_dist, si, device=device)
        vmf_img_np = np.transpose(vmf_img_np, (1, 0, 2))
        self.vmf_view.setPixmap(numpy_to_qpixmap(vmf_img_np))

        # Update 3D interactive sphere plot
        html = visualize_pdf_on_sphere_html(guiding_dist, res=50)
        self.vmf_sphere_view.setHtml(html)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MitsubaViewer()
    # viewer.showMaximized()
    viewer.show() 
    sys.exit(app.exec())