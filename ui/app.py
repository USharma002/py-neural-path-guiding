# Standard Library Imports
import os
import random
import sys
import time

# Third-Party Imports
import drjit as dr
import matplotlib.cm as cm
import mitsuba as mi
import numpy as np
import torch
from PIL import Image

# PyQt6 Imports
from PyQt6.QtCore import QTimer, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QColor, QKeyEvent
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget, QCheckBox,
    QComboBox, QGridLayout
)

# Local Project Imports
from guiding.config import get_config, get_logger, FPSProfiler, VisualizerConfig, NUMBA_AVAILABLE
from rendering.renderer import Renderer
from guiding.training_data import prepare_shared_training_data
from utils.visualization_helpers import (
    visualize_pdf_on_sphere_html, plot_nrc_radiance, plot_guiding_distribution,
    numpy_to_qpixmap, draw_marker_on_image
)
import sensor.spherical as spherical  # Register custom sensors (gt_spherical_camera)
import guiding.registry as registry  # Import to ensure methods are registered

from ui.dialogs import SaveGtDialog, TrainSettingsDialog, GuidingSettingsDialog

mi.set_variant('cuda_ad_rgb')
mi.set_log_level(mi.LogLevel.Warn)

# Initialize logging and apply PyTorch optimizations
logger = get_logger("visualizer")
device = "cuda"

# Log numba availability
if NUMBA_AVAILABLE:
    logger.info("Numba JIT compilation available - image processing will be accelerated")
else:
    logger.info("Numba not available - using numpy fallback for image processing")

custom_render = Renderer().render




class MitsubaViewer(QMainWindow):
    """
    An interactive PyQt application for visualizing Mitsuba 3 scenes,
    inspecting BSDFs, and training a path guiding neural network.
    """

    def __init__(self, config: VisualizerConfig = None):
        super().__init__()
        
        # Use provided config or get from global
        if config is None:
            config = get_config().visualizer
        self.config = config
        
        # Resolution from config
        self.MAIN_RESOLUTION = config.main_resolution
        self.HOVER_RESOLUTION = config.hover_resolution
        
        # Initialize FPS profiler
        self.fps_profiler = FPSProfiler(
            update_interval=config.fps_update_interval,
            smoothing=config.fps_smoothing
        )
        
        self._base_title = "Mitsuba Path Guiding Viewer"
        self.setWindowTitle(self._base_title)

        # --- State Variables ---
        self.scene = None
        self.integrator = None
        self.guiding_system = None
        self.main_sensor = None
        self.hover_sensor = None
        self.original_main_image_np = None
        self.last_click_coords = None
        self._last_si = None  # Store last surface interaction for reprocessing
        self.is_rendering = False
        self.visualizations_enabled = True  # Toggle with Esc to speed up training

        # --- Initialization ---
        self._init_ui()
        self._init_renderer()
        self._connect_signals()
        
        logger.info("MitsubaViewer initialized.")

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
        self.train_button = QPushButton("Train...")
        self.save_gt_button = QPushButton("Save GT Images...")
        self.settings_button = QPushButton("Settings...")
        
        # Guiding method dropdown
        self.guiding_method_label = QLabel("Guiding:")
        self.guiding_method_combo = QComboBox()
        for method_name in registry.get_method_names():
            self.guiding_method_combo.addItem(method_name)
        # Set default selection
        default_method = registry.get_default_method()
        if default_method:
            idx = self.guiding_method_combo.findText(default_method)
            if idx >= 0:
                self.guiding_method_combo.setCurrentIndex(idx)
        
        top_controls_layout.addWidget(self.browse_button)
        top_controls_layout.addWidget(self.train_button)
        top_controls_layout.addWidget(self.save_gt_button)
        top_controls_layout.addWidget(self.settings_button)
        top_controls_layout.addWidget(self.guiding_method_label)
        top_controls_layout.addWidget(self.guiding_method_combo)
        top_controls_layout.addStretch()
        self.main_layout.addLayout(top_controls_layout)

        # --- Checkboxes ---
        checkbox_layout = QHBoxLayout()
        self.cosine_checkbox = QCheckBox("Cosine Product")
        self.bsdf_checkbox = QCheckBox("BSDF Product")
        self.guiding_checkbox = QCheckBox("Use Path Guiding")
        self.realtime_checkbox = QCheckBox("Real-Time Render")
        self.train_realtime_checkbox = QCheckBox("Train in Real-Time")
        self.train_nrc_checkbox = QCheckBox("Train NRC")
        self.accumulate_checkbox = QCheckBox("Accumulate Frames")
        self.bsdf_checkbox.setChecked(True)
        for widget in [self.cosine_checkbox, self.bsdf_checkbox, self.guiding_checkbox, self.realtime_checkbox, self.train_realtime_checkbox, self.train_nrc_checkbox, self.accumulate_checkbox]:
            checkbox_layout.addWidget(widget)
        checkbox_layout.addStretch()
        self.main_layout.addLayout(checkbox_layout)

        # --- Main Visualization Area (2x3 Grid) ---
        vis_layout = QHBoxLayout()
        
        # Left side: Main render view (larger)
        self.main_view = QLabel("Click 'Browse' to load a scene")
        self.main_view.setMinimumSize(self.MAIN_RESOLUTION[0] // 2, self.MAIN_RESOLUTION[1] // 2)
        self.main_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_layout.addWidget(self.main_view, stretch=2)
        
        # Right side: 2x3 grid of detail views
        detail_grid = QGridLayout()
        detail_grid.setSpacing(5)
        
        self.hover_view = QLabel("Hemisphere View\n(Click to inspect)")
        self.normalized_pdf_view = QLabel("Ground Truth (sRGB)")
        self.vmf_view = QLabel("Guiding Distribution")
        self.nrc_view = QLabel("NRC Radiance")
        self.vmf_sphere_view = QWebEngineView()
        self.extra_view = QLabel("Extra View\n(Reserved)")
        
        # Configure QLabels
        detail_views = [self.hover_view, self.normalized_pdf_view, self.vmf_view, 
                        self.nrc_view, self.extra_view]
        for view in detail_views:
            view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setStyleSheet("QLabel { background-color: #1a1a1a; color: white; border: 1px solid #333; }")
        
        # Configure QWebEngineView
        self.vmf_sphere_view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
        self.vmf_sphere_view.page().setBackgroundColor(QColor("black"))
        
        # Add to grid: row 0 (top), row 1 (bottom)
        detail_grid.addWidget(self.hover_view, 0, 0)
        detail_grid.addWidget(self.normalized_pdf_view, 0, 1)
        detail_grid.addWidget(self.vmf_view, 0, 2)
        detail_grid.addWidget(self.nrc_view, 1, 0)
        detail_grid.addWidget(self.vmf_sphere_view, 1, 1)
        detail_grid.addWidget(self.extra_view, 1, 2)
        
        vis_layout.addLayout(detail_grid, stretch=3)
        self.main_layout.addLayout(vis_layout)
        
        # --- Status bar hint ---
        self.statusBar().showMessage("Press Esc to toggle visualizations for faster training")

        # Timer for real-time rendering
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(self.config.realtime_interval_ms)
        
        # Timer for FPS display update
        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(int(self.config.fps_update_interval * 1000))
        self.fps_timer.timeout.connect(self._update_title_with_fps)
        self.fps_timer.start()

    def _reset_accumulator(self):
        """Resets the accumulation buffer and SPP counter."""
        logger.debug("Accumulator reset")
        h, w = self.MAIN_RESOLUTION[1], self.MAIN_RESOLUTION[0]
        # Initialize on the GPU for fast accumulation
        self.accumulated_image = np.zeros((h, w, 3))
        self.spp_counter = 0

    def _init_renderer(self):
        """Loads the default scene and sets up the Mitsuba integrator."""
        self.integrator = mi.load_dict({"type": "path_guiding_integrator"})
        self.guiding_system = self.integrator.guiding_system
        self.nrc_system = self.integrator.nrc_system
        self.load_default_scene()

    def _connect_signals(self):
        """Connects all widget signals to their corresponding slots."""
        self.browse_button.clicked.connect(self.load_scene_from_file)
        self.train_button.clicked.connect(self._open_train_dialog)
        self.save_gt_button.clicked.connect(self.save_ground_truth_images)
        self.settings_button.clicked.connect(self._open_settings_dialog)
        
        self.cosine_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.bsdf_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.guiding_checkbox.stateChanged.connect(lambda: self.re_render_main_image(spp=1))
        self.realtime_checkbox.stateChanged.connect(self.toggle_realtime_render)
        self.accumulate_checkbox.stateChanged.connect(self._reset_accumulator)
        
        # Guiding method dropdown
        self.guiding_method_combo.currentTextChanged.connect(self._on_guiding_method_changed)


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
            logger.warning("Could not find 'light' in scene dictionary to transform.")
        return scene_dict

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key press events for the main window."""
        if event.key() == Qt.Key.Key_Escape:
            self._toggle_visualizations()
        else:
            super().keyPressEvent(event)
    
    def _toggle_visualizations(self):
        """Toggle visualizations on/off to speed up training."""
        self.visualizations_enabled = not self.visualizations_enabled
        status = "enabled" if self.visualizations_enabled else "disabled (faster training)"
        logger.info(f"Visualizations {status}")
        
        # Update window title to show status
        if self.visualizations_enabled:
            self.setWindowTitle(self._base_title)
        else:
            self.setWindowTitle(f"{self._base_title} [VIS OFF]")
        
        # Clear selection when disabling
        if not self.visualizations_enabled:
            self.clear_selection()

    def clear_selection(self):
        """Resets the UI after a point was selected."""
        if self.last_click_coords is None:
            return

        logger.debug("Clearing selected point.")
        self.last_click_coords = None
        self._last_si = None

        # Redraw the main view without the marker
        self._update_main_view_with_marker()

        # Reset the detail views to their initial placeholder text
        self.hover_view.setText("Click render to inspect point\n(Viridis PDF)")
        self.normalized_pdf_view.setText("Ground Truth (sRGB)")
        self.vmf_view.setText("Guiding Distribution")
        self.nrc_view.setText("NRC Radiance\n(Click to inspect)")
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
            logger.info(f"Setting up integrator for a wavefront of {num_rays} rays.")
            
            self.integrator.setup(
                num_rays=num_rays,
                bbox_min=self.scene.bbox().min,
                bbox_max=self.scene.bbox().max
            )
            self.re_render_main_image(spp=16)

        except Exception as e:
            logger.error(f"Failed to load scene: {e}")
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
        
        logger.info(f"Starting {'online guided' if is_online_guided else 'unguided'} render for {duration} seconds...")
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

        logger.info(f"Rendered {spp_counter} SPP in ~{duration} seconds.")

        return final_image.detach().cpu().numpy()

    def _update_title_with_fps(self):
        """Update window title with FPS stats."""
        fps_string = self.fps_profiler.get_stats_string()
        guiding_status = "Guiding ON" if self.guiding_checkbox.isChecked() else "Guiding OFF"
        spp_info = f"SPP: {self.spp_counter}" if self.accumulate_checkbox.isChecked() else ""
        parts = [self._base_title, fps_string, guiding_status]
        if spp_info:
            parts.append(spp_info)
        self.setWindowTitle(" | ".join(parts))

    def realtime_render_loop(self):
        """Performs one step of rendering and training."""
        if self.is_rendering: return
        self.is_rendering = True
        
        # Start frame timing
        self.fps_profiler.begin_frame()
        
        # Cache checkbox states
        use_guiding = self.guiding_checkbox.isChecked()
        do_training = self.train_realtime_checkbox.isChecked()
        do_nrc_training = self.train_nrc_checkbox.isChecked()
        do_accumulate = self.accumulate_checkbox.isChecked()
        
        try:
            self.integrator.set_guiding(use_guiding)
            
            # Render one frame
            with torch.inference_mode():
                image_np = custom_render(self.scene, spp=1, integrator=self.integrator, 
                                         seed=random.randint(0, 100000), guiding=use_guiding, progress=False).cpu().numpy()
            
            # Prepare shared training data
            if do_training or do_nrc_training:
                training_batch = prepare_shared_training_data(self.integrator, device=device)
            
            # Train guiding
            if do_training:
                loss = self.guiding_system.train_step_from_batch(training_batch)
                self.train_button.setText(f"Train 100 Steps (Loss: {loss:.2f})")
            
            # Train NRC
            if do_nrc_training:
                nrc_loss = self.nrc_system.train_step_from_batch(training_batch)
            
            # Accumulate
            if do_accumulate:
                self.accumulated_image += image_np
                self.spp_counter += 1
                self.original_main_image_np = self.accumulated_image / self.spp_counter
            else:
                self.original_main_image_np = image_np
            
            self._update_main_view_with_marker()

            # FIX: Use the stored widget position for accurate re-casting
            # We check 'last_widget_click_pos' instead of 'last_click_coords' for the raycast source
            if self.visualizations_enabled and hasattr(self, 'last_widget_click_pos') and self.last_widget_click_pos:
                si, _, _ = self._get_interaction_from_pos(self.last_widget_click_pos)
                
                # Only update if we still hit a valid surface
                if dr.any(si.is_valid()):
                    self._update_vmf_visualizations(si)

        finally:
            self.fps_profiler.end_frame()
            self.is_rendering = False

    # -------------------------------------------------------------------------
    # --- Event Handlers (Slots)
    # -------------------------------------------------------------------------


    def _open_train_dialog(self):
        """Opens training settings dialog and runs training."""
        # Get current learning rate from guiding system's config
        current_lr = 5e-3
        if self.guiding_system and hasattr(self.guiding_system, '_distribution'):
            dist = self.guiding_system._distribution
            config = getattr(dist, 'config', None)
            if config and hasattr(config, 'learning_rate'):
                current_lr = config.learning_rate
        
        dialog = TrainSettingsDialog(self, current_lr=current_lr)
        if dialog.exec():
            settings = dialog.getValues()
            self._run_training(settings['iterations'], settings['learning_rate'])
    
    def _run_training(self, iterations: int, learning_rate: float = None):
        """Run training with specified settings."""
        logger.info(f"Training run ({iterations} steps) started.")
        
        # Update learning rate if specified
        if learning_rate is not None and self.guiding_system:
            dist = self.guiding_system._distribution
            if hasattr(dist, 'optimizer'):
                for param_group in dist.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                logger.info(f"Learning rate set to {learning_rate}")
        
        self.integrator.set_guiding(False)

        for i in range(iterations):
            
            custom_render(self.scene, spp=1, integrator=self.integrator, progress=False, 
                         seed=random.randint(0, 100000), guiding=False)
        
            training_batch = prepare_shared_training_data(self.integrator, device=device)
            loss = self.guiding_system.train_step_from_batch(training_batch)
            # loss = self.guiding_system.train_step(self.integrator)
            if (i + 1) % 10 == 0:
                logger.info(f"Step {i+1}/{iterations}, Loss: {loss:.4f}")
                self.train_button.setText(f"Training... ({i+1}/{iterations})")

                # Force update of PDF plots if a coordinate is selected
                if self.visualizations_enabled and self.last_click_coords:
                    self.reprocess_last_click()
                QApplication.processEvents()  # Keep UI responsive
        
        self.train_button.setText("Train...")
        self.integrator.set_guiding(self.guiding_checkbox.isChecked())
        self.reprocess_last_click()
        logger.info("Training finished.")
    
    def _open_settings_dialog(self):
        """Opens the settings dialog for render and guiding distribution."""
        dialog = GuidingSettingsDialog(self, self.guiding_system, self.MAIN_RESOLUTION)
        if dialog.exec():
            # Handle resolution change
            new_resolution = dialog.getResolution()
            if new_resolution != self.MAIN_RESOLUTION:
                self.MAIN_RESOLUTION = new_resolution
                self.change_resolution(new_resolution)
                self._reset_accumulator()
                logger.info(f"Resolution changed to {new_resolution}")
                if self.scene:
                    self.re_render_main_image(spp=1)
            
            # Apply guiding config settings
            values = dialog.getValues()
            if values and self.guiding_system:
                dist = self.guiding_system._distribution
                # All distributions store config in self.config (from base class)
                config = getattr(dist, 'config', None)
                if config:
                    for key, value in values.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                            logger.info(f"Updated {key} = {value}")
                    
                    # Update optimizer learning rate if changed
                    if 'learning_rate' in values and hasattr(dist, 'optimizer'):
                        for param_group in dist.optimizer.param_groups:
                            param_group['lr'] = values['learning_rate']
    
    def train_step_clicked(self):
        """Manually triggers training iterations (legacy, now uses dialog)."""
        self._run_training(100)

    def re_render_main_image(self, spp=1):
        """Force a re-render of the main view, typically after changing a setting."""
        if self.is_rendering or self.scene is None: return
        
        # When re-rendering, always reset the accumulator for a fresh start
        self._reset_accumulator()

        self.is_rendering = True
        logger.debug(f"Re-rendering main image with path guiding: {self.guiding_checkbox.isChecked()}")
        try:
            use_guiding = self.guiding_checkbox.isChecked()
            self.integrator.set_guiding( use_guiding )
            image_np = custom_render(self.scene, spp=spp, integrator=self.integrator, seed=0, guiding=use_guiding).cpu().numpy()
            self.original_main_image_np = image_np
            self._update_main_view_with_marker()
        finally:
            self.is_rendering = False

    def reprocess_last_click(self):
        """Refresh visualizations for the last clicked point."""
        if not hasattr(self, '_last_si') or self._last_si is None:
            return
        if not dr.any(self._last_si.is_valid()):
            return
        
        self.is_rendering = True
        self.hover_view.setText("Updating...")
        QApplication.processEvents()
        
        try:
            self._update_hemisphere_gt_view(self._last_si)
            self._update_vmf_visualizations(self._last_si)
        finally:
            self.is_rendering = False

    def toggle_realtime_render(self, state):
        """Starts or stops the real-time rendering timer."""
        if state:
            self.render_timer.start()
        else:
            self.render_timer.stop()
            self.train_button.setText("Train...")
    
    def _on_guiding_method_changed(self, method_name: str):
        """Handle guiding method dropdown change."""
        if self.guiding_system is None:
            return
        
        logger.info(f"Switching guiding method to: {method_name}")
        self.guiding_system.switch_method(method_name)
        
        # Reset accumulator since the guiding distribution changed
        self._reset_accumulator()
        
        # Update the integrator's reference if needed
        if hasattr(self.integrator, 'guiding_system'):
            # The integrator already has a reference to the same object
            pass
        
        # Re-render and update visualizations
        if self.scene is not None:
            self.re_render_main_image(spp=1)
            # Update PDF visualizations if a point was selected
            self.reprocess_last_click() 

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
    def on_mouse_click(self, event: QMouseEvent):
        """Handles mouse clicks on the main render view to inspect a point."""
        if self.is_rendering: return
        
        # Re-enable visualizations on click if they were disabled
        if not self.visualizations_enabled:
            self.visualizations_enabled = True
            self.setWindowTitle(self._base_title)
            logger.info("Visualizations re-enabled (click detected)")
        
        # FIX: Store the raw widget position for the realtime loop to re-use
        self.last_widget_click_pos = event.pos()

        si, x, y = self._get_interaction_from_pos(event.pos())
        if not dr.any(si.is_valid()):
            self.hover_view.setText("Miss (No surface hit)")
            return

        self.last_click_coords = (x, y)
        self._last_si = si
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

    def _get_interaction_from_pos(self, q_pos: QPoint):
        """Converts a QPoint on the main view to a Mitsuba SurfaceInteraction."""
        # Get click position relative to widget
        widget_x, widget_y = q_pos.x(), q_pos.y()
        
        # Get widget size and film size
        widget_size = self.main_view.size()
        film_size = self.main_sensor.film().size()
        
        # Get the pixmap size if available (actual displayed image size)
        pixmap = self.main_view.pixmap()
        if pixmap and not pixmap.isNull():
            # Calculate the actual displayed image rect within the widget
            # QLabel centers the pixmap by default
            img_w, img_h = pixmap.width(), pixmap.height()
            widget_w, widget_h = widget_size.width(), widget_size.height()
            
            # Calculate offset if image is centered
            offset_x = (widget_w - img_w) / 2
            offset_y = (widget_h - img_h) / 2
            
            # Adjust click position relative to the actual image
            img_x = widget_x - offset_x
            img_y = widget_y - offset_y
            
            # Check bounds
            if not (0 <= img_x < img_w and 0 <= img_y < img_h):
                return mi.SurfaceInteraction3f(), -1, -1
            
            # Scale from pixmap coordinates to film coordinates
            x = int(img_x * film_size[0] / img_w)
            y = int(img_y * film_size[1] / img_h)
        else:
            # Fallback: scale directly from widget to film
            x = int(widget_x * film_size[0] / widget_size.width())
            y = int(widget_y * film_size[1] / widget_size.height())
        
        # Clamp to valid range
        x = max(0, min(x, film_size[0] - 1))
        y = max(0, min(y, film_size[1] - 1))

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
        wi = -ray.d.torch()  # Incoming direction (towards surface)
        roughness = torch.tensor([[1.0]], device=device)  # Placeholder

        # Use the generic plot function - works with any GuidingDistribution
        vmf_img_np = plot_guiding_distribution(
            self.guiding_system, si, position, wi, roughness, device=device
        )
        vmf_img_np = np.transpose(vmf_img_np, (1, 0, 2))
        self.vmf_view.setPixmap(numpy_to_qpixmap(vmf_img_np))

        # Update 3D interactive sphere plot - get visualization distribution
        vis_dist = self.guiding_system.get_distribution_for_visualization(position, wi, roughness)
        html = visualize_pdf_on_sphere_html(vis_dist, res=50)
        self.vmf_sphere_view.setHtml(html)
        
        # Update NRC visualization
        self._update_nrc_visualization(si)

    def _update_nrc_visualization(self, si: mi.SurfaceInteraction3f):
        """Visualizes the Neural Radiance Cache prediction for a surface point.
        
        Shows a spherical view of the scene as predicted by the NRC -
        like placing a spherical camera at the clicked point.
        """
        try:
            # Get surface properties
            position = si.p.torch()  # (1, 3)
            normal = si.n.torch()    # (1, 3)
            
            # Get bbox from integrator for proper normalization
            bbox_min = self.integrator.bbox_min.torch().to(device)
            bbox_max = self.integrator.bbox_max.torch().to(device)
            
            # Render scene view using NRC predictions
            nrc_img_np = plot_nrc_radiance(
                self.nrc_system, 
                si, 
                position, 
                normal,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                resolution=self.HOVER_RESOLUTION,
                device=device
            )
            nrc_img_np = np.transpose(nrc_img_np, (1, 0, 2))  # Match orientation
            
            self.nrc_view.setPixmap(numpy_to_qpixmap(nrc_img_np))
            
        except Exception as e:
            logger.error(f"Failed to update NRC visualization: {e}")
            import traceback
            traceback.print_exc()
            self.nrc_view.setText(f"NRC Error:\\n{str(e)[:50]}")
