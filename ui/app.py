"""Interactive viewer: real-time rendering, online training, and per-pixel guiding/NRC inspection."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import drjit as dr
import matplotlib.cm as cm
import mitsuba as mi
import numpy as np
import torch
from PIL import Image

from PyQt6.QtCore import QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None

from guiding.config import FPSProfiler, VisualizerConfig, get_logger
from rendering.renderer import Renderer
from rendering.scenes import modified_cornell_box
from utils.visualization_helpers import (
    draw_marker_on_image, numpy_to_qpixmap, plot_guiding_distribution, plot_nrc_radiance,
    visualize_pdf_on_sphere_html,
)
import guiding.registry as registry
from ui.dialogs import GuidingSettingsDialog, SaveGtDialog, TrainingDataDialog, TrainSettingsDialog

mi.set_variant("cuda_ad_rgb")
mi.set_log_level(mi.LogLevel.Warn)

logger = get_logger("visualizer")
custom_render = Renderer().render
PANEL_STYLE = "QLabel { background-color: #1a1a1a; color: white; border: 1px solid #333; }"


def welford_update(mean: Optional[np.ndarray], m2: Optional[np.ndarray], n: int, frame: np.ndarray):
    """One step of Welford's running mean / M2 (allocates buffers on first call)."""
    if mean is None:
        mean = np.zeros_like(frame, dtype=np.float32)
        m2 = np.zeros_like(frame, dtype=np.float32)
    n += 1
    delta = frame - mean
    mean += delta / n
    m2 += delta * (frame - mean)
    return mean, m2, n


class LossPlotWidget(QWidget):
    def __init__(self, parent=None, *, title: str, label: str, color: str):
        super().__init__(parent)
        self._title, self._label, self._color = title, label, color
        self._loss: List[float] = []
        self.setMinimumSize(256, 256)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")

    def append(self, value: float):
        self._loss = (self._loss + [float(value)])[-2000:]
        self.update()

    def clear(self):
        self._loss.clear()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = max(1, self.width()), max(1, self.height())
        p.setPen(QPen(QColor("#333"), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.setPen(QPen(QColor("#aaa"), 1))
        p.drawText(8, 16, self._title)

        y = np.array([v for v in self._loss if np.isfinite(v)])
        if len(y) < 2:
            p.setPen(QPen(QColor("#666"), 1))
            p.drawText(8, 34, "No data")
            return

        # Axis range from the 2nd-98th percentiles so single spikes do not flatten the curve
        lo, hi = np.percentile(y, 2), np.percentile(y, 98)
        hi = hi if hi > lo else lo + 1.0
        x0, y0, x1, y1 = 6, 24, w - 6, h - 6
        to_px = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
        if lo < 0.0 < hi:
            p.setPen(QPen(QColor("#444"), 1))
            p.drawLine(x0, int(to_px(0.0)), x1, int(to_px(0.0)))

        xs = np.linspace(x0, x1, len(y))
        ys = to_px(np.clip(y, lo, hi))
        p.setPen(QPen(QColor(self._color), 2))
        for i in range(len(y) - 1):
            p.drawLine(int(xs[i]), int(ys[i]), int(xs[i + 1]), int(ys[i + 1]))
        p.setPen(QPen(QColor(self._color), 1))
        p.drawText(8, h - 10, f"{self._label}: {y[-1]:.6f}")


class VizWorker(QThread):
    """Computes the right-panel images for a surface point off the UI thread (guiding + NRC only if include_hemisphere=False)."""
    result_ready = pyqtSignal(int, object)  # job_id, dict
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, viewer: "MitsubaViewer", si: mi.SurfaceInteraction3f, render_sphere: bool,
                 include_hemisphere: bool = True):
        super().__init__()
        self.job_id, self.viewer, self.si = job_id, viewer, si
        self.render_sphere, self.include_hemisphere = render_sphere, include_hemisphere

    def run(self):
        try:
            v = self.viewer
            out: Dict[str, Any] = {}
            if self.include_hemisphere:
                out["hemi_rgb"] = v._render_hemisphere_rgb(self.si)
                out["hemi_pdf_rgb"] = v._hemi_rgb_to_pdf_viz(out["hemi_rgb"])
            out["vmf_rgb"] = v._render_guiding_2d(self.si)
            out["nrc_rgb"] = v._render_nrc(self.si)
            out["vmf_html"] = v._render_guiding_sphere_html(self.si) if self.render_sphere else ""
            self.result_ready.emit(self.job_id, out)
        except Exception as e:
            self.failed.emit(self.job_id, str(e))


class MitsubaViewer(QMainWindow):
    def __init__(self, config: Optional[VisualizerConfig] = None):
        super().__init__()
        self.config = config or VisualizerConfig()
        self.MAIN_RESOLUTION = tuple(self.config.main_resolution)
        self.HOVER_RESOLUTION = tuple(self.config.hover_resolution)
        self.fps_profiler = FPSProfiler()
        self._base_title = "Mitsuba Path Guiding Viewer"
        self.setWindowTitle(self._base_title)

        self.scene = None
        self.integrator = None
        self.main_sensor = None
        self.original_main_image_np = None

        self.last_click_coords: Optional[Tuple[int, int]] = None
        self.last_widget_click_pos: Optional[QPoint] = None
        self._last_si: Optional[mi.SurfaceInteraction3f] = None

        self.is_rendering = False
        self.visualizations_enabled = True

        # Welford accumulators for mean / variance
        self.spp_counter = 0
        self._accum_n = 0
        self._accum_mean = None
        self._accum_m2 = None

        self._click_job_id = 0
        self._click_job_full = True
        self._click_worker: Optional[VizWorker] = None
        self._viz_busy = False

        # Click position and toggles the right panel was last rendered for
        self._last_vis_click_coords: Optional[Tuple[int, int]] = None
        self._last_vis_flags: Optional[Tuple] = None
        self._last_sphere_html = ""
        self._last_training_viz_refresh_t = 0.0

        self._init_ui()
        self._create_integrator(self.guiding_method_combo.currentText())
        self.load_scene()
        self._connect_signals()

    # --- UI ---

    def _init_ui(self):
        central = QWidget()
        self.main_layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        top = QHBoxLayout()
        self.browse_button = QPushButton("Browse Scene")
        self.train_button = QPushButton("Train...")
        self.save_gt_button = QPushButton("Save GT Images...")
        self.settings_button = QPushButton("Settings...")
        self.training_data_button = QPushButton("Training Data...")
        self.guiding_method_combo = QComboBox()
        self.guiding_method_combo.addItems(registry.get_method_names())
        self.guiding_method_combo.setCurrentText(registry.get_default_method())
        for w in (self.browse_button, self.train_button, self.save_gt_button, self.settings_button,
                  self.training_data_button, QLabel("Guiding:"), self.guiding_method_combo):
            top.addWidget(w)
        top.addStretch()
        self.main_layout.addLayout(top)

        checks = QHBoxLayout()
        self.cosine_checkbox = QCheckBox("Cosine Product")
        self.bsdf_checkbox = QCheckBox("BSDF Product")
        self.guiding_checkbox = QCheckBox("Use Path Guiding")
        self.use_nee_checkbox = QCheckBox("Use NEE")
        self.realtime_checkbox = QCheckBox("Real-Time Render")
        self.train_realtime_checkbox = QCheckBox("Train in Real-Time")
        self.train_nrc_checkbox = QCheckBox("Train NRC")
        self.use_nrc_checkbox = QCheckBox("Use NRC")
        self.accumulate_checkbox = QCheckBox("Accumulate Frames")
        self.show_variance_checkbox = QCheckBox("Show Variance")
        self.show_sphere_checkbox = QCheckBox("3D Sphere")
        self.show_sphere_checkbox.setChecked(True)
        self.bsdf_checkbox.setChecked(True)
        for w in (self.cosine_checkbox, self.bsdf_checkbox, self.guiding_checkbox, self.use_nee_checkbox,
                  self.realtime_checkbox, self.train_realtime_checkbox, self.train_nrc_checkbox,
                  self.use_nrc_checkbox, self.accumulate_checkbox, self.show_variance_checkbox,
                  self.show_sphere_checkbox):
            checks.addWidget(w)
        checks.addStretch()
        self.main_layout.addLayout(checks)

        vis_layout = QHBoxLayout()
        self.main_view = QLabel("Click 'Browse Scene' to load a scene")
        self.main_view.setMinimumSize(self.MAIN_RESOLUTION[0] // 2, self.MAIN_RESOLUTION[1] // 2)
        self.main_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_view.setStyleSheet("QLabel { background-color: #101010; color: white; border: 1px solid #333; }")
        vis_layout.addWidget(self.main_view, stretch=2)

        self.hover_view = QLabel()
        self.normalized_pdf_view = QLabel()
        self.vmf_view = QLabel()
        self.nrc_view = QLabel()
        self.guiding_loss_plot = LossPlotWidget(title="Guiding Loss", label="G", color="#4cc9f0")
        self.nrc_loss_plot = LossPlotWidget(title="NRC Loss", label="NRC", color="#f77f00")
        if QWebEngineView is not None:
            self.vmf_sphere_view = QWebEngineView()
            self.vmf_sphere_view.page().setBackgroundColor(QColor("black"))
        else:
            self.vmf_sphere_view = QLabel("QWebEngineView not available")
        for view in (self.hover_view, self.normalized_pdf_view, self.vmf_view, self.nrc_view, self.vmf_sphere_view):
            view.setMinimumSize(*self.HOVER_RESOLUTION)
            if isinstance(view, QLabel):
                view.setAlignment(Qt.AlignmentFlag.AlignCenter)
                view.setStyleSheet(PANEL_STYLE)
        self._set_panel_placeholders()

        self.sphere_or_nrc_stack = QStackedWidget()
        self.sphere_or_nrc_stack.addWidget(self.vmf_sphere_view)
        self.sphere_or_nrc_stack.addWidget(self.nrc_loss_plot)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.hover_view, 0, 0)
        grid.addWidget(self.normalized_pdf_view, 0, 1)
        grid.addWidget(self.vmf_view, 0, 2)
        grid.addWidget(self.nrc_view, 1, 0)
        grid.addWidget(self.sphere_or_nrc_stack, 1, 1)
        grid.addWidget(self.guiding_loss_plot, 1, 2)
        vis_layout.addLayout(grid, stretch=3)
        self.main_layout.addLayout(vis_layout)

        self.statusBar().showMessage("Press Esc to toggle visualizations for faster training")

        self.render_timer = QTimer(self)
        self.render_timer.setInterval(self.config.realtime_interval_ms)
        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(int(self.config.fps_update_interval * 1000))
        self.fps_timer.timeout.connect(self._update_title_with_fps)
        self.fps_timer.start()

        self._reset_accumulator()
        self._apply_sphere_visibility()

    def _set_panel_placeholders(self):
        self.hover_view.setText("Hemisphere PDF (Viridis)\n(Click main render to inspect)")
        self.normalized_pdf_view.setText("Hemisphere RGB (sRGB)")
        self.vmf_view.setText("Guiding Distribution")
        self.nrc_view.setText("NRC Radiance")

    def _connect_signals(self):
        self.browse_button.clicked.connect(self.load_scene_from_file)
        self.train_button.clicked.connect(self._open_train_dialog)
        self.save_gt_button.clicked.connect(self.save_ground_truth_images)
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self.training_data_button.clicked.connect(self._open_training_data_dialog)

        self.cosine_checkbox.stateChanged.connect(self._on_viz_setting_changed)
        self.bsdf_checkbox.stateChanged.connect(self._on_viz_setting_changed)
        self.guiding_checkbox.stateChanged.connect(lambda: self.re_render_main_image(spp=1))
        self.use_nee_checkbox.stateChanged.connect(self._on_nee_toggled)
        self.use_nrc_checkbox.stateChanged.connect(self._on_nrc_toggled)
        self.realtime_checkbox.stateChanged.connect(self.toggle_realtime_render)
        self.accumulate_checkbox.stateChanged.connect(self._reset_accumulator)
        self.show_variance_checkbox.stateChanged.connect(lambda _: self._update_main_view_with_marker())
        self.show_sphere_checkbox.stateChanged.connect(lambda _: self._apply_sphere_visibility())
        self.guiding_method_combo.currentTextChanged.connect(self._on_guiding_method_changed)

        self.render_timer.timeout.connect(self.realtime_render_loop)
        self.main_view.mousePressEvent = self.on_mouse_click

    def _open_training_data_dialog(self):
        if self.scene is None:
            return
        # Training compacts the record; re-render so it is back in (H*W*D) layout.
        if self.integrator._data_scattered or self.integrator.surfaceInteractionRecord is None:
            self._render_one_spp(guiding=self.guiding_checkbox.isChecked())
        TrainingDataDialog(self, integrator=self.integrator, resolution=self.MAIN_RESOLUTION).exec()

    def _sphere_supported(self) -> bool:
        return QWebEngineView is not None and isinstance(self.vmf_sphere_view, QWebEngineView)

    def _apply_sphere_visibility(self):
        enabled = self.show_sphere_checkbox.isChecked()
        self.sphere_or_nrc_stack.setCurrentWidget(self.vmf_sphere_view if enabled else self.nrc_loss_plot)
        if enabled and self._sphere_supported() and self._last_sphere_html:
            self.vmf_sphere_view.setHtml(self._last_sphere_html)

    def _on_viz_setting_changed(self):
        self._last_vis_flags = None
        self.reprocess_last_click()

    # --- Integrator and scene ---

    def _num_rays(self) -> int:
        return self.MAIN_RESOLUTION[0] * self.MAIN_RESOLUTION[1]

    def _create_integrator(self, method_name: str):
        """Build the method's integrator, keeping the NRC and NEE setting from the previous one."""
        old = self.integrator
        self.integrator = registry.create_integrator(method_name, nrc_system=old.nrc_system if old else None)
        self.integrator.use_nee = self.use_nee_checkbox.isChecked()
        if self.scene is not None:
            self.integrator.setup(self.scene, self._num_rays())

    def _on_nee_toggled(self):
        self.integrator.use_nee = self.use_nee_checkbox.isChecked()
        self.re_render_main_image(spp=1)
        self.reprocess_last_click()

    def _on_nrc_toggled(self):
        self.integrator.set_nrc(self.use_nrc_checkbox.isChecked())
        self.re_render_main_image(spp=1)

    def load_scene_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mitsuba Scene", "", "XML Files (*.xml)")
        if file_path:
            self.load_scene(file_path)

    def load_scene(self, scene_path=None):
        try:
            self.scene = mi.load_file(scene_path) if scene_path else mi.load_dict(modified_cornell_box(self.MAIN_RESOLUTION))
            self.main_sensor = self.scene.sensors()[0]
            self.change_resolution(self.MAIN_RESOLUTION)
            self.integrator.setup(self.scene, self._num_rays())
            self._reset_accumulator()
            self.re_render_main_image(spp=16)
        except Exception as e:
            logger.exception("Failed to load scene")
            self.main_view.setText(f"Failed to load scene:\n{e}")

    def change_resolution(self, resolution: Tuple[int, int]):
        if self.main_sensor is None:
            return
        params = mi.traverse(self.main_sensor)
        params["film.size"] = resolution
        params.update()

    # --- Film click mapping (KeepAspectRatio-safe) ---

    def _get_interaction_from_pos(self, q_pos: QPoint):
        rect = self.main_view.contentsRect()
        film_w, film_h = (int(v) for v in self.main_sensor.film().size())
        scale = min(rect.width() / film_w, rect.height() / film_h)
        if scale <= 0:
            return mi.SurfaceInteraction3f(), -1, -1
        disp_w, disp_h = film_w * scale, film_h * scale
        u = (q_pos.x() - rect.x() - (rect.width() - disp_w) / 2) / disp_w
        v = (q_pos.y() - rect.y() - (rect.height() - disp_h) / 2) / disp_h
        if not (0 <= u < 1 and 0 <= v < 1):
            return mi.SurfaceInteraction3f(), -1, -1

        x, y = min(int(u * film_w), film_w - 1), min(int(v * film_h), film_h - 1)
        ray, _ = self.main_sensor.sample_ray(0.0, 0.5, [(x + 0.5) / film_w, (y + 0.5) / film_h], [0.5, 0.5])
        return self.scene.ray_intersect(ray), x, y

    # --- Accumulation and variance ---

    def _reset_accumulator(self):
        w, h = self.MAIN_RESOLUTION
        self.spp_counter = self._accum_n = 0
        self._accum_mean = np.zeros((h, w, 3), dtype=np.float32)
        self._accum_m2 = np.zeros((h, w, 3), dtype=np.float32)
        self.original_main_image_np = self._accum_mean.copy()
        self._update_main_view_with_marker()

    def _welford_update(self, frame_np: np.ndarray):
        self._accum_mean, self._accum_m2, self._accum_n = welford_update(
            self._accum_mean, self._accum_m2, self._accum_n, frame_np)
        self.spp_counter = self._accum_n

    def _variance_display(self) -> np.ndarray:
        """Per-pixel sample variance, normalized by its 99th percentile."""
        var = self._accum_m2 / float(self._accum_n - 1)
        vmax = float(np.percentile(var, 99.0))
        return np.clip(var / vmax, 0, 1).astype(np.float32) if vmax > 0 else np.zeros_like(var)

    # --- Rendering ---

    def _render_one_spp(self, guiding: bool, use_nrc: Optional[bool] = None) -> Optional[np.ndarray]:
        if self.scene is None:
            return None
        self.integrator.set_guiding(guiding)
        self.integrator.set_nrc(self.use_nrc_checkbox.isChecked() if use_nrc is None else use_nrc)
        with torch.inference_mode():
            img = custom_render(self.scene, sensor=self.main_sensor, spp=1, integrator=self.integrator,
                                seed=random.randint(0, 100000), progress=False)
        return img.cpu().numpy().astype(np.float32)

    def re_render_main_image(self, spp=1):
        if self.is_rendering or self.scene is None:
            return
        self.is_rendering = True
        try:
            self._reset_accumulator()
            for _ in range(max(1, int(spp))):
                self._welford_update(self._render_one_spp(guiding=self.guiding_checkbox.isChecked()))
                self.original_main_image_np = self._accum_mean.copy()
                self._update_main_view_with_marker()
        finally:
            self.is_rendering = False

    def _update_main_view_with_marker(self):
        if self.original_main_image_np is None:
            return
        if self.show_variance_checkbox.isChecked():
            if not (self.accumulate_checkbox.isChecked() and self._accum_n > 1):
                self.main_view.setText("Variance requires:\n- Accumulate Frames\n- > 1 spp")
                return
            image = self._variance_display()
        else:
            image = self.original_main_image_np
        if self.last_click_coords:
            image = draw_marker_on_image(image, *self.last_click_coords)

        pixmap = numpy_to_qpixmap(np.clip(image, 0, 1) ** (1 / 2.2))
        rect = self.main_view.contentsRect()
        if rect.width() > 0 and rect.height() > 0:
            pixmap = pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.main_view.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_main_view_with_marker()

    # --- Right panel generation ---

    def _render_hemisphere_rgb(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        sensor = mi.load_dict({"type": "gt_spherical_camera", "film": {
            "type": "hdrfilm", "width": self.HOVER_RESOLUTION[0], "height": self.HOVER_RESOLUTION[1]}})
        sensor.world_transform = mi.Transform4f().translate(si.spawn_ray(si.n).o)
        sensor.initial_si = si
        sensor.product_bsdf = self.bsdf_checkbox.isChecked()
        sensor.product_cosine = self.cosine_checkbox.isChecked()
        hemi = mi.render(self.scene, sensor=sensor, spp=16).numpy()
        return np.clip(np.transpose(hemi, (1, 0, 2)), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _hemi_rgb_to_pdf_viz(hemi_rgb: np.ndarray) -> np.ndarray:
        return cm.viridis(np.log1p(hemi_rgb.mean(axis=2)) * 10.0)[..., :3].astype(np.float32)

    @staticmethod
    def _guide_query(si: mi.SurfaceInteraction3f):
        """(position, local wi, normal, roughness) exactly as the integrator queries the guide."""
        return si.p.torch(), si.wi.torch(), si.n.torch(), torch.ones((1, 1), device="cuda")

    def _render_guiding_2d(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        img = plot_guiding_distribution(self.integrator.guide, si, *self._guide_query(si))
        return np.transpose(img, (1, 0, 2)).astype(np.float32)

    def _render_guiding_sphere_html(self, si: mi.SurfaceInteraction3f) -> str:
        if not self._sphere_supported():
            return ""
        try:
            return visualize_pdf_on_sphere_html(self.integrator.guide, *self._guide_query(si), res=50)
        except Exception:
            logger.exception("Failed to build the 3D sphere view")
            return ""

    def _render_nrc(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        try:
            nrc = plot_nrc_radiance(self.integrator.nrc_system, si, si.p.torch(), si.n.torch(),
                                    resolution=self.HOVER_RESOLUTION, exposure=1.5)
            return np.transpose(nrc, (1, 0, 2)).astype(np.float32)
        except Exception:
            logger.exception("Failed to update NRC visualization")
            return np.zeros((self.HOVER_RESOLUTION[1], self.HOVER_RESOLUTION[0], 3), dtype=np.float32)

    # --- Click handling and async panel updates ---

    def on_mouse_click(self, event: QMouseEvent):
        if self.is_rendering or self.scene is None:
            return
        if not self.visualizations_enabled:
            self.visualizations_enabled = True
            self.setWindowTitle(self._base_title)

        self.last_widget_click_pos = event.pos()
        si, x, y = self._get_interaction_from_pos(event.pos())
        if not dr.any(si.is_valid()):
            self.hover_view.setText("Miss (No surface hit)")
            return
        self.last_click_coords = (x, y)
        self._last_si = si
        self._update_main_view_with_marker()
        self.reprocess_last_click()

    def reprocess_last_click(self):
        if self._last_si is None or not dr.any(self._last_si.is_valid()):
            return
        self._last_vis_click_coords = None
        self._last_vis_flags = None
        self.hover_view.setText("Loading hemisphere PDF...")
        self.normalized_pdf_view.setText("Loading hemisphere sRGB...")
        self.vmf_view.setText("Loading guiding distribution...")
        self.nrc_view.setText("Loading NRC...")
        self._start_viz_job(True)

    def _start_viz_job(self, include_hemisphere: bool):
        self._click_job_id += 1
        self._click_job_full = include_hemisphere
        self._viz_busy = True
        self._click_worker = VizWorker(self._click_job_id, self, self._last_si, self.show_sphere_checkbox.isChecked(),
                                       include_hemisphere=include_hemisphere)
        self._click_worker.result_ready.connect(self._on_viz_result)
        self._click_worker.failed.connect(self._on_viz_failed)
        self._click_worker.start()

    def _on_viz_failed(self, job_id: int, msg: str):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False
        if self._click_job_full:
            self.hover_view.setText(f"Hemisphere error:\n{msg[:120]}")
            self.normalized_pdf_view.setText("Error")
        self.vmf_view.setText("Error")
        self.nrc_view.setText("Error")

    def _on_viz_result(self, job_id: int, out: Dict[str, Any]):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False
        if "hemi_rgb" in out:
            self.hover_view.setPixmap(numpy_to_qpixmap(out["hemi_pdf_rgb"]))
            self.normalized_pdf_view.setPixmap(numpy_to_qpixmap(np.clip(out["hemi_rgb"], 0, 1) ** (1 / 2.2)))
        self.vmf_view.setPixmap(numpy_to_qpixmap(out["vmf_rgb"]))
        self.nrc_view.setPixmap(numpy_to_qpixmap(out["nrc_rgb"]))

        self._last_sphere_html = out["vmf_html"]
        if self._sphere_supported() and self.show_sphere_checkbox.isChecked():
            self.vmf_sphere_view.setHtml(self._last_sphere_html)
        if self._click_job_full:
            self._last_vis_click_coords = self.last_click_coords
            self._last_vis_flags = self._current_vis_flags()

    def _current_vis_flags(self) -> Tuple:
        return (self.bsdf_checkbox.isChecked(), self.cosine_checkbox.isChecked(),
                self.guiding_method_combo.currentText(), self.use_nee_checkbox.isChecked())

    def _update_title_with_fps(self):
        parts = [self._base_title, self.fps_profiler.get_stats_string(),
                 "Guiding ON" if self.guiding_checkbox.isChecked() else "Guiding OFF"]
        if self.accumulate_checkbox.isChecked():
            parts.append(f"SPP: {self.spp_counter}")
        self.setWindowTitle(" | ".join(parts))

    # --- Real-time loop ---

    def realtime_render_loop(self):
        if self.is_rendering or self.scene is None:
            return
        self.is_rendering = True
        self.fps_profiler.begin_frame()
        train_guide = self.train_realtime_checkbox.isChecked()
        train_nrc = self.train_nrc_checkbox.isChecked()
        training = train_guide or train_nrc
        try:
            frame = self._render_one_spp(guiding=self.guiding_checkbox.isChecked(),
                                         use_nrc=False if training else None)
            if training:
                self._train_on_last_frame(train_guide, train_nrc)

            if self.accumulate_checkbox.isChecked():
                self._welford_update(frame)
                self.original_main_image_np = self._accum_mean.copy()
            else:
                self.original_main_image_np = frame
                self.spp_counter = 0
            self._update_main_view_with_marker()

            if self.visualizations_enabled and self.last_widget_click_pos is not None and not self._viz_busy:
                if (self._last_vis_click_coords != self.last_click_coords
                        or self._last_vis_flags != self._current_vis_flags()):
                    si, _, _ = self._get_interaction_from_pos(self.last_widget_click_pos)
                    if dr.any(si.is_valid()):
                        self._last_si = si
                        self.reprocess_last_click()
                elif training and time.time() - self._last_training_viz_refresh_t > 0.2:
                    # Networks changed: refresh guiding/NRC panels, at most every 0.2 s
                    self._start_viz_job(False)
                    self._last_training_viz_refresh_t = time.time()
        finally:
            self.fps_profiler.end_frame()
            self.is_rendering = False

    def toggle_realtime_render(self, state: int):
        if state:
            self.render_timer.start()
        else:
            self.render_timer.stop()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() != Qt.Key.Key_Escape:
            return super().keyPressEvent(event)
        self.visualizations_enabled = not self.visualizations_enabled
        self.setWindowTitle(self._base_title if self.visualizations_enabled else f"{self._base_title} [VIS OFF]")
        self.clear_selection()

    def clear_selection(self):
        self.last_click_coords = self.last_widget_click_pos = self._last_si = None
        self._last_vis_click_coords = self._last_vis_flags = None
        self._last_sphere_html = ""
        self._update_main_view_with_marker()
        self._set_panel_placeholders()
        if self._sphere_supported():
            self.vmf_sphere_view.setHtml("")

    # --- Training, settings and method switching ---

    def _open_train_dialog(self):
        dialog = TrainSettingsDialog(self, current_lr=self.integrator.guide.config.learning_rate)
        if dialog.exec():
            settings = dialog.get_values()
            self._run_training(settings["iterations"], settings["learningrate"])

    def _train_on_last_frame(self, train_guide: bool, train_nrc: bool) -> None:
        batch = self.integrator.make_batch(max_samples=16384)
        guide_loss, nrc_loss = self.integrator.train_step(batch, train_guide=train_guide, train_nrc=train_nrc)
        if guide_loss is not None:
            self.guiding_loss_plot.append(guide_loss)
        if nrc_loss is not None:
            self.nrc_loss_plot.append(nrc_loss)

    def _run_training(self, iterations: int, learning_rate: float):
        if self.scene is None:
            return
        for group in self.integrator.guide.optimizer.param_groups:
            group["lr"] = learning_rate
        for i in range(iterations):
            self._render_one_spp(guiding=False, use_nrc=False)
            self._train_on_last_frame(True, self.train_nrc_checkbox.isChecked())
            if (i + 1) % 10 == 0:
                self.train_button.setText(f"Training... ({i + 1}/{iterations})")
                QApplication.processEvents()
        self.train_button.setText("Train...")
        self.integrator.set_guiding(self.guiding_checkbox.isChecked())
        self.reprocess_last_click()

    def _open_settings_dialog(self):
        dialog = GuidingSettingsDialog(self, self.integrator.guide, self.MAIN_RESOLUTION)
        if not dialog.exec():
            return
        resolution = tuple(dialog.get_resolution())
        if resolution != self.MAIN_RESOLUTION:
            self.MAIN_RESOLUTION = resolution
            self.change_resolution(resolution)
            self._reset_accumulator()
            self.re_render_main_image(spp=1)
        for key, value in dialog.get_values().items():
            setattr(self.integrator.guide.config, key, value)

    def _on_guiding_method_changed(self, method_name: str):
        logger.info(f"Switching guiding method to: {method_name}")
        self._create_integrator(method_name)
        self.guiding_loss_plot.clear()
        self.nrc_loss_plot.clear()
        self._reset_accumulator()
        self._last_vis_flags = None
        self.re_render_main_image(spp=1)
        self.reprocess_last_click()

    # --- Ground-truth export ---

    def save_ground_truth_images(self):
        if self.scene is None:
            return
        dialog = SaveGtDialog(self, guiding_methods=registry.get_method_names(),
                              current_method=self.guiding_method_combo.currentText())
        if not dialog.exec():
            return
        settings = dialog.get_values()
        output_dir = settings["outputpath"]
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        do_unguided, do_guided = settings["render_unguided"], settings["render_guided"]

        was_running = self.render_timer.isActive()
        self.render_timer.stop()
        was_accumulating = self.accumulate_checkbox.isChecked()
        self.accumulate_checkbox.setChecked(False)
        original_res = self.MAIN_RESOLUTION
        target_res = (settings["width"], settings["height"])

        try:
            self.MAIN_RESOLUTION = target_res
            self.integrator.setup(self.scene, self._num_rays())
            self.change_resolution(target_res)
            self._reset_accumulator()

            progress = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            is_online, total_time, total_spp = settings["online"], settings["time"], settings["spp"]

            def phase(desc, budget_time, budget_spp, prog_range, guiding, train=False, accum=(None, None, 0)):
                return self._accumulate_phase(desc, is_online, budget_time, budget_spp, progress,
                                              prog_range, guiding, train, accum)

            if do_unguided:
                accum = phase("Baseline (Unguided)", total_time, total_spp, (0, 33) if do_guided else (0, 100),
                              guiding=False)
                self._save_image(os.path.join(output_dir, f"unguided_{timestamp}"), *accum)

            # Train (the guided samples count toward the image), then keep rendering guided
            if do_guided:
                train_share = settings["trainingbudgetpercent"] / 100.0
                train_time = total_time * train_share if is_online else 0
                train_spp = 0 if is_online else int(total_spp * train_share)

                method = settings["guidingmethod"]
                if method != self.guiding_method_combo.currentText():
                    self.guiding_method_combo.blockSignals(True)
                    self.guiding_method_combo.setCurrentText(method)
                    self.guiding_method_combo.blockSignals(False)
                    self._on_guiding_method_changed(method)

                accum = phase("Training Guiding", train_time, train_spp, (33, 66) if do_unguided else (0, 33),
                              guiding=True, train=True)
                self.integrator.bsdfSamplingFraction = settings["bsdfsamplingfraction"]
                accum = phase("Guided Render", total_time - train_time, total_spp - train_spp,
                              (66, 100) if do_unguided else (33, 100), guiding=True, accum=accum)
                self._save_image(os.path.join(output_dir, f"guided_{timestamp}"), *accum)

            progress.close()
            QMessageBox.information(self, "Success", f"Saved images to:\n{output_dir}")
        except Exception as e:
            logger.exception("Save GT failed")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")
        finally:
            self.MAIN_RESOLUTION = original_res
            self.change_resolution(original_res)
            self.integrator.setup(self.scene, self._num_rays())
            self.integrator.bsdfSamplingFraction = self.integrator.BSDF_SAMPLING_FRACTION
            self.accumulate_checkbox.setChecked(was_accumulating)
            self.integrator.set_guiding(self.guiding_checkbox.isChecked())
            self._reset_accumulator()
            if was_running:
                self.render_timer.start()

    def _accumulate_phase(self, desc, is_online, budget_time, budget_spp, progress, prog_range, guiding, train, accum):
        """Render (and optionally train) until the time or spp budget is spent; returns (mean, m2, n)."""
        mean, m2, n = accum
        n0, (lo, hi), start = n, prog_range, time.time()
        while True:
            elapsed, done = time.time() - start, n - n0
            budget, used = (budget_time, elapsed) if is_online else (budget_spp, done)
            if used >= budget:
                break
            progress.setValue(int(lo + min(1.0, used / budget) * (hi - lo)))
            progress.setLabelText(f"{desc}: {done} spp / {elapsed:.1f}s")
            if progress.wasCanceled():
                raise InterruptedError("Cancelled")

            frame = self._render_one_spp(guiding=guiding, use_nrc=False if train else None)
            mean, m2, n = welford_update(mean, m2, n, frame)
            if train:
                self._train_on_last_frame(True, self.train_nrc_checkbox.isChecked())
            self.original_main_image_np = mean
            self.spp_counter = n
            self._update_main_view_with_marker()
            QApplication.processEvents()
        return mean, m2, n

    def _save_image(self, path_prefix, img_np, m2_buffer, n_samples):
        """Write the mean image (EXR + PNG) and, with > 1 sample, its variance (EXR + log heatmap PNG)."""
        if img_np is None:
            img_np = np.zeros((self.MAIN_RESOLUTION[1], self.MAIN_RESOLUTION[0], 3), dtype=np.float32)
        img_np = np.asarray(img_np, dtype=np.float32)
        mi.Bitmap(img_np).write(f"{path_prefix}.exr")
        Image.fromarray((np.clip(img_np, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(f"{path_prefix}.png")

        if m2_buffer is not None and n_samples > 1:
            variance = m2_buffer / (n_samples - 1)
            mi.Bitmap(variance).write(f"{path_prefix}_variance.exr")
            var_log = np.log1p(variance.mean(axis=2))
            span = var_log.max() - var_log.min()
            var_norm = (var_log - var_log.min()) / span if span > 0 else np.zeros_like(var_log)
            Image.fromarray((cm.magma(var_norm)[..., :3] * 255).astype(np.uint8)).save(f"{path_prefix}_variance.png")
