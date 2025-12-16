# app.py
#
# Most of is it generated using Copilot as GUI was not the main focus.

import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, List

import drjit as dr
import matplotlib.cm as cm
import mitsuba as mi
import numpy as np
import torch
from PIL import Image

from PyQt6.QtCore import QTimer, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QColor, QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QMessageBox,
    QProgressDialog,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:
    QWebEngineView = None
    HAS_WEBENGINE = False

from guiding.config import get_config, get_logger, FPSProfiler, VisualizerConfig, NUMBA_AVAILABLE
from rendering.renderer import Renderer
from guiding.training_data import prepare_shared_training_data

# ---- robust helper import (project uses multiple names across files) ----
def _import_vis_helpers():
    try:
        from utils.visualization_helpers import (
            visualize_pdf_on_sphere_html,
            plot_nrc_radiance,
            plot_guiding_distribution,
            numpy_to_qpixmap,
            draw_marker_on_image,
        )
        return (
            visualize_pdf_on_sphere_html,
            plot_nrc_radiance,
            plot_guiding_distribution,
            numpy_to_qpixmap,
            draw_marker_on_image,
        )
    except Exception:
        pass

    try:
        from visualization_helpers import (
            visualizepdfonspherehtml,
            plotnrcradiance,
            plotguidingdistribution,
            numpytoqpixmap,
            drawmarkeronimage,
        )
        return (
            visualizepdfonspherehtml,
            plotnrcradiance,
            plotguidingdistribution,
            numpytoqpixmap,
            drawmarkeronimage,
        )
    except Exception:
        raise


(
    visualize_pdf_on_sphere_html,
    plot_nrc_radiance,
    plot_guiding_distribution,
    numpy_to_qpixmap,
    draw_marker_on_image,
) = _import_vis_helpers()

import sensor.spherical as spherical  # noqa: F401 (registers gt_spherical_camera)
import guiding.registry as registry

from ui.dialogs import SaveGtDialog, TrainSettingsDialog, GuidingSettingsDialog, TrainingDataDialog

mi.set_variant("cuda_ad_rgb")
mi.set_log_level(mi.LogLevel.Warn)

logger = get_logger("visualizer")
device = "cuda" if torch.cuda.is_available() else "cpu"

custom_render = Renderer().render

class GuidingNrcVizWorker(QThread):
    result_ready = pyqtSignal(int, object)  # job_id, dict
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, viewer: "MitsubaViewer", si: mi.SurfaceInteraction3f):
        super().__init__()
        self.job_id = job_id
        self.viewer = viewer
        self.si = si

    def run(self):
        try:
            out: Dict[str, Any] = {}
            out["vmf_rgb"] = self.viewer._render_guiding_2d(self.si)
            out["nrc_rgb"] = self.viewer._render_nrc(self.si)
            if self.viewer._sphere_supported():
                out["vmf_html"] = self.viewer._render_guiding_sphere_html(self.si)
            else:
                out["vmf_html"] = ""
            self.result_ready.emit(self.job_id, out)
        except Exception as e:
            self.failed.emit(self.job_id, str(e))


class LossPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._loss: List[float] = []
        self._max_points = 2000
        self.setMinimumSize(256, 256)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")

    def append(self, value: float):
        if value is None:
            return
        try:
            v = float(value)
        except Exception:
            return
        self._loss.append(v)
        if len(self._loss) > self._max_points:
            self._loss = self._loss[-self._max_points :]
        self.update()

    def clear(self):
        self._loss.clear()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = max(1, self.width())
        h = max(1, self.height())

        p.setPen(QPen(QColor("#333"), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        p.setPen(QPen(QColor("#aaa"), 1))
        p.drawText(8, 16, "Loss")

        if len(self._loss) < 2:
            p.setPen(QPen(QColor("#666"), 1))
            p.drawText(8, 34, "No data")
            return

        y = np.asarray(self._loss, dtype=np.float64)
        y = y[np.isfinite(y)]
        if y.size < 2:
            return

        y_lo = float(np.percentile(y, 2))
        y_hi = float(np.percentile(y, 98))
        if not np.isfinite(y_lo):
            y_lo = float(np.min(y))
        if not np.isfinite(y_hi):
            y_hi = float(np.max(y))
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0

        x0, y0 = 6, 24
        x1, y1 = w - 6, h - 6
        n = len(self._loss)

        xs = np.linspace(x0, x1, n)
        yy = np.asarray(self._loss, dtype=np.float64)
        yy = np.clip(yy, y_lo, y_hi)
        ys = y1 - (yy - y_lo) / (y_hi - y_lo) * (y1 - y0)

        if y_lo < 0.0 < y_hi:
            z = y1 - (0.0 - y_lo) / (y_hi - y_lo) * (y1 - y0)
            p.setPen(QPen(QColor("#444"), 1))
            p.drawLine(int(x0), int(z), int(x1), int(z))

        p.setPen(QPen(QColor("#4cc9f0"), 2))
        for i in range(n - 1):
            p.drawLine(int(xs[i]), int(ys[i]), int(xs[i + 1]), int(ys[i + 1]))

        p.setPen(QPen(QColor("#bbb"), 1))
        p.drawText(8, h - 10, f"{float(self._loss[-1]):.6f}")


class ClickVizWorker(QThread):
    result_ready = pyqtSignal(int, object)  # job_id, dict
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, viewer: "MitsubaViewer", si: mi.SurfaceInteraction3f):
        super().__init__()
        self.job_id = job_id
        self.viewer = viewer
        self.si = si

    def run(self):
        try:
            out: Dict[str, Any] = {}
            hemi_rgb = self.viewer._render_hemisphere_rgb(self.si)
            out["hemi_rgb"] = hemi_rgb
            out["hemi_pdf_rgb"] = self.viewer._hemi_rgb_to_pdf_viz(hemi_rgb)
            out["vmf_rgb"] = self.viewer._render_guiding_2d(self.si)
            out["nrc_rgb"] = self.viewer._render_nrc(self.si)

            if self.viewer._sphere_supported():
                out["vmf_html"] = self.viewer._render_guiding_sphere_html(self.si)
            else:
                out["vmf_html"] = ""

            self.result_ready.emit(self.job_id, out)
        except Exception as e:
            self.failed.emit(self.job_id, str(e))


class MitsubaViewer(QMainWindow):
    def __init__(self, config: Optional[VisualizerConfig] = None):
        super().__init__()

        if config is None:
            config = get_config().visualizer

        self.config = config
        self.MAIN_RESOLUTION = tuple(config.main_resolution)
        self.HOVER_RESOLUTION = tuple(config.hover_resolution)

        self.fps_profiler = FPSProfiler(
            update_interval=getattr(config, "fps_update_interval", 0.5),
            smoothing=getattr(config, "fps_smoothing", 0.9),
        )

        self._base_title = "Mitsuba Path Guiding Viewer"
        self.setWindowTitle(self._base_title)

        # state
        self.scene = None
        self.integrator = None
        self.guiding_system = None
        self.nrc_system = None
        self.main_sensor = None
        self.hover_sensor = None
        self.original_main_image_np = None

        self.last_click_coords: Optional[Tuple[int, int]] = None
        self.last_widget_click_pos: Optional[QPoint] = None
        self._last_si: Optional[mi.SurfaceInteraction3f] = None

        self.is_rendering = False
        self.visualizations_enabled = True

        # accumulation (variance)
        self.spp_counter = 0
        self._accum_n = 0
        self._accum_mean = None
        self._accum_m2 = None

        # click viz async
        self._click_job_id = 0
        self._click_worker: Optional[ClickVizWorker] = None
        self._viz_busy = False

        # right panel cache keys
        self._last_vis_click_coords: Optional[Tuple[int, int]] = None
        self._last_vis_flags: Optional[Tuple] = None

        # cached sphere HTML (for toggle)
        self._last_sphere_html: str = ""

        # ---- training data (VRAM-safe) ----
        # Keep only a CPU snapshot for UI. Never store GPU training batches here.
        self._last_training_batch_cpu = None
        self._last_training_batch_spp: Optional[int] = None

        self._init_ui()
        self._init_renderer()
        self._connect_signals()

        logger.info("MitsubaViewer initialized.")

        self._training_viz_dirty = False
        self._last_training_viz_refresh_t = 0.0

    # -------------------------------------------------------------------------
    # Training batch CPU caching (VRAM-safe)
    # -------------------------------------------------------------------------

    def _cache_training_batch_cpu(self, batch, current_spp: Optional[int] = None):
        """
        Store a CPU snapshot for visualization.

        - Float tensors -> float16 on CPU.
        - Int/bool tensors -> keep dtype on CPU.
        """
        if batch is None or not hasattr(batch, "num_samples") or int(getattr(batch, "num_samples", 0)) <= 0:
            self._last_training_batch_cpu = None
            self._last_training_batch_spp = current_spp
            return

        def _to_cpu_tensor(t: torch.Tensor):
            if not torch.is_tensor(t) or t.numel() == 0:
                return t
            tt = t.detach()
            if tt.dtype in (torch.float32, torch.float64):
                return tt.to("cpu", non_blocking=True).to(torch.float16)
            return tt.to("cpu", non_blocking=True)

        cpu_fields = {}
        for k, v in vars(batch).items():
            if torch.is_tensor(v):
                cpu_fields[k] = _to_cpu_tensor(v)
            else:
                cpu_fields[k] = v

        # SimpleNamespace gives attribute access without importing TrainingBatch type here.
        self._last_training_batch_cpu = SimpleNamespace(**cpu_fields)
        self._last_training_batch_spp = current_spp

        # Drop GPU refs aggressively (VRAM friendliness; allocator may still reserve).
        try:
            del batch
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _init_ui(self):
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        top = QHBoxLayout()

        self.browse_button = QPushButton("Browse Scene")
        self.train_button = QPushButton("Train...")
        self.save_gt_button = QPushButton("Save GT Images...")
        self.settings_button = QPushButton("Settings...")
        self.training_data_button = QPushButton("Training Data...")

        self.guiding_method_label = QLabel("Guiding:")
        self.guiding_method_combo = QComboBox()
        for m in registry.get_method_names():
            self.guiding_method_combo.addItem(m)

        default_method = registry.get_default_method()
        if default_method:
            idx = self.guiding_method_combo.findText(default_method)
            if idx >= 0:
                self.guiding_method_combo.setCurrentIndex(idx)

        top.addWidget(self.browse_button)
        top.addWidget(self.train_button)
        top.addWidget(self.save_gt_button)
        top.addWidget(self.settings_button)
        top.addWidget(self.training_data_button)
        top.addWidget(self.guiding_method_label)
        top.addWidget(self.guiding_method_combo)
        top.addStretch()

        self.main_layout.addLayout(top)

        cbox = QHBoxLayout()
        self.cosine_checkbox = QCheckBox("Cosine Product")
        self.bsdf_checkbox = QCheckBox("BSDF Product")
        self.guiding_checkbox = QCheckBox("Use Path Guiding")
        self.use_nee_checkbox = QCheckBox("Use NEE")
        self.realtime_checkbox = QCheckBox("Real-Time Render")
        self.train_realtime_checkbox = QCheckBox("Train in Real-Time")
        self.train_nrc_checkbox = QCheckBox("Train NRC")
        self.accumulate_checkbox = QCheckBox("Accumulate Frames")
        self.show_variance_checkbox = QCheckBox("Show Variance")
        self.show_sphere_checkbox = QCheckBox("3D Sphere")

        self.show_sphere_checkbox.setChecked(True)
        self.bsdf_checkbox.setChecked(True)
        self.use_nee_checkbox.setChecked(False)

        for w in [
            self.cosine_checkbox,
            self.bsdf_checkbox,
            self.guiding_checkbox,
            self.use_nee_checkbox,
            self.realtime_checkbox,
            self.train_realtime_checkbox,
            self.train_nrc_checkbox,
            self.accumulate_checkbox,
            self.show_variance_checkbox,
            self.show_sphere_checkbox,
        ]:
            cbox.addWidget(w)

        cbox.addStretch()
        self.main_layout.addLayout(cbox)

        vis_layout = QHBoxLayout()

        self.main_view = QLabel("Click 'Browse Scene' to load a scene")
        self.main_view.setMinimumSize(self.MAIN_RESOLUTION[0] // 2, self.MAIN_RESOLUTION[1] // 2)
        self.main_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_view.setStyleSheet("QLabel { background-color: #101010; color: white; border: 1px solid #333; }")
        vis_layout.addWidget(self.main_view, stretch=2)

        detail_grid = QGridLayout()
        detail_grid.setSpacing(5)

        self.hover_view = QLabel("Hemisphere PDF (Viridis)\n(Click main render to inspect)")
        self.normalized_pdf_view = QLabel("Hemisphere RGB (sRGB)")
        self.vmf_view = QLabel("Guiding Distribution")
        self.nrc_view = QLabel("NRC Radiance")
        self.loss_plot = LossPlotWidget()

        for view in [self.hover_view, self.normalized_pdf_view, self.vmf_view, self.nrc_view]:
            view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setStyleSheet("QLabel { background-color: #1a1a1a; color: white; border: 1px solid #333; }")

        if HAS_WEBENGINE:
            self.vmf_sphere_view = QWebEngineView()
            self.vmf_sphere_view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
            self.vmf_sphere_view.page().setBackgroundColor(QColor("black"))
        else:
            self.vmf_sphere_view = QLabel("QWebEngineView not available")
            self.vmf_sphere_view.setMinimumSize(self.HOVER_RESOLUTION[0], self.HOVER_RESOLUTION[1])
            self.vmf_sphere_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vmf_sphere_view.setStyleSheet("QLabel { background-color: #000; color: #bbb; border: 1px solid #333; }")

        detail_grid.addWidget(self.hover_view, 0, 0)
        detail_grid.addWidget(self.normalized_pdf_view, 0, 1)
        detail_grid.addWidget(self.vmf_view, 0, 2)
        detail_grid.addWidget(self.nrc_view, 1, 0)
        detail_grid.addWidget(self.vmf_sphere_view, 1, 1)
        detail_grid.addWidget(self.loss_plot, 1, 2)

        vis_layout.addLayout(detail_grid, stretch=3)
        self.main_layout.addLayout(vis_layout)

        self.statusBar().showMessage("Press Esc to toggle visualizations for faster training")

        self.render_timer = QTimer(self)
        self.render_timer.setInterval(int(getattr(self.config, "realtime_interval_ms", 30)))

        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(int(getattr(self.config, "fps_update_interval", 0.5) * 1000))
        self.fps_timer.timeout.connect(self._update_title_with_fps)
        self.fps_timer.start()

        self._reset_accumulator()
        self._apply_sphere_visibility()

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

        self.realtime_checkbox.stateChanged.connect(self.toggle_realtime_render)
        self.accumulate_checkbox.stateChanged.connect(self._reset_accumulator)
        self.show_variance_checkbox.stateChanged.connect(lambda _: self._update_main_view_with_marker())

        self.show_sphere_checkbox.stateChanged.connect(self._on_sphere_toggled)
        self.guiding_method_combo.currentTextChanged.connect(self._on_guiding_method_changed)

        self.render_timer.timeout.connect(self.realtime_render_loop)
        self.main_view.mousePressEvent = self.on_mouse_click

    def _open_training_data_dialog(self):
        if self.scene is None or self.integrator is None:
            QMessageBox.information(self, "Training Data", "No scene/integrator loaded.")
            return

        # If training compacted the record, refresh once so it's back to (H*W*D) layout.
        if bool(getattr(self.integrator, "datascattered", False)):
            _ = self._render_one_spp(guiding=self.guiding_checkbox.isChecked())

        if getattr(self.integrator, "surfaceInteractionRecord", None) is None:
            QMessageBox.information(self, "Training Data", "No surfaceInteractionRecord yet. Render once.")
            return

        dialog = TrainingDataDialog(
            self,
            integrator=self.integrator,
            resolution=self.MAIN_RESOLUTION,
            current_spp=self.spp_counter,
        )
        dialog.exec()

    # -------------------------------------------------------------------------
    # Sphere toggling
    # -------------------------------------------------------------------------

    def _sphere_supported(self) -> bool:
        return bool(HAS_WEBENGINE and isinstance(self.vmf_sphere_view, QWebEngineView))

    def _apply_sphere_visibility(self):
        enabled = self.show_sphere_checkbox.isChecked()
        if self.vmf_sphere_view is None:
            return
        self.vmf_sphere_view.setVisible(bool(enabled))
        if enabled and self._sphere_supported() and self._last_sphere_html:
            self.vmf_sphere_view.setHtml(self._last_sphere_html)

    def _on_sphere_toggled(self, state: int):
        self._apply_sphere_visibility()

    # -------------------------------------------------------------------------
    # Cache invalidation for right-panel
    # -------------------------------------------------------------------------

    def _on_viz_setting_changed(self):
        self._last_vis_flags = None
        self.reprocess_last_click()

    # -------------------------------------------------------------------------
    # Integrator init / scene load
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

    def _init_renderer(self):
        self.integrator = mi.load_dict({"type": "path_guiding_integrator"})
        self.guiding_system = getattr(self.integrator, "guiding_system", None)
        self.nrc_system = getattr(self.integrator, "nrc_system", None)
        self._ensure_integrator_use_nee()
        self.load_default_scene()

    def _ensure_integrator_use_nee(self):
        if self.integrator is None:
            return
        v = bool(getattr(self.integrator, "usenee", False))
        self.use_nee_checkbox.setChecked(v)

    def _on_nee_toggled(self):
        if self.integrator is None:
            return
        setattr(self.integrator, "usenee", bool(self.use_nee_checkbox.isChecked()))

    def load_default_scene(self):
        self.load_scene(scene_path=None)

    def load_scene_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mitsuba Scene", "", "XML Files (*.xml)")
        if file_path:
            self.load_scene(file_path)

    def load_scene(self, scene_path=None):
        try:
            if scene_path:
                self.scene = mi.load_file(scene_path)
            else:
                cbox = mi.cornell_box()
                cbox["integrator"]["max_depth"] = 5
                cbox["sensor"]["film"]["width"], cbox["sensor"]["film"]["height"] = self.MAIN_RESOLUTION
                self._transform_light(cbox)
                
                self.scene = mi.load_dict(cbox)

            self.main_sensor = self.scene.sensors()[0]
            sp = mi.traverse(self.main_sensor)
            sp["film.size"] = self.MAIN_RESOLUTION
            sp.update()

            num_rays = self.MAIN_RESOLUTION[0] * self.MAIN_RESOLUTION[1]
            bbox = self.scene.bbox()

            try:
                self.integrator.setup(scene=self.scene, numrays=num_rays, bboxmin=bbox.min, bboxmax=bbox.max)
            except TypeError:
                self.integrator.setup(scene=self.scene, num_rays=num_rays, bbox_min=bbox.min, bbox_max=bbox.max)

            self._reset_accumulator()
            self.re_render_main_image(spp=16)

        except Exception as e:
            logger.exception("Failed to load scene")
            self.main_view.setText(f"Failed to load scene:\n{e}")

    def change_resolution(self, resolution: Tuple[int, int]):
        if self.main_sensor is None:
            return
        sp = mi.traverse(self.main_sensor)
        sp["film.size"] = resolution
        sp.update()

    # -------------------------------------------------------------------------
    # Film click mapping (KeepAspectRatio-safe)
    # -------------------------------------------------------------------------

    def _get_display_rect_in_label(self) -> Tuple[float, float, float, float]:
        rect = self.main_view.contentsRect()
        rect_w = float(rect.width())
        rect_h = float(rect.height())

        film_size = self.main_sensor.film().size()
        film_w = float(film_size[0])
        film_h = float(film_size[1])

        if film_w <= 0 or film_h <= 0 or rect_w <= 0 or rect_h <= 0:
            return 0.0, 0.0, 0.0, 0.0

        scale = min(rect_w / film_w, rect_h / film_h)
        disp_w = film_w * scale
        disp_h = film_h * scale
        offset_x = float(rect.x()) + (rect_w - disp_w) * 0.5
        offset_y = float(rect.y()) + (rect_h - disp_h) * 0.5
        return offset_x, offset_y, disp_w, disp_h

    def _get_interaction_from_pos(self, q_pos: QPoint):
        xw, yw = float(q_pos.x()), float(q_pos.y())
        ox, oy, dw, dh = self._get_display_rect_in_label()
        if dw <= 0 or dh <= 0:
            return mi.SurfaceInteraction3f(), -1, -1
        if not (ox <= xw < ox + dw and oy <= yw < oy + dh):
            return mi.SurfaceInteraction3f(), -1, -1

        film_size = self.main_sensor.film().size()
        film_w, film_h = int(film_size[0]), int(film_size[1])

        u_img = (xw - ox) / dw
        v_img = (yw - oy) / dh

        x = int(u_img * film_w)
        y = int(v_img * film_h)
        x = max(0, min(x, film_w - 1))
        y = max(0, min(y, film_h - 1))

        u, v = (x + 0.5) / film_w, (y + 0.5) / film_h
        ray, _ = self.main_sensor.sample_ray(0.0, 0.5, [u, v], [0.5, 0.5])
        return self.scene.ray_intersect(ray), x, y

    # -------------------------------------------------------------------------
    # Accumulation + manual variance
    # -------------------------------------------------------------------------

    def _reset_accumulator(self):
        h, w = self.MAIN_RESOLUTION[1], self.MAIN_RESOLUTION[0]
        self.spp_counter = 0
        self._accum_n = 0
        self._accum_mean = np.zeros((h, w, 3), dtype=np.float32)
        self._accum_m2 = np.zeros((h, w, 3), dtype=np.float32)
        self.original_main_image_np = self._accum_mean.copy()
        self._update_main_view_with_marker()

    def _welford_update(self, frame_np: np.ndarray):
        if frame_np is None:
            return
        self._accum_n += 1
        self.spp_counter = self._accum_n
        delta = frame_np - self._accum_mean
        self._accum_mean += delta / float(self._accum_n)
        delta2 = frame_np - self._accum_mean
        self._accum_m2 += delta * delta2

    def _manual_variance_raw(self) -> np.ndarray:
        if self._accum_n <= 1:
            return np.zeros_like(self._accum_mean)
        return self._accum_m2 / float(self._accum_n - 1)

    def _manual_variance_display(self) -> np.ndarray:
        var = self._manual_variance_raw()
        vmax = float(np.percentile(var, 99.0))
        if vmax > 0:
            return np.clip(var / vmax, 0, 1).astype(np.float32)
        return np.zeros_like(var).astype(np.float32)

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _render_one_spp(self, guiding: bool) -> np.ndarray:
        if self.scene is None:
            return None

        if hasattr(self.integrator, "set_guiding"):
            self.integrator.set_guiding(bool(guiding))
        elif hasattr(self.integrator, "setGuiding"):
            self.integrator.setGuiding(bool(guiding))
        else:
            setattr(self.integrator, "guiding", bool(guiding))

        with torch.inference_mode():
            img = custom_render(
                self.scene,
                sensor=self.main_sensor,
                spp=1,
                integrator=self.integrator,
                seed=random.randint(0, 100000),
                guiding=bool(guiding),
                progress=False,
            )
            return img.detach().cpu().numpy().astype(np.float32)

    def re_render_main_image(self, spp=1):
        if self.is_rendering or self.scene is None:
            return
        self.is_rendering = True
        try:
            self._reset_accumulator()
            use_guiding = self.guiding_checkbox.isChecked()
            spp = int(max(1, spp))
            for _ in range(spp):
                frame_np = self._render_one_spp(guiding=use_guiding)
                self._welford_update(frame_np)
                self.original_main_image_np = self._accum_mean.copy()
                self._update_main_view_with_marker()
        finally:
            self.is_rendering = False

    def _update_main_view_with_marker(self):
        if self.original_main_image_np is None:
            return

        if self.show_variance_checkbox.isChecked():
            if self.accumulate_checkbox.isChecked() and self._accum_n > 1:
                image_to_show = self._manual_variance_display()
            else:
                self.main_view.setText("Variance requires:\n- Accumulate Frames\n- > 1 spp")
                return
        else:
            image_to_show = self.original_main_image_np

        if self.last_click_coords:
            x, y = self.last_click_coords
            image_to_show = draw_marker_on_image(image_to_show.copy(), x, y)

        img_disp = np.clip(image_to_show, 0, 1) ** (1 / 2.2)
        pixmap = numpy_to_qpixmap(img_disp)
        rect = self.main_view.contentsRect()
        if rect.width() > 0 and rect.height() > 0:
            pixmap = pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.main_view.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_main_view_with_marker()

    # -------------------------------------------------------------------------
    # Right panel generation
    # -------------------------------------------------------------------------

    def _render_hemisphere_rgb(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        sensor = mi.load_dict(
            {
                "type": "gt_spherical_camera",
                "film": {
                    "type": "hdrfilm",
                    "width": self.HOVER_RESOLUTION[0],
                    "height": self.HOVER_RESOLUTION[1],
                },
            }
        )

        dummy_ray = si.spawn_ray(si.n)
        sensor.world_transform = mi.Transform4f.translate(dummy_ray.o)
        sensor.initial_si = si

        sensor.product_bsdf = self.bsdf_checkbox.isChecked()
        sensor.product_cosine = self.cosine_checkbox.isChecked()

        gt_image_np = mi.render(self.scene, sensor=sensor, spp=128).numpy()
        hemi = np.transpose(gt_image_np, (1, 0, 2)).astype(np.float32)
        return np.clip(hemi, 0.0, 1.0)

    def _hemi_rgb_to_pdf_viz(self, hemi_rgb: np.ndarray) -> np.ndarray:
        luminance = np.mean(hemi_rgb, axis=2)
        log_luminance = np.log1p(luminance)
        visualization_scale = 10.0
        scaled = log_luminance * visualization_scale
        colored_rgb = cm.viridis(scaled)[..., :3].astype(np.float32)
        return np.clip(colored_rgb, 0.0, 1.0)

    def _render_guiding_2d(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        if self.guiding_system is None:
            return np.zeros((self.HOVER_RESOLUTION[1], self.HOVER_RESOLUTION[0], 3), dtype=np.float32)

        ray = si.spawn_ray(dr.zeros(mi.Vector3f))
        position = si.p.torch()
        wi = (-ray.d).torch()
        roughness = torch.tensor([[1.0]], device=device)

        img = plot_guiding_distribution(self.guiding_system, si, position, wi, roughness, device=device)
        return np.transpose(img, (1, 0, 2)).astype(np.float32)

    def _render_guiding_sphere_html(self, si: mi.SurfaceInteraction3f) -> str:
        if not self._sphere_supported() or self.guiding_system is None:
            return ""
        try:
            ray = si.spawn_ray(dr.zeros(mi.Vector3f))
            position = si.p.torch()
            wi = (-ray.d).torch()
            roughness = torch.tensor([[1.0]], device=device)
            vis_dist = self.guiding_system.get_distribution_for_visualization(position, wi, roughness)
            return visualize_pdf_on_sphere_html(vis_dist, res=50)
        except Exception:
            return ""

    def _render_nrc(self, si: mi.SurfaceInteraction3f) -> np.ndarray:
        if self.nrc_system is None:
            return np.zeros((self.HOVER_RESOLUTION[1], self.HOVER_RESOLUTION[0], 3), dtype=np.float32)

        try:
            position = si.p.torch()
            normal = si.n.torch()

            bbox_min = None
            bbox_max = None
            if hasattr(self.integrator, "bboxmin"):
                bbox_min = self.integrator.bboxmin.torch().to(device)
            if hasattr(self.integrator, "bboxmax"):
                bbox_max = self.integrator.bboxmax.torch().to(device)
            if bbox_min is None and hasattr(self.integrator, "bbox_min"):
                bbox_min = self.integrator.bbox_min.torch().to(device)
            if bbox_max is None and hasattr(self.integrator, "bbox_max"):
                bbox_max = self.integrator.bbox_max.torch().to(device)

            nrc = plot_nrc_radiance(
                self.nrc_system,
                si,
                position,
                normal,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                resolution=self.HOVER_RESOLUTION,
                device=device,
            )
            return np.transpose(nrc, (1, 0, 2)).astype(np.float32)
        except Exception:
            logger.exception("Failed to update NRC visualization")
            return np.zeros((self.HOVER_RESOLUTION[1], self.HOVER_RESOLUTION[0], 3), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Click handling + async update
    # -------------------------------------------------------------------------

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

        self._last_vis_click_coords = None
        self._last_vis_flags = None

        self.hover_view.setText("Loading hemisphere PDF...")
        self.normalized_pdf_view.setText("Loading hemisphere sRGB...")
        self.vmf_view.setText("Loading guiding distribution...")
        self.nrc_view.setText("Loading NRC...")

        self._click_job_id += 1
        job_id = self._click_job_id
        self._viz_busy = True

        self._click_worker = ClickVizWorker(job_id, self, si)
        self._click_worker.result_ready.connect(self._on_click_viz_result)
        self._click_worker.failed.connect(self._on_click_viz_failed)
        self._click_worker.start()

    def reprocess_last_click(self):
        if self._last_si is None or (not dr.any(self._last_si.is_valid())):
            return

        self._last_vis_click_coords = None
        self._last_vis_flags = None

        self.hover_view.setText("Loading hemisphere PDF...")
        self.normalized_pdf_view.setText("Loading hemisphere sRGB...")
        self.vmf_view.setText("Loading guiding distribution...")
        self.nrc_view.setText("Loading NRC...")

        self._click_job_id += 1
        job_id = self._click_job_id
        self._viz_busy = True

        self._click_worker = ClickVizWorker(job_id, self, self._last_si)
        self._click_worker.result_ready.connect(self._on_click_viz_result)
        self._click_worker.failed.connect(self._on_click_viz_failed)
        self._click_worker.start()

    def _on_click_viz_failed(self, job_id: int, msg: str):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False
        self.hover_view.setText(f"Hemisphere error:\n{msg[:120]}")
        self.normalized_pdf_view.setText("Error")
        self.vmf_view.setText("Error")
        self.nrc_view.setText("Error")

    def _on_click_viz_result(self, job_id: int, out: Dict[str, Any]):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False

        hemi_pdf = out.get("hemi_pdf_rgb", None)
        hemi_rgb = out.get("hemi_rgb", None)
        vmf = out.get("vmf_rgb", None)
        nrc = out.get("nrc_rgb", None)
        html = out.get("vmf_html", "")

        if hemi_pdf is not None:
            self.hover_view.setPixmap(numpy_to_qpixmap(hemi_pdf))
        if hemi_rgb is not None:
            self.normalized_pdf_view.setPixmap(numpy_to_qpixmap(np.clip(hemi_rgb, 0, 1) ** (1 / 2.2)))
        if vmf is not None:
            self.vmf_view.setPixmap(numpy_to_qpixmap(vmf))
        if nrc is not None:
            self.nrc_view.setPixmap(numpy_to_qpixmap(nrc))

        self._last_sphere_html = html or ""
        if self._sphere_supported() and self.show_sphere_checkbox.isChecked():
            self.vmf_sphere_view.setHtml(self._last_sphere_html or "")

        self._last_vis_click_coords = self.last_click_coords
        self._last_vis_flags = (
            self.bsdf_checkbox.isChecked(),
            self.cosine_checkbox.isChecked(),
            self.guiding_method_combo.currentText(),
        )

    # -------------------------------------------------------------------------
    # FPS title
    # -------------------------------------------------------------------------

    def _update_title_with_fps(self):
        fps_string = self.fps_profiler.get_stats_string()
        guiding_status = "Guiding ON" if self.guiding_checkbox.isChecked() else "Guiding OFF"
        spp_info = f"SPP: {self.spp_counter}" if self.accumulate_checkbox.isChecked() else ""
        parts = [self._base_title, fps_string, guiding_status]
        if spp_info:
            parts.append(spp_info)
        self.setWindowTitle(" | ".join(parts))

    def _refresh_guiding_nrc_silent(self):
        """Update ONLY guiding + NRC panels. No 'Loading...' text, no hemisphere recompute."""
        if self._last_si is None or (not dr.any(self._last_si.is_valid())):
            return
        if self._viz_busy:
            return

        # job id stream separate from click is fine; reuse same counter
        self._click_job_id += 1
        job_id = self._click_job_id
        self._viz_busy = True

        self._click_worker = GuidingNrcVizWorker(job_id, self, self._last_si)
        self._click_worker.result_ready.connect(self._on_guiding_nrc_viz_result)
        self._click_worker.failed.connect(self._on_guiding_nrc_viz_failed)
        self._click_worker.start()


    def _on_guiding_nrc_viz_failed(self, job_id: int, msg: str):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False
        # Do NOT overwrite hemisphere panels; only mark the ones we updated
        self.vmf_view.setText("Error")
        self.nrc_view.setText("Error")


    def _on_guiding_nrc_viz_result(self, job_id: int, out: Dict[str, Any]):
        if job_id != self._click_job_id:
            return
        self._viz_busy = False

        vmf = out.get("vmf_rgb", None)
        nrc = out.get("nrc_rgb", None)
        html = out.get("vmf_html", "")

        if vmf is not None:
            self.vmf_view.setPixmap(numpy_to_qpixmap(vmf))
        if nrc is not None:
            self.nrc_view.setPixmap(numpy_to_qpixmap(nrc))

        self._last_sphere_html = html or ""
        if self._sphere_supported() and self.show_sphere_checkbox.isChecked():
            self.vmf_sphere_view.setHtml(self._last_sphere_html or "")

    # -------------------------------------------------------------------------
    # Real-time loop
    # -------------------------------------------------------------------------

    def realtime_render_loop(self):
        if self.is_rendering or self.scene is None:
            return

        # lazy init (no __init__ edits)
        if not hasattr(self, "_last_training_viz_refresh_t"):
            self._last_training_viz_refresh_t = 0.0

        self.is_rendering = True
        self.fps_profiler.begin_frame()

        use_guiding = self.guiding_checkbox.isChecked()
        do_training = self.train_realtime_checkbox.isChecked()
        do_nrc_training = self.train_nrc_checkbox.isChecked()
        do_accumulate = self.accumulate_checkbox.isChecked()

        try:
            frame_np = self._render_one_spp(guiding=use_guiding)

            training_batch = None
            did_train_step = False

            if do_training or do_nrc_training:
                training_batch = prepare_shared_training_data(self.integrator, device=device, max_samples=16384)

            if do_training and training_batch is not None and hasattr(self.guiding_system, "train_step_from_batch"):
                loss = self.guiding_system.train_step_from_batch(training_batch)
                self.loss_plot.append(loss)
                did_train_step = True

            if do_nrc_training and training_batch is not None and hasattr(self.nrc_system, "train_step_from_batch"):
                _ = self.nrc_system.train_step_from_batch(training_batch)
                did_train_step = True

            if training_batch is not None:
                self._cache_training_batch_cpu(training_batch, current_spp=self.spp_counter)

            if do_accumulate:
                self._welford_update(frame_np)
                self.original_main_image_np = self._accum_mean.copy()
            else:
                self.original_main_image_np = frame_np
                self.spp_counter = 0

            self._update_main_view_with_marker()

            # ---- right panel updates ----
            if (
                self.visualizations_enabled
                and (self.last_click_coords is not None)
                and (not self._viz_busy)
                and (self.last_widget_click_pos is not None)
            ):
                current_flags = (
                    self.bsdf_checkbox.isChecked(),
                    self.cosine_checkbox.isChecked(),
                    self.guiding_method_combo.currentText(),
                )

                point_or_flags_changed = (
                    (self._last_vis_click_coords != self.last_click_coords)
                    or (self._last_vis_flags != current_flags)
                )

                if point_or_flags_changed:
                    # full refresh (this is when "Loading..." is acceptable)
                    si, _, _ = self._get_interaction_from_pos(self.last_widget_click_pos)
                    if dr.any(si.is_valid()):
                        self._last_si = si
                        self.reprocess_last_click()
                else:
                    # training refresh: ONLY guiding+nrc, silently + throttled
                    if did_train_step:
                        now_t = time.time()
                        if (now_t - float(self._last_training_viz_refresh_t)) > 0.2:
                            self._refresh_guiding_nrc_silent()
                            self._last_training_viz_refresh_t = now_t

        finally:
            self.fps_profiler.end_frame()
            self.is_rendering = False

    def toggle_realtime_render(self, state: int):
        if state:
            self.render_timer.start()
        else:
            self.render_timer.stop()
        self.train_button.setText("Train...")

    # -------------------------------------------------------------------------
    # Key handling
    # -------------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.visualizations_enabled = not self.visualizations_enabled
            if self.visualizations_enabled:
                self.setWindowTitle(self._base_title)
            else:
                self.setWindowTitle(f"{self._base_title} [VIS OFF]")
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    def clear_selection(self):
        self.last_click_coords = None
        self.last_widget_click_pos = None
        self._last_si = None
        self._last_vis_click_coords = None
        self._last_vis_flags = None
        self._last_sphere_html = ""

        self._update_main_view_with_marker()
        self.hover_view.setText("Hemisphere PDF (Viridis)\n(Click main render to inspect)")
        self.normalized_pdf_view.setText("Hemisphere RGB (sRGB)")
        self.vmf_view.setText("Guiding Distribution")
        self.nrc_view.setText("NRC Radiance")
        if self._sphere_supported():
            self.vmf_sphere_view.setHtml("")

    # -------------------------------------------------------------------------
    # Dialogs / method switching / saving GT
    # -------------------------------------------------------------------------

    def _open_train_dialog(self):
        current_lr = 5e-3
        try:
            dist = getattr(self.guiding_system, "_distribution", None)
            cfg = getattr(dist, "config", None)
            if cfg is not None and hasattr(cfg, "learning_rate"):
                current_lr = float(getattr(cfg, "learning_rate"))
        except Exception:
            pass

        dialog = TrainSettingsDialog(self, current_lr=current_lr)
        if dialog.exec():
            settings = dialog.get_values()
            self._run_training(settings["iterations"], settings["learningrate"])

    def _run_training(self, iterations: int, learning_rate: Optional[float] = None):
        if self.scene is None or self.guiding_system is None:
            return

        try:
            dist = getattr(self.guiding_system, "_distribution", None)
            if dist is not None and hasattr(dist, "optimizer") and learning_rate is not None:
                for param_group in dist.optimizer.param_groups:
                    param_group["lr"] = float(learning_rate)
        except Exception:
            pass

        if hasattr(self.integrator, "set_guiding"):
            self.integrator.set_guiding(False)

        for i in range(int(iterations)):
            _ = self._render_one_spp(guiding=False)
            batch = prepare_shared_training_data(self.integrator, device=device, max_samples=16384)

            if hasattr(self.guiding_system, "train_step_from_batch"):
                loss = self.guiding_system.train_step_from_batch(batch)
                self.loss_plot.append(loss)

            # Cache CPU snapshot for UI (VRAM-safe)
            self._cache_training_batch_cpu(batch, current_spp=i + 1)

            if (i + 1) % 10 == 0:
                self.train_button.setText(f"Training... ({i+1}/{iterations})")
                QApplication.processEvents()

        self.train_button.setText("Train...")

        if hasattr(self.integrator, "set_guiding"):
            self.integrator.set_guiding(self.guiding_checkbox.isChecked())
        self.reprocess_last_click()

    def _open_settings_dialog(self):
        dialog = GuidingSettingsDialog(self, self.guiding_system, self.MAIN_RESOLUTION)
        if dialog.exec():
            new_resolution = dialog.get_resolution()
            if new_resolution != self.MAIN_RESOLUTION:
                self.MAIN_RESOLUTION = tuple(new_resolution)
                self.change_resolution(self.MAIN_RESOLUTION)
                self._reset_accumulator()
                if self.scene:
                    self.re_render_main_image(spp=1)

            values = dialog.get_values()
            if values and self.guiding_system:
                dist = getattr(self.guiding_system, "_distribution", None)
                cfg = getattr(dist, "config", None)
                if cfg is not None:
                    for key, value in values.items():
                        if hasattr(cfg, key):
                            setattr(cfg, key, value)

    def _on_guiding_method_changed(self, method_name: str):
        if self.guiding_system is None:
            return
        logger.info(f"Switching guiding method to: {method_name}")

        if hasattr(self.guiding_system, "switch_method"):
            self.guiding_system.switch_method(method_name)
        elif hasattr(self.guiding_system, "switchMethod"):
            self.guiding_system.switchMethod(method_name)

        self.loss_plot.clear()
        self._reset_accumulator()
        self._last_vis_flags = None

        if self.scene is not None:
            self.re_render_main_image(spp=1)
        self.reprocess_last_click()

    def save_ground_truth_images(self):
        if self.scene is None:
            return

        dialog = SaveGtDialog(
            self,
            guiding_methods=registry.get_method_names(),
            current_method=self.guiding_method_combo.currentText(),
        )
        if not dialog.exec():
            return

        settings = dialog.get_values()
        output_dir = settings.get("outputpath", "renders")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        do_unguided = settings.get("render_unguided", True)
        do_guided = settings.get("render_guided", True)

        # 1. Stop main loops
        was_running = self.render_timer.isActive()
        self.render_timer.stop()
        was_accumulating = self.accumulate_checkbox.isChecked()
        self.accumulate_checkbox.setChecked(False)

        original_res = self.MAIN_RESOLUTION
        target_res = (settings["width"], settings["height"])

        try:
            # 2. Resize
            logger.info(f"Resizing to {target_res} for GT generation...")
            self.MAIN_RESOLUTION = target_res # Update for accumulator
            bbox = self.scene.bbox()
            num_rays = target_res[0] * target_res[1]
            
            self.integrator.setup(
                self.scene, 
                num_rays=num_rays, 
                bbox_min=bbox.min, 
                bbox_max=bbox.max
            )
            self.change_resolution(target_res)
            self._reset_accumulator()
            
            # Setup Progress
            progress_bar = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
            progress_bar.setWindowModality(Qt.WindowModality.WindowModal)
            progress_bar.setMinimumDuration(0)
            progress_bar.resize(400, 100)
            progress_bar.show()

            is_online = settings["online"]
            total_time = settings.get("time", 60)
            total_spp = settings.get("spp", 256)
            
            # --- PHASE 1: BASELINE (UNGUIDED) ---
            if do_unguided:
                logger.info("Starting Baseline Render...")
                if hasattr(self.integrator, "set_guiding"):
                    self.integrator.set_guiding(False)
                
                img_unguided, _, m2_unguided, n_unguided = self._render_accumulated(
                    is_online=is_online,
                    budget_time=total_time,
                    budget_spp=total_spp,
                    progress_bar=progress_bar,
                    range_tuple=(0, 33) if do_guided else (0, 100),
                    desc="Baseline (Unguided)"
                )
                self._save_image(img_unguided, os.path.join(output_dir, f"unguided_{timestamp}"), m2_unguided, n_unguided)

            # --- PHASE 2 & 3: GUIDED ---
            if do_guided:
                train_percent = settings.get("trainingbudgetpercent", 20) / 100.0
                train_time = total_time * train_percent if is_online else 0
                train_spp = int(total_spp * train_percent) if not is_online else 0
                
                render_time = total_time - train_time
                render_spp = total_spp - train_spp
                
                req_method = settings.get("guidingmethod")
                if req_method and req_method != self.guiding_method_combo.currentText():
                     self.guiding_method_combo.blockSignals(True)
                     self.guiding_method_combo.setCurrentText(req_method)
                     self.guiding_method_combo.blockSignals(False)
                     self._on_guiding_method_changed(req_method)

                # Training
                logger.info(f"Starting Training (Budget: {train_percent*100:.1f}%)...")
                if hasattr(self.integrator, "set_guiding"):
                    self.integrator.set_guiding(True)

                accum_guided, samples_done, m2_guided, _ = self._run_training_phase(
                    is_online=is_online,
                    budget_time=train_time,
                    budget_spp=train_spp,
                    progress_bar=progress_bar,
                    range_tuple=(33, 66) if do_unguided else (0, 33),
                    desc="Training Guiding",
                    accum_buffer=None,
                    accum_m2=None
                )

                # Guided Rendering
                logger.info("Starting Guided Render...")
                render_bsdf_frac = settings.get("bsdfsamplingfraction", 0.5)
                
                if hasattr(self.integrator, "bsdfSamplingFraction"):
                    self.integrator.bsdfSamplingFraction = float(render_bsdf_frac)
                elif hasattr(self.integrator, "set_bsdf_sampling_fraction"):
                    self.integrator.set_bsdf_sampling_fraction(float(render_bsdf_frac))
                
                img_guided, _, m2_guided, n_guided = self._render_accumulated(
                    is_online=is_online,
                    budget_time=render_time,
                    budget_spp=render_spp,
                    progress_bar=progress_bar,
                    range_tuple=(66, 100) if do_unguided else (33, 100),
                    desc="Guided Render",
                    accum_buffer=accum_guided,
                    accum_m2=m2_guided,
                    current_spp=samples_done
                )
                self._save_image(img_guided, os.path.join(output_dir, f"guided_{timestamp}"), m2_guided, n_guided)

            progress_bar.close()
            QMessageBox.information(self, "Success", f"Saved images to:\n{output_dir}")

        except Exception as e:
            logger.exception("Save GT failed")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            
        finally:
            # Restore
            logger.info("Restoring resolution...")
            self.MAIN_RESOLUTION = original_res
            self.change_resolution(original_res)
            
            bbox = self.scene.bbox()
            num_rays = original_res[0] * original_res[1]
            self.integrator.setup(self.scene, num_rays=num_rays, bbox_min=bbox.min, bbox_max=bbox.max)

            self.accumulate_checkbox.setChecked(was_accumulating)
            if hasattr(self.integrator, "set_guiding"):
                self.integrator.set_guiding(self.guiding_checkbox.isChecked())
            self._reset_accumulator()
            if was_running:
                self.render_timer.start()

    def _render_accumulated(self, is_online, budget_time, budget_spp, progress_bar, range_tuple, desc="", accum_buffer=None, accum_m2=None, current_spp=0):
        start_min, start_max = range_tuple
        range_w = start_max - start_min
        
        accum_mean = accum_buffer
        accum_m2 = accum_m2
        accum_n = current_spp
        
        start_t = time.time()
        
        while True:
            now = time.time()
            elapsed = now - start_t
            
            if is_online:
                if elapsed >= budget_time: break
                prog = elapsed / budget_time if budget_time > 0 else 1.0
                status_str = f"{desc}: {elapsed:.1f}/{budget_time}s ({accum_n} spp)"
            else:
                target_spp = budget_spp + (current_spp if accum_buffer is not None else 0)
                if accum_n >= target_spp: break
                prog = (accum_n - current_spp) / budget_spp if budget_spp > 0 else 1.0
                status_str = f"{desc}: {accum_n}/{target_spp} spp ({elapsed:.1f}s)"

            prog = min(1.0, max(0.0, prog))
            val = int(start_min + prog * range_w)
            progress_bar.setValue(val)
            progress_bar.setLabelText(status_str)
            if progress_bar.wasCanceled(): raise InterruptedError("Cancelled")

            use_guiding = bool(getattr(self.integrator, "guiding", False))
            frame = self._render_one_spp(guiding=use_guiding)

            # Welford Accumulation (Mean + Variance)
            if accum_mean is None:
                accum_mean = np.zeros_like(frame, dtype=np.float32)
                accum_m2 = np.zeros_like(frame, dtype=np.float32)

            accum_n += 1
            delta = frame - accum_mean
            accum_mean += delta / accum_n
            delta2 = frame - accum_mean
            accum_m2 += delta * delta2

            # Visual Feedback
            self.original_main_image_np = accum_mean
            self.spp_counter = accum_n
            self._update_main_view_with_marker()
            QApplication.processEvents()
        
        if accum_mean is None:
            accum_mean = np.zeros((self.MAIN_RESOLUTION[1], self.MAIN_RESOLUTION[0], 3), dtype=np.float32)
            accum_m2 = np.zeros_like(accum_mean)

        return accum_mean, accum_n, accum_m2, accum_n

    def _run_training_phase(self, is_online, budget_time, budget_spp, progress_bar, range_tuple, desc="", accum_buffer=None, accum_m2=None):
        start_min, start_max = range_tuple
        range_w = start_max - start_min
        
        start_t = time.time()
        iters = 0
        
        accum_mean = accum_buffer
        accum_m2 = accum_m2
        # If continuing, accum_n needs to know previous count, but usually this is fresh start (0)
        accum_n = 0 
        
        while True:
            now = time.time()
            elapsed = now - start_t
            
            if is_online:
                if elapsed >= budget_time: break
                prog = elapsed / budget_time if budget_time > 0 else 1.0
            else:
                if iters >= budget_spp: break 
                prog = iters / budget_spp if budget_spp > 0 else 1.0
            
            val = int(start_min + prog * range_w)
            progress_bar.setValue(val)
            progress_bar.setLabelText(f"{desc}: {iters} iters / {elapsed:.1f}s")
            if progress_bar.wasCanceled(): raise InterruptedError()
            
            frame = self._render_one_spp(guiding=True)
            
            if accum_mean is None:
                accum_mean = np.zeros_like(frame, dtype=np.float32)
                accum_m2 = np.zeros_like(frame, dtype=np.float32)
            
            accum_n += 1
            delta = frame - accum_mean
            accum_mean += delta / accum_n
            delta2 = frame - accum_mean
            accum_m2 += delta * delta2
            
            batch = prepare_shared_training_data(self.integrator, device=device, max_samples=16384)
            if batch is not None and hasattr(self.guiding_system, "train_step_from_batch"):
                loss_val = self.guiding_system.train_step_from_batch(batch)
                self.loss_plot.append(loss_val)
            
            iters += 1
            
            self.original_main_image_np = accum_mean
            self.spp_counter = iters
            self._update_main_view_with_marker()
            QApplication.processEvents()

        return accum_mean, accum_n, accum_m2, accum_n

    def _save_image(self, img_np, path_prefix, m2_buffer=None, n_samples=0):
        img_np = np.asarray(img_np, dtype=np.float32)
        # 1. Save Radiance (EXR + PNG)
        mi.Bitmap(img_np).write(f"{path_prefix}.exr")
        img_srgb = (np.clip(img_np, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        Image.fromarray(img_srgb).save(f"{path_prefix}.png")
        
        # 2. Save Variance (EXR + Heatmap PNG)
        if m2_buffer is not None and n_samples > 1:
            # Variance = M2 / (n - 1)
            variance = m2_buffer / (n_samples - 1)
            mi.Bitmap(variance).write(f"{path_prefix}_variance.exr")
            
            # For visualization, average over channels and tone map
            # Use a heatmap for better visibility of noise distribution
            var_mono = np.mean(variance, axis=2)
            # Normalize for display: Log scale is usually best for variance
            var_log = np.log1p(var_mono)
            v_min, v_max = np.min(var_log), np.max(var_log)
            if v_max > v_min:
                var_norm = (var_log - v_min) / (v_max - v_min)
            else:
                var_norm = np.zeros_like(var_log)
            
            # Apply magma/inferno colormap
            colormap = cm.get_cmap('magma')
            var_colored = colormap(var_norm) # Returns RGBA
            var_uint8 = (var_colored[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(var_uint8).save(f"{path_prefix}_variance.png")

def main():
    app = QApplication(sys.argv)
    w = MitsubaViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
