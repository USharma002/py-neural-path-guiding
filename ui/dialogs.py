from __future__ import annotations

from typing import Literal, get_origin, get_args, get_type_hints, Optional, List, Tuple
from dataclasses import fields, is_dataclass

import numpy as np
import torch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QSpinBox,
    QFormLayout,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QHBoxLayout,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QSlider,
)


class SaveGtDialog(QDialog):
    def __init__(self, parent=None, guiding_methods=None, current_method=None):
        super().__init__(parent)
        self.setWindowTitle("Save Ground Truth Options")

        layout = QFormLayout(self)

        # --- Render Components ---
        self.render_unguided_cb = QCheckBox("Render Baseline (Unguided)")
        self.render_unguided_cb.setChecked(True)
        layout.addRow(self.render_unguided_cb)

        self.render_guided_cb = QCheckBox("Render Guided")
        self.render_guided_cb.setChecked(True)
        layout.addRow(self.render_guided_cb)

        # --- Guiding Method ---
        self.guiding_method_combo = QComboBox()
        if guiding_methods:
            self.guiding_method_combo.addItems(list(guiding_methods))
        if current_method and guiding_methods and current_method in guiding_methods:
            self.guiding_method_combo.setCurrentIndex(list(guiding_methods).index(current_method))
        layout.addRow("Guiding Method:", self.guiding_method_combo)

        # --- Comparison Settings ---
        self.online_checkbox = QCheckBox("Equal Time Online Comparison")
        self.online_checkbox.toggled.connect(self.toggle_mode)
        layout.addRow(self.online_checkbox)

        self.spp_label = QLabel("Samples per Pixel (SPP):")
        self.spp_spin = QSpinBox()
        self.spp_spin.setRange(1, 65536)
        self.spp_spin.setValue(256)
        layout.addRow(self.spp_label, self.spp_spin)

        self.time_label = QLabel("Time Budget (seconds):")
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 3600)
        self.time_spin.setValue(60)
        layout.addRow(self.time_label, self.time_spin)

        self.training_budget_label = QLabel("Training Budget (%):")
        self.training_budget_spin = QSpinBox()
        self.training_budget_spin.setRange(0, 100)
        self.training_budget_spin.setValue(20)
        self.training_budget_spin.setSuffix("%")
        layout.addRow(self.training_budget_label, self.training_budget_spin)

        self.bsdf_fraction_label = QLabel("BSDF Sampling Fraction (render):")
        self.bsdf_fraction_spin = QDoubleSpinBox()
        self.bsdf_fraction_spin.setRange(0.0, 1.0)
        self.bsdf_fraction_spin.setValue(0.5)
        self.bsdf_fraction_spin.setSingleStep(0.1)
        self.bsdf_fraction_spin.setDecimals(2)
        layout.addRow(self.bsdf_fraction_label, self.bsdf_fraction_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 8192)
        self.width_spin.setValue(1024)
        layout.addRow("Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 8192)
        self.height_spin.setValue(1024)
        layout.addRow("Height:", self.height_spin)

        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit("./renders")
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_button)
        layout.addRow("Output Directory:", output_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.toggle_mode(self.online_checkbox.isChecked())

    def browse_output_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_path_edit.text())
        if directory:
            self.output_path_edit.setText(directory)

    def toggle_mode(self, is_online: bool):
        self.time_label.setVisible(is_online)
        self.time_spin.setVisible(is_online)
        self.spp_label.setVisible(not is_online)
        self.spp_spin.setVisible(not is_online)

        self.training_budget_label.setVisible(True)
        self.training_budget_spin.setVisible(True)
        self.bsdf_fraction_label.setVisible(True)
        self.bsdf_fraction_spin.setVisible(True)

    def get_values(self):
        return {
            "render_unguided": self.render_unguided_cb.isChecked(),
            "render_guided": self.render_guided_cb.isChecked(),
            "guidingmethod": self.guiding_method_combo.currentText(),
            "online": self.online_checkbox.isChecked(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "spp": self.spp_spin.value(),
            "time": self.time_spin.value(),
            "trainingbudgetpercent": self.training_budget_spin.value(),
            "bsdfsamplingfraction": float(self.bsdf_fraction_spin.value()),
            "outputpath": self.output_path_edit.text(),
        }

class TrainSettingsDialog(QDialog):
    def __init__(self, parent=None, current_lr: float = 5e-3):
        super().__init__(parent)
        self.setWindowTitle("Training Settings")

        layout = QFormLayout(self)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100000)
        self.iterations_spin.setValue(100)
        layout.addRow("Training Iterations:", self.iterations_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-8, 1.0)
        self.lr_spin.setDecimals(8)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(float(current_lr))
        layout.addRow("Learning Rate:", self.lr_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_values(self):
        return {"iterations": self.iterations_spin.value(), "learningrate": float(self.lr_spin.value())}


class GuidingSettingsDialog(QDialog):
    def __init__(self, parent=None, guiding_system=None, main_resolution=(256, 256)):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.guiding_system = guiding_system
        self.config_widgets = {}

        layout = QVBoxLayout(self)

        render_group = QGroupBox("Render Settings")
        render_layout = QFormLayout()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 8192)
        self.width_spin.setValue(int(main_resolution[0]))
        render_layout.addRow("Render Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 8192)
        self.height_spin.setValue(int(main_resolution[1]))
        render_layout.addRow("Render Height:", self.height_spin)

        render_group.setLayout(render_layout)
        layout.addWidget(render_group)

        guiding_group = QGroupBox("Guiding Distribution Settings")
        guiding_layout = QFormLayout()

        dist = None
        if guiding_system is not None:
            dist = getattr(guiding_system, "distribution", None)
            if dist is None:
                dist = getattr(guiding_system, "_distribution", None)

        if dist is None:
            guiding_layout.addRow(QLabel("No guiding system available"))
        else:
            cfg = getattr(dist, "config", None)
            if cfg is None:
                guiding_layout.addRow(QLabel("No configuration available"))
            elif is_dataclass(cfg):
                try:
                    type_hints = get_type_hints(type(cfg))
                except Exception:
                    type_hints = {}

                for f in fields(cfg):
                    if f.name == "device":
                        continue
                    value = getattr(cfg, f.name)
                    resolved_type = type_hints.get(f.name, f.type)
                    widget = self.create_widget_for_field(f.name, resolved_type, value)
                    if widget is not None:
                        guiding_layout.addRow(f"{f.name.replace('_', ' ').title()}:", widget)
                        self.config_widgets[f.name] = widget
            else:
                guiding_layout.addRow(QLabel("Config is not a dataclass"))

        guiding_group.setLayout(guiding_layout)
        layout.addWidget(guiding_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def create_widget_for_field(self, name: str, field_type, value):
        origin = get_origin(field_type)
        if origin is Literal:
            options = get_args(field_type)
            w = QComboBox()
            w.addItems([str(o) for o in options])
            idx = w.findText(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w

        if field_type is int or field_type == "int":
            w = QSpinBox()
            w.setRange(-10**9, 10**9)
            w.setValue(int(value))
            return w

        if field_type is float or field_type == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(8)
            w.setSingleStep(1e-4)
            w.setValue(float(value))
            return w

        if field_type is bool or field_type == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            return w

        return None

    def get_values(self):
        values = {}
        for name, widget in self.config_widgets.items():
            if isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = float(widget.value())
            elif isinstance(widget, QCheckBox):
                values[name] = bool(widget.isChecked())
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values

    def get_resolution(self):
        return (self.width_spin.value(), self.height_spin.value())


class TrainingDataDialog(QDialog):
    """
    Shows training data from integrator.surfaceInteractionRecord (debug.py-style).
    Uses the exact debug.py reshape logic:
      buf = torch.tensor(field.numpy.T).reshape(H, W, spp(=1), depth, C)
    Caches only one field on CPU so the depth slider is fast and VRAM-friendly.
    """

    def __init__(self, parent=None, integrator=None, resolution=(256, 256), current_spp: Optional[int] = None):
        super().__init__(parent)

        title = "Training Data"
        self.setWindowTitle(title)

        self._integrator = integrator
        self._rec = getattr(integrator, "surfaceInteractionRecord", None) if integrator is not None else None

        self._width = int(resolution[0])
        self._height = int(resolution[1])
        self._depth = int(getattr(integrator, "maxdepth", 5)) if integrator is not None else 1

        # Import visualization helper used by viewer (keep same fallback behavior)
        try:
            from utils.visualization_helpers import numpy_to_qpixmap as _np2pix
        except Exception:
            from visualization_helpers import numpytoqpixmap as _np2pix
        self._numpy_to_qpixmap = _np2pix

        # Dropdown: (display_name, record_attr_name)
        self._fields: List[Tuple[str, str]] = [
            ("Emitted radiance", "emittedRadiance"),
            ("Throughput radiance (L)", "throughputRadiance"),
            ("BSDF weight", "bsdf"),
            ("Outgoing radiance (scalar)", "radiance"),
            ("Product / target (spectrum)", "product"),
            ("Normal", "normal"),
            ("Wi (local)", "wi"),
            ("Wo (local)", "wo"),
            ("Wo (world)", "woworld"),
            ("Combined PDF (woPdf)", "woPdf"),
            ("BSDF PDF", "bsdfPdf"),
            ("Guide PDF", "guidePdf"),
            ("BSDF fraction", "bsdfFraction"),
            ("Sample source (0=BSDF,1=Guide)", "sampleSource"),
            ("Is delta", "isDelta"),
            ("Guiding active", "guidingActive"),
            ("Active mask", "active"),
            ("NEE radiance", "radiancenee"),
        ]

        # Cache (CPU): only current field reshaped like debug.py
        self._cache_display_name: Optional[str] = None
        self._cache_tensor_hw1dc: Optional[torch.Tensor] = None  # (H,W,1,D,C) float32 CPU

        # UI
        layout = QVBoxLayout(self)

        controls = QFormLayout()
        self.field_combo = QComboBox()
        controls.addRow("Field", self.field_combo)

        depth_row = QHBoxLayout()
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setTracking(True)
        self.depth_label = QLabel("0")
        depth_row.addWidget(self.depth_slider, stretch=1)
        depth_row.addWidget(self.depth_label)
        controls.addRow("Depth", depth_row)

        layout.addLayout(controls)

        self.image_label = QLabel("No data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #111; color: #bbb; border: 1px solid #333; }")
        layout.addWidget(self.image_label, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._populate_fields()
        self._setup_depth_slider()

        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        self.depth_slider.valueChanged.connect(self._update_view)

        self._update_view()

    def _populate_fields(self):
        self.field_combo.clear()
        if self._rec is None:
            return

        # only show fields that actually exist
        for display, attr in self._fields:
            if hasattr(self._rec, attr):
                self.field_combo.addItem(display)

    def _setup_depth_slider(self):
        depth_max = max(1, int(self._depth))
        self.depth_slider.setRange(0, max(0, depth_max - 1))
        self.depth_slider.setValue(0)
        self.depth_label.setText("0")

    def _on_field_changed(self, _):
        # invalidate cache so we reload once for the new field
        self._cache_display_name = None
        self._cache_tensor_hw1dc = None
        self._update_view()

    def _display_to_attr(self, display_name: str) -> Optional[str]:
        for d, a in self._fields:
            if d == display_name:
                return a
        return None

    def _get_numpy_for_attr(self, attr: str) -> Optional[np.ndarray]:
        """
        Return numpy array as (N,C) or (N,1) suitable for debug.py-style reshape.
        Transpose ONLY if needed (debug.py used .T, but layout can vary by build/field).
        """
        if self._rec is None:
            return None

        f = getattr(self._rec, attr, None)
        if f is None:
            return None

        np_attr = getattr(f, "numpy", None)
        try:
            arr = np_attr() if callable(np_attr) else np_attr
        except Exception:
            return None
        if arr is None:
            return None

        arr = np.asarray(arr)

        height = int(self._height)
        width = int(self._width)
        depth = max(1, int(self._depth))
        n_expected = height * width * depth

        if arr.ndim == 1:
            if arr.size < n_expected:
                return None
            return arr.reshape(-1, 1)  # (N,1)

        if arr.ndim != 2:
            return None

        a0, a1 = int(arr.shape[0]), int(arr.shape[1])

        # prefer (N,C) where N matches expected and C small
        if a0 >= n_expected and a1 <= 16:
            return arr  # (N,C)

        # if it's (C,N), transpose to (N,C)
        if a1 >= n_expected and a0 <= 16:
            return arr.T

        # fallback: if one dimension exactly matches n_expected
        if a0 == n_expected:
            return arr
        if a1 == n_expected:
            return arr.T

        return None

    def _ensure_cache_loaded(self, display_name: str) -> bool:
        if self._cache_display_name == display_name and self._cache_tensor_hw1dc is not None:
            return True

        attr = self._display_to_attr(display_name)
        if not attr:
            return False

        arr_nc = self._get_numpy_for_attr(attr)
        if arr_nc is None:
            self._cache_display_name = None
            self._cache_tensor_hw1dc = None
            return False

        height = int(self._height)
        width = int(self._width)
        depth = max(1, int(self._depth))
        n_expected = height * width * depth

        n_samples, n_channels = int(arr_nc.shape[0]), int(arr_nc.shape[1])
        if n_samples < n_expected:
            self._cache_display_name = None
            self._cache_tensor_hw1dc = None
            return False

        arr_nc = arr_nc[:n_expected, :]
        arr_nc = np.asarray(arr_nc, dtype=np.float32)
        t = torch.from_numpy(arr_nc).reshape(height, width, 1, depth, n_channels)

        self._cache_display_name = display_name
        self._cache_tensor_hw1dc = t
        return True

    def _normalize_to_rgb(self, img_hwc: np.ndarray, display_name: str, attr: str) -> Optional[np.ndarray]:
        if img_hwc is None:
            return None

        height, width, n_channels = img_hwc.shape
        x = img_hwc

        # direction-like: [-1,1] -> [0,1]
        if attr in ("normal", "wi", "wo", "woworld"):
            x = np.clip(x * 0.5 + 0.5, 0.0, 1.0)

        # PDFs: log + percentile normalize
        if attr in ("woPdf", "bsdfPdf", "guidePdf") or attr.lower().endswith("pdf"):
            x = np.log1p(np.maximum(x, 0.0))
            lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                x = (x - lo) / (hi - lo)
            x = np.clip(x, 0.0, 1.0)

        # Convert channels -> RGB
        if n_channels == 3:
            rgb = x.astype(np.float32)
            # gamma-correct radiance-ish buffers for display (like debug plotting)
            if ("radiance" in attr.lower()) or (attr in ("bsdf", "product", "throughputRadiance", "emittedRadiance", "radiancenee")):
                rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)
            return np.clip(rgb, 0.0, 1.0)

        if n_channels == 2:
            rgb = np.zeros((height, width, 3), dtype=np.float32)
            rgb[:, :, 0:2] = x[:, :, 0:2].astype(np.float32)
            return np.clip(rgb, 0.0, 1.0)

        gray = x[:, :, 0].astype(np.float32) if n_channels >= 1 else np.zeros((height, width), dtype=np.float32)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return np.clip(rgb, 0.0, 1.0)

    def _update_view(self):
        if self._rec is None or self.field_combo.count() == 0:
            self.image_label.setText("No surfaceInteractionRecord yet. Render at least one frame.")
            return

        display = self.field_combo.currentText()
        attr = self._display_to_attr(display)
        if not attr:
            self.image_label.setText("Unknown field.")
            return

        depth_idx = int(self.depth_slider.value())
        depth_idx = int(np.clip(depth_idx, 0, max(0, int(self._depth) - 1)))
        self.depth_label.setText(str(depth_idx))

        if not self._ensure_cache_loaded(display):
            self.image_label.setText(
                f"Cannot reshape field to (H,W,1,D,C).\n"
                f"Expected N={self._height * self._width * max(1, int(self._depth))} entries.\n"
                f"Try rendering once after opening the scene."
            )
            return

        t = self._cache_tensor_hw1dc  # (H,W,1,D,C)
        img = t[:, :, 0, depth_idx, :].detach().cpu().numpy()  # (H,W,C)

        rgb = self._normalize_to_rgb(img, display_name=display, attr=attr)
        if rgb is None:
            self.image_label.setText("No image.")
            return

        self.image_label.setPixmap(self._numpy_to_qpixmap(rgb.astype(np.float32)))