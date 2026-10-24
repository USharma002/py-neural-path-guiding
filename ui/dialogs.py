from __future__ import annotations

from dataclasses import fields
from typing import Literal, get_args, get_origin, get_type_hints

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QVBoxLayout,
)

import guiding.registry as registry
from integrators.path_guiding_integrator import _to_torch
from utils.visualization_helpers import numpy_to_qpixmap

DIRECTION_FIELDS = {"normal", "wi", "wo", "wo_world", "next_wi", "next_normal"}


def _spin(value, lo, hi, decimals=None, step=None):
    w = QSpinBox() if decimals is None else QDoubleSpinBox()
    w.setRange(lo, hi)
    if decimals is not None:
        w.setDecimals(decimals)
    if step is not None:
        w.setSingleStep(step)
    w.setValue(value)
    return w


def _ok_cancel(dialog: QDialog) -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


class SaveGtDialog(QDialog):
    def __init__(self, parent=None, guiding_methods=(), current_method=None):
        super().__init__(parent)
        self.setWindowTitle("Save Ground Truth Options")
        layout = QFormLayout(self)

        self.render_unguided_cb = QCheckBox("Render Baseline (Unguided)")
        self.render_guided_cb = QCheckBox("Render Guided")
        self.render_unguided_cb.setChecked(True)
        self.render_guided_cb.setChecked(True)
        layout.addRow(self.render_unguided_cb)
        layout.addRow(self.render_guided_cb)

        self.guiding_method_combo = QComboBox()
        self.guiding_method_combo.addItems(list(guiding_methods))
        if current_method in guiding_methods:
            self.guiding_method_combo.setCurrentText(current_method)
        layout.addRow("Guiding Method:", self.guiding_method_combo)

        self.online_checkbox = QCheckBox("Equal Time Online Comparison")
        layout.addRow(self.online_checkbox)

        self.spp_label, self.spp_spin = QLabel("Samples per Pixel (SPP):"), _spin(256, 1, 65536)
        self.time_label, self.time_spin = QLabel("Time Budget (seconds):"), _spin(60, 1, 3600)
        self.training_budget_spin = _spin(20, 0, 100)
        self.training_budget_spin.setSuffix("%")
        self.bsdf_fraction_spin = _spin(0.5, 0.0, 1.0, decimals=2, step=0.1)
        self.width_spin = _spin(1024, 1, 8192)
        self.height_spin = _spin(1024, 1, 8192)
        layout.addRow(self.spp_label, self.spp_spin)
        layout.addRow(self.time_label, self.time_spin)
        layout.addRow("Training Budget (%):", self.training_budget_spin)
        layout.addRow("BSDF Sampling Fraction (render):", self.bsdf_fraction_spin)
        layout.addRow("Width:", self.width_spin)
        layout.addRow("Height:", self.height_spin)

        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit("./renders")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(browse)
        layout.addRow("Output Directory:", output_layout)
        layout.addRow(_ok_cancel(self))

        # Defaults follow the selected method's paper settings
        self.guiding_method_combo.currentTextChanged.connect(self._apply_method_defaults)
        self._apply_method_defaults(self.guiding_method_combo.currentText())
        self.online_checkbox.toggled.connect(self.toggle_mode)
        self.toggle_mode(False)

    def _apply_method_defaults(self, method_name: str):
        if method_name:
            cls = registry.get_integrator_class(method_name)
            self.bsdf_fraction_spin.setValue(cls.BSDF_SAMPLING_FRACTION)
            self.training_budget_spin.setValue(cls.TRAINING_BUDGET_PERCENT)

    def browse_output_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_path_edit.text())
        if directory:
            self.output_path_edit.setText(directory)

    def toggle_mode(self, is_online: bool):
        for w in (self.time_label, self.time_spin):
            w.setVisible(is_online)
        for w in (self.spp_label, self.spp_spin):
            w.setVisible(not is_online)

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
        self.iterations_spin = _spin(100, 1, 100000)
        self.lr_spin = _spin(float(current_lr), 1e-8, 1.0, decimals=8, step=1e-4)
        layout.addRow("Training Iterations:", self.iterations_spin)
        layout.addRow("Learning Rate:", self.lr_spin)
        layout.addRow(_ok_cancel(self))

    def get_values(self):
        return {"iterations": self.iterations_spin.value(), "learningrate": float(self.lr_spin.value())}


class GuidingSettingsDialog(QDialog):
    """Render resolution and an editor for the guide's config dataclass (int, float, bool, Literal fields)."""

    def __init__(self, parent=None, guide=None, main_resolution=(256, 256)):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)

        render_group = QGroupBox("Render Settings")
        render_layout = QFormLayout(render_group)
        self.width_spin = _spin(int(main_resolution[0]), 64, 8192)
        self.height_spin = _spin(int(main_resolution[1]), 64, 8192)
        render_layout.addRow("Render Width:", self.width_spin)
        render_layout.addRow("Render Height:", self.height_spin)
        layout.addWidget(render_group)

        guiding_group = QGroupBox("Guiding Distribution Settings")
        guiding_layout = QFormLayout(guiding_group)
        self.config_widgets = {}
        cfg = guide.config
        hints = get_type_hints(type(cfg))
        for f in fields(cfg):
            if f.name == "device":
                continue
            widget = self._widget(hints.get(f.name, f.type), getattr(cfg, f.name))
            if widget is not None:
                guiding_layout.addRow(f"{f.name.replace('_', ' ').title()}:", widget)
                self.config_widgets[f.name] = widget
        layout.addWidget(guiding_group)
        layout.addWidget(_ok_cancel(self))

    @staticmethod
    def _widget(field_type, value):
        if get_origin(field_type) is Literal:
            w = QComboBox()
            w.addItems([str(o) for o in get_args(field_type)])
            w.setCurrentText(str(value))
            return w
        if field_type is bool:
            w = QCheckBox()
            w.setChecked(value)
            return w
        if field_type is int:
            return _spin(value, -10**9, 10**9)
        if field_type is float:
            return _spin(value, -1e9, 1e9, decimals=8, step=1e-4)
        return None

    def get_values(self):
        values = {}
        for name, w in self.config_widgets.items():
            if isinstance(w, QComboBox):
                values[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                values[name] = w.isChecked()
            else:
                values[name] = w.value()
        return values

    def get_resolution(self):
        return (self.width_spin.value(), self.height_spin.value())


class TrainingDataDialog(QDialog):
    """Browse the integrator's per-vertex record, one field and path depth at a time."""

    def __init__(self, parent=None, integrator=None, resolution=(256, 256)):
        super().__init__(parent)
        self.setWindowTitle("Training Data")
        self._rec = integrator.surfaceInteractionRecord
        self._shape = (resolution[1], resolution[0], integrator.max_depth)
        self._cache = (None, None)

        self.field_combo = QComboBox()
        self.field_combo.addItems(list(integrator.record_type.DRJIT_STRUCT))
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setRange(0, integrator.max_depth - 1)
        self.depth_label = QLabel("0")
        self.image_label = QLabel("No data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #111; color: #bbb; border: 1px solid #333; }")

        controls = QFormLayout()
        controls.addRow("Field", self.field_combo)
        depth_row = QHBoxLayout()
        depth_row.addWidget(self.depth_slider, stretch=1)
        depth_row.addWidget(self.depth_label)
        controls.addRow("Depth", depth_row)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(close)

        self.field_combo.currentTextChanged.connect(self._update_view)
        self.depth_slider.valueChanged.connect(self._update_view)
        self._update_view()

    def _field(self, name: str) -> np.ndarray:
        """Field as (H, W, depth, C), cached for the depth slider."""
        if self._cache[0] != name:
            x = _to_torch(getattr(self._rec, name)).float().cpu().numpy()
            self._cache = (name, x.reshape(*self._shape, -1))
        return self._cache[1]

    def _update_view(self):
        name, depth = self.field_combo.currentText(), self.depth_slider.value()
        self.depth_label.setText(str(depth))
        try:
            x = self._field(name)[:, :, depth, :]
        except ValueError:
            self.image_label.setText("The record does not have one entry per pixel and depth; render a frame first.")
            return

        if name in DIRECTION_FIELDS:
            x = x * 0.5 + 0.5
        elif "pdf" in name.lower():
            x = np.log1p(np.maximum(x, 0.0))
            lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
            x = (x - lo) / (hi - lo) if hi > lo else x
        elif x.shape[-1] == 3:
            x = np.clip(x, 0.0, 1.0) ** (1 / 2.2)
        rgb = np.repeat(x, 3, axis=2) if x.shape[-1] == 1 else x
        self.image_label.setPixmap(numpy_to_qpixmap(np.clip(rgb, 0.0, 1.0)))
