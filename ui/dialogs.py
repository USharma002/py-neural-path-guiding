from typing import Literal, get_origin, get_args, get_type_hints
from dataclasses import fields, is_dataclass

from PyQt6.QtWidgets import (
    QVBoxLayout, QCheckBox, QDialog,
    QSpinBox, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QGroupBox, QLabel, QComboBox
)

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


class TrainSettingsDialog(QDialog):
    """Dialog for configuring training parameters."""
    def __init__(self, parent=None, current_lr: float = 5e-3):
        super().__init__(parent)
        self.setWindowTitle("Training Settings")
        
        layout = QFormLayout(self)
        
        # Number of iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(100)
        layout.addRow("Training Iterations:", self.iterations_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(current_lr)
        layout.addRow("Learning Rate:", self.lr_spin)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def getValues(self):
        return {
            "iterations": self.iterations_spin.value(),
            "learning_rate": self.lr_spin.value()
        }


class GuidingSettingsDialog(QDialog):
    """Dialog for configuring the current guiding distribution and render settings."""
    def __init__(self, parent=None, guiding_system=None, main_resolution=(256, 256)):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.guiding_system = guiding_system
        self.config_widgets = {}
        
        layout = QVBoxLayout(self)
        
        # --- Render Settings Group ---
        render_group = QGroupBox("Render Settings")
        render_layout = QFormLayout()
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(main_resolution[0])
        render_layout.addRow("Render Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(main_resolution[1])
        render_layout.addRow("Render Height:", self.height_spin)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        
        # --- Guiding Distribution Settings Group ---
        guiding_group = QGroupBox("Guiding Distribution Settings")
        guiding_layout = QFormLayout()
        
        if guiding_system is None or not hasattr(guiding_system, '_distribution'):
            guiding_layout.addRow(QLabel("No guiding system available"))
        else:
            dist = guiding_system._distribution
            config = getattr(dist, 'config', None)
            
            if config is None:
                guiding_layout.addRow(QLabel("No configuration available"))
            else:
                if is_dataclass(config):
                    # Use get_type_hints to resolve string annotations (from __future__ import annotations)
                    try:
                        type_hints = get_type_hints(type(config))
                    except Exception:
                        type_hints = {}
                    
                    for field in fields(config):
                        if field.name == 'device':
                            continue  # Skip device field
                        
                        value = getattr(config, field.name)
                        # Use resolved type hint if available, otherwise fall back to field.type
                        resolved_type = type_hints.get(field.name, field.type)
                        widget = self._create_widget_for_field(field.name, resolved_type, value)
                        if widget:
                            guiding_layout.addRow(f"{field.name.replace('_', ' ').title()}:", widget)
                            self.config_widgets[field.name] = widget
                else:
                    guiding_layout.addRow(QLabel("Config is not a dataclass"))
        
        guiding_group.setLayout(guiding_layout)
        layout.addWidget(guiding_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_widget_for_field(self, name: str, field_type, value):
        """Create appropriate widget based on field type."""
        
        # Check if it's a Literal type using get_origin
        origin = get_origin(field_type)
        if origin is Literal:
            options = get_args(field_type)  # Extract ('nearest', 'linear') from Literal["nearest", "linear"]
            widget = QComboBox()
            widget.addItems([str(opt) for opt in options])
            # Set current value if it exists in options
            str_value = str(value)
            index = widget.findText(str_value)
            if index >= 0:
                widget.setCurrentIndex(index)
            return widget
        
        # Handle basic types (may be actual types or string annotations)
        if field_type is int or field_type == 'int':
            widget = QSpinBox()
            widget.setRange(1, 10000)
            widget.setValue(value)
            return widget
        elif field_type is float or field_type == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(1e-8, 1000.0)
            widget.setDecimals(6)
            widget.setSingleStep(1e-4)
            widget.setValue(value)
            return widget
        elif field_type is bool or field_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(value)
            return widget
        
        return None
    
    def getValues(self):
        """Get updated config values."""
        values = {}
        for name, widget in self.config_widgets.items():
            if isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values
    
    def getResolution(self):
        """Get the render resolution."""
        return (self.width_spin.value(), self.height_spin.value())
