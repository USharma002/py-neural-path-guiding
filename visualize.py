import sys

import mitsuba as mi
import drjit as dr
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QFileDialog, QCheckBox 
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from PyQt6.QtGui import QPixmap, QImage, QMouseEvent, QColor
from PyQt6.QtCore import Qt, QPointF, QPoint
import matplotlib.cm as cm

from path_guiding_integrator import PathGuidingIntegrator
from path_guiding_system import PathGuidingSystem
from vmf_mixture import *

import random 
import io, time
from PIL import Image

import matplotlib.pyplot as plt
from spherical import GTSphericalCamera

from renderer import Renderer

custom_render = Renderer().render

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')
# mi.set_log_level(mi.LogLevel.Warn)

# dr.set_flag(dr.JitFlag.LoopRecord, False)
# dr.set_flag(dr.JitFlag.VCallRecord, False)

import plotly.graph_objects as go

def add_rgb_axes_to_fig(fig, length=1.1, cone_size=0.07):
    """Adds colored RGB arrows to a Plotly 3D figure to represent the axes."""
    axis_names = ['X', 'Y', 'Z']
    colors = ['red', 'lime', 'blue']
    directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # For cone direction

    for i in range(3):
        # Calculate the end point of the axis arrow
        x_end, y_end, z_end = [d * length for d in directions[i]]
        
        # Axis Line: Provide x, y, and z as separate lists of [start, end]
        fig.add_trace(go.Scatter3d(x=[0, x_end], y=[0, y_end], z=[0, z_end],
                                     mode='lines',
                                     line=dict(color=colors[i], width=5),
                                     name=axis_names[i]))
        # Arrow Head (Cone)
        fig.add_trace(go.Cone(x=[x_end], y=[y_end], z=[z_end],
                               u=[directions[i][0]], v=[directions[i][1]], w=[directions[i][2]],
                               colorscale=[[0, colors[i]], [1, colors[i]]],
                               sizemode="absolute", sizeref=cone_size,
                               showscale=False))
    return fig


def visualize_vmf_on_sphere_html(model, res=100):
    xyz, xgrid, ygrid, zgrid = generate_sphere_mesh(res)

    with torch.no_grad():
        pdf_vals = model.pdf(xyz)
        pdf_vals = pdf_vals.reshape(res, res)

    x = xgrid.cpu().numpy()
    y = ygrid.cpu().numpy()
    z = zgrid.cpu().numpy()
    pdf = pdf_vals.detach().cpu().numpy()

    
    fig = go.Figure(data=go.Surface(
        x=x, y=y, z=z,
        surfacecolor=pdf,
        colorscale='plasma',
        cmin=pdf.min(), cmax=pdf.max(),
        showscale=False,
        opacity=1.0
    ))
    fig = add_rgb_axes_to_fig(fig)

    fig.update_layout(
        title='vMF Mixture on Unit Sphere',
        scene=dict(
            xaxis=dict(visible=False,),
            yaxis=dict(visible=False,),
            zaxis=dict(visible=False,),
            bgcolor='black'
        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig.to_html(include_plotlyjs='cdn')


def generate_sphere_mesh(res=100):
    theta = torch.linspace(0, 2 * np.pi, res)    
    phi = torch.linspace(0, np.pi, res) 
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    xyz = torch.stack([x, y, z], dim=-1) 
    return xyz.reshape(-1, 3), x, y, z


device = "cuda"

def spherical_to_cartesian_torch(theta, phi):
    """Converts torch tensors of spherical coordinates to Cartesian."""
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def plot_non_batched_mixture(vmf, si, resolution=(256, 256), device='cuda'):
    width, height = 256, 256
    i = dr.linspace(mi.Float, 0, width - 1, width)
    j = dr.linspace(mi.Float, 0, height - 1, height)

    ii, jj = dr.meshgrid(i, j)
    film_size = mi.Vector2f(width, height)
    position_sample = (mi.Vector2f(ii, jj) + 0.5) / film_size

    film_size = mi.Vector2f([256, 256])
    half_pixel_offset = 0.5 / film_size
    corrected_position_sample = position_sample - half_pixel_offset

    d = mi.warp.square_to_uniform_sphere(corrected_position_sample)
    d = dr.normalize(d)

    dirs_local = si.to_local(d)

    pdf_values = vmf.pdf(dirs_local)
    
    value_grid = torch.log1p(pdf_values.reshape(height, width).detach().cpu()).numpy()
    
    max_val = np.max(value_grid)
    if max_val > 1e-6:
        value_grid /= max_val
    
    colored_rgb = cm.magma(value_grid)[..., :3]
    return colored_rgb

def numpy_to_qpixmap(numpy_array):
    if numpy_array is None:
        return QPixmap()
    if numpy_array.ndim == 2:
        h, w = numpy_array.shape
        ch = 1
    else:
        h, w, ch = numpy_array.shape
    if numpy_array.dtype != np.uint8:
        numpy_array = (np.clip(numpy_array, 0, 1) * 255).astype(np.uint8)
    if ch == 1:
        format = QImage.Format.Format_Grayscale8
    elif ch == 3:
        format = QImage.Format.Format_RGB888
    elif ch == 4:
        format = QImage.Format.Format_RGBA8888
    else:
        return QPixmap()
    image_data = np.ascontiguousarray(numpy_array)
    bytes_per_line = ch * w
    q_image = QImage(image_data.data, w, h, bytes_per_line, format)
    return QPixmap.fromImage(q_image.copy())


def draw_marker_on_image(image_np, x, y, size=3, color=[0.0, 1.0, 0.0]):
    marked_image = image_np.copy()
    h, w, _ = marked_image.shape
    x_start, x_end = max(0, x - size), min(w, x + size + 1)
    y_start, y_end = max(0, y - size), min(h, y + size + 1)
    marked_image[y_start:y_end, x_start:x_end] = color
    return marked_image


class MitsubaViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mitsuba Click-Based Viewer")

        self.main_resolution = (256, 256)
        self.last_click_coords = None

        # Central widget & layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # --- BROWSE BUTTON ---
        self.browse_button = QPushButton("Browse Scene")
        self.browse_button.clicked.connect(self.load_scene_from_file)
        self.main_layout.addWidget(self.browse_button)

        # --- TRAIN STEP BUTTON ---
        self.train_button = QPushButton("Train Step")
        self.train_button.clicked.connect(self.train_step_clicked)
        self.main_layout.addWidget(self.train_button)


        # --- CHECKBOXES ---
        self.checkbox_layout = QHBoxLayout()
        self.cosine_checkbox = QCheckBox("Cosine Product")
        self.bsdf_checkbox = QCheckBox("BSDF Product")
        self.guiding_checkbox = QCheckBox("Path Guiding")
        self.bsdf_checkbox.setChecked(True)

        self.checkbox_layout.addWidget(self.cosine_checkbox)
        self.checkbox_layout.addWidget(self.bsdf_checkbox)
        self.checkbox_layout.addWidget(self.guiding_checkbox)

        self.main_layout.addLayout(self.checkbox_layout)

        self.cosine_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.bsdf_checkbox.stateChanged.connect(self.reprocess_last_click)
        self.guiding_checkbox.stateChanged.connect(self.re_render_main_image)

        # --- HORIZONTAL RENDER LAYOUT ---
        self.layout = QHBoxLayout()
        self.main_layout.addLayout(self.layout)

        # Setup views (same as before)
        self.main_view = QLabel("Main Render")
        self.main_view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self.hover_view = QLabel("Click to inspect point\n(Viridis PDF)")
        self.hover_view.setMinimumSize(256, 256)
        self.hover_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.normalized_pdf_view = QLabel("Normalized PDF")
        self.normalized_pdf_view.setMinimumSize(256, 256)
        self.normalized_pdf_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vmf_view = QLabel("vMF Mixture")
        self.vmf_view.setMinimumSize(256, 256)
        self.vmf_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(self.main_view, stretch=1)
        self.layout.addWidget(self.hover_view, stretch=0)
        self.layout.addWidget(self.normalized_pdf_view, stretch=0)
        self.layout.addWidget(self.vmf_view, stretch=0)

        self.vmf_sphere_view = QWebEngineView()
        self.vmf_sphere_view.setMinimumSize(256, 256)
        self.vmf_sphere_view.page().setBackgroundColor(QColor("black"))
        self.layout.addWidget(self.vmf_sphere_view, stretch=1)

        self.main_view.mousePressEvent = self.on_mouse_click
        self.is_rendering = False

        self.integrator = mi.load_dict({
            "type": "path_guiding_integrator",
            "max_depth": 5
        })

        # Load initial scene
        self.load_default_scene()

    def re_render_main_image(self):
        if self.is_rendering:
            return

        self.is_rendering = True
        try:
            self.integrator.set_guiding( self.guiding_checkbox.isChecked() )
            print(f"Re-rendering main image with path guiding: {self.guiding_checkbox.isChecked()}")
            image_np = custom_render(self.scene, spp=64, integrator=self.integrator, seed=0).numpy()

            self.original_main_image_np = image_np 
            if self.last_click_coords:
                x, y = self.last_click_coords
                image_np = draw_marker_on_image(image_np, x, y)

            pixmap = numpy_to_qpixmap(image_np ** (1/2.2))
            self.main_view.setPixmap(pixmap)

        finally:
            self.is_rendering = False


    def load_default_scene(self):
        width, height = self.main_resolution
        cbox = mi.cornell_box()
        cbox['integrator']['max_depth'] = 5
        cbox['integrator']['rr_depth'] = 5
        cbox['sensor']['film']['width'] = width
        cbox['sensor']['film']['height'] = height

        self.scene_dict = cbox
        self.transform_light(self.scene_dict)
        self.scene = mi.load_dict(self.scene_dict)

        self.integrator = mi.load_dict({
            'type': 'path_guiding_integrator',
            'max_depth': cbox['integrator']['max_depth']
        })

        num_rays = cbox['sensor']['film']['width'] * cbox['sensor']['film']['width']
        print(f"Setting up integrator for a wavefront of {num_rays} rays.")

        self.integrator.setup(
            num_rays=num_rays,
            bbox_min=self.scene.bbox().min,
            bbox_max=self.scene.bbox().max
        )

        self.integrator.set_guiding( False )
        self.main_sensor = self.scene.sensors()[0]

        sensor_params = mi.traverse(self.main_sensor)
        sensor_params["film.size"] = self.main_resolution
        sensor_params.update()

        self.original_main_image_np = custom_render(self.scene, spp=16, integrator=self.integrator, seed=0).numpy()

        self.hover_sensor = mi.load_dict({
            'type': 'gt_spherical_camera',
            'film': {'type': 'hdrfilm', 'width': 256, 'height': 256}
        })

        
        self.guiding_system = self.integrator.guiding_system

        initial_pixmap = numpy_to_qpixmap(self.original_main_image_np ** (1/2.2))
        self.main_view.setPixmap(initial_pixmap)

    def load_scene_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mitsuba Scene", "", "XML Files (*.xml)")
        if not file_path:
            return
        try:
            print(f"Loading scene from: {file_path}")
            self.scene = mi.load_file(file_path)
            self.main_sensor = self.scene.sensors()[0]

            sensor_params = mi.traverse(self.main_sensor)
            sensor_params["film.size"] = self.main_resolution
            sensor_params.update()

            self.integrator = mi.load_dict({
                'type': 'path_guiding_integrator'
            })
            num_rays = self.main_resolution[0] * self.main_resolution[1]
            self.integrator.setup(
                num_rays=num_rays,
                bbox_min=self.scene.bbox().min,
                bbox_max=self.scene.bbox().max
            )

            self.original_main_image_np = custom_render(self.scene, spp=16, integrator=self.integrator).numpy()
            pixmap = numpy_to_qpixmap(self.original_main_image_np ** (1/2.2))
            self.main_view.setPixmap(pixmap)

        except Exception as e:
            print(f"Failed to load scene: {e}")


    def transform_light(self, scene_dict):
        original_transform = np.array(scene_dict['light']['to_world'].matrix)
        original_position = original_transform[:3, 3]
        flip_x = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        flipped_transform = flip_x @ original_transform
        flipped_transform[:3, 3] = original_position - np.array([0, 0.2, 0])
        scene_dict['light']['to_world'] = mi.ScalarTransform4f(flipped_transform)
        return scene_dict

    def on_mouse_click(self, event):
        if self.is_rendering:
            return

        pos = event.pos()
        x, y = pos.x(), pos.y()
        self.last_click_coords = (x, y)
        film_size = self.main_sensor.film().size()
        if not (0 <= x < film_size[0] and 0 <= y < film_size[1]): return

        u, v = (x + 0.5) / film_size[0], (y + 0.5) / film_size[1]
        ray, _ = self.main_sensor.sample_ray(0.0, 0.5, [u, v], [0.5, 0.5])
        si = self.scene.ray_intersect(ray)

        position = si.p.torch()
        normal = si.n.torch()
        wo = -ray.d.torch()
        roughness = torch.tensor([[1]], device=device)

        self.guiding_system = self.integrator.guiding_system
        vmf_params = self.guiding_system.gnn(position, wo, roughness)
        guiding_dist = MixedSphericalGaussianDistribution(vmf_params, batch_idx=0)

        vmf_img_np = plot_non_batched_mixture(guiding_dist, si, device=device)
        vmf_img_np = np.transpose(vmf_img_np, (1, 0, 2)) 

        self.vmf_view.setPixmap(numpy_to_qpixmap(vmf_img_np))

        # vMF Sphere visualization
        # vmf_sphere_np = visualize_vmf_on_sphere_image(guiding_dist, res=100)
        # self.vmf_sphere_view.setPixmap(numpy_to_qpixmap(vmf_sphere_np)) 
        html = visualize_vmf_on_sphere_html(guiding_dist, res=100)
        self.vmf_sphere_view.setHtml(html)
        
        image_with_marker_np = draw_marker_on_image(self.original_main_image_np, x, y)
        self.main_view.setPixmap(numpy_to_qpixmap(image_with_marker_np ** (1/2.2)))

        if not dr.any(si.is_valid()):
            self.hover_view.setText("Miss (No surface hit)")
            self.normalized_pdf_view.setText("")
            return

        try:
            self.is_rendering = True
            
            self.hover_view.setText("Rendering...")
            self.normalized_pdf_view.setText("Rendering...")
            QApplication.processEvents()

            # --- RENDER LOGIC ---
            dummy_ray = si.spawn_ray(si.n)
            safe_origin = dummy_ray.o
            self.hover_sensor.world_transform = mi.Transform4f.translate(safe_origin)
            self.hover_sensor.initial_si = si
            self.hover_sensor.product_bsdf = self.bsdf_checkbox.isChecked()
            self.hover_sensor.product_cosine = self.cosine_checkbox.isChecked()

            spp = 128
            gt_image_untransposed_np = mi.render(self.scene, sensor=self.hover_sensor, spp=spp).numpy()
            hover_image_np = np.transpose(gt_image_untransposed_np, (1, 0, 2))  

            # --- VISUALIZATION FOR VIRIDIS VIEW (Block 2) ---
            luminance = np.mean(hover_image_np, axis=2)
            log_luminance = np.log1p(luminance)

            visualization_scale = 10.0
            scaled_luminance = log_luminance * visualization_scale

            # normalized_luminance = np.clip(scaled_luminance, 0, 1)
            colored_rgba = cm.viridis(scaled_luminance)
            colored_rgb = colored_rgba[..., :3]

            pdf_pixmap = numpy_to_qpixmap(colored_rgb)
            self.hover_view.setPixmap(pdf_pixmap)

            image_pixmap = numpy_to_qpixmap(hover_image_np ** (1/2.2))
            self.normalized_pdf_view.setPixmap(image_pixmap)

        finally:
            self.is_rendering = False

    def reprocess_last_click(self):
        if self.last_click_coords is None:
            return 

        x, y = self.last_click_coords

        local_pos = QPointF(x, y)
        global_pos = QPointF(self.main_view.mapToGlobal(QPoint(x, y)))

        fake_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            local_pos,
            global_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )

        self.on_mouse_click(fake_event)

    def train_step_clicked(self):
        self.integrator.set_guiding( False )
        print("Manual train step triggered.")

        for i in range(100):
            custom_render(self.scene, spp = 1, integrator = self.integrator, progress=False, seed=1)
            loss = self.guiding_system.train_step(self.integrator)
            if (i + 1) % 10 == 0:
                print(f"Loss : {loss}")

        self.integrator.set_guiding( self.guiding_checkbox.isChecked() )
        self.reprocess_last_click()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MitsubaViewer()
    viewer.show()
    sys.exit(app.exec())