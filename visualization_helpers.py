import numpy as np
import torch
import matplotlib.cm as cm
import plotly.graph_objects as go

import drjit as dr
import mitsuba as mi

from PyQt6.QtGui import QImage, QPixmap

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


def visualize_pdf_on_sphere_html(model, res=100):
    xyz, xgrid, ygrid, zgrid = generate_sphere_mesh(res)

    with torch.no_grad():
        pdf_vals = model.pdf(xyz) # get the pdf values
        pdf_vals = pdf_vals.reshape(res, res)

    # create grid
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
    # only one mixture model visualization 

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
    # draw a green marker on the image at given location
    marked_image = image_np.copy()
    h, w, _ = marked_image.shape
    
    x_start, x_end = max(0, x - size), min(w, x + size + 1)
    y_start, y_end = max(0, y - size), min(h, y + size + 1)

    marked_image[y_start:y_end, x_start:x_end] = color
    return marked_image