import numpy as np
import torch
import matplotlib.cm as cm
import plotly.graph_objects as go

import drjit as dr
import mitsuba as mi

from PyQt6.QtGui import QImage, QPixmap

from guiding.config import (
    NUMBA_AVAILABLE, 
    fast_tone_map_and_gamma, 
    fast_float_to_uint8
)

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
    """Visualize PDF on unit sphere as 3D interactive plot.
    
    Supports both:
    - Old interface: model.pdf(directions)
    - New GuidingDistribution interface: uses stored _vis_position, _vis_wi, _vis_roughness
    """
    xyz, xgrid, ygrid, zgrid = generate_sphere_mesh(res)
    xyz = xyz.to('cuda')

    with torch.no_grad():
        # Check if model has the new interface with stored visualization context
        if hasattr(model, '_vis_position') and model._vis_position is not None:
            # New GuidingDistribution interface
            n_dirs = xyz.shape[0]
            pos = model._vis_position.expand(n_dirs, -1)
            wi = model._vis_wi.expand(n_dirs, -1)
            roughness = model._vis_roughness
            if roughness.dim() == 2:
                roughness = roughness.expand(n_dirs, -1)
            else:
                roughness = roughness.expand(n_dirs)
            pdf_vals = model.pdf(pos, wi, xyz, roughness)
        else:
            # Legacy interface: pdf(directions)
            pdf_vals = model.pdf(xyz)
        
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
        title='Guiding PDF on Unit Sphere',
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

def plot_guiding_distribution(guiding_dist, si, position, wi, roughness, resolution=(256, 256), device='cuda'):
    """Plot any GuidingDistribution on sphere using its pdf() method.
    
    This is the generic visualization function that works with any
    GuidingDistribution subclass (VMF, NIS, etc.).
    
    Args:
        guiding_dist: GuidingDistribution instance with pdf(pos, wi, wo, roughness)
        si: Surface interaction for local coordinate transform
        position: Surface position (1, 3)
        wi: Incoming direction (1, 3)  
        roughness: Surface roughness (1, 1)
        resolution: Output image resolution
        device: Compute device
        
    Returns:
        RGB image array (H, W, 3)
    """
    width, height = resolution
    
    # Generate sphere directions
    i = dr.linspace(mi.Float, 0, width - 1, width)
    j = dr.linspace(mi.Float, 0, height - 1, height)
    ii, jj = dr.meshgrid(i, j)
    
    film_size = mi.Vector2f(width, height)
    position_sample = (mi.Vector2f(ii, jj) + 0.5) / film_size
    half_pixel_offset = 0.5 / film_size
    corrected_position_sample = position_sample - half_pixel_offset
    
    d = mi.warp.square_to_uniform_sphere(corrected_position_sample)
    d = dr.normalize(d)
    
    # Transform to local coordinates
    dirs_local = si.to_local(d)
    dirs_torch = dirs_local.torch()  # (H*W, 3)
    
    n_dirs = dirs_torch.shape[0]
    
    # Expand conditioning to match directions
    pos_expanded = position.expand(n_dirs, -1)
    wi_expanded = wi.expand(n_dirs, -1)
    roughness_expanded = roughness.expand(n_dirs, -1) if roughness.dim() > 1 else roughness.expand(n_dirs)
    
    # Query PDF for all directions
    pdf_values = guiding_dist.pdf(pos_expanded, wi_expanded, dirs_torch, roughness_expanded)
    
    # Convert to visualization
    if hasattr(pdf_values, 'detach'):
        value_grid = torch.log1p(pdf_values.reshape(height, width).detach().cpu()).numpy()
    else:
        value_grid = np.log1p(np.array(pdf_values).reshape(height, width))
    
    max_val = np.max(value_grid)
    if max_val > 1e-6:
        value_grid /= max_val
    
    colored_rgb = cm.magma(value_grid)[..., :3]
    return colored_rgb

def plot_nrc_radiance(nrc_model, si, position, normal, bbox_min=None, bbox_max=None, resolution=(256, 256), device='cuda'):
    """Render actual scene radiance using NRC predictions.
    
    Creates a spherical view from the clicked point, querying the NRC
    for predicted radiance in each direction - like a spherical camera.
    
    Args:
        nrc_model: The Neural Radiance Cache model
        si: Surface interaction for coordinate transformation
        position: Surface position (1, 3) tensor
        normal: Surface normal (1, 3) tensor  
        bbox_min: Scene bounding box min for position normalization
        bbox_max: Scene bounding box max for position normalization
        resolution: Output image resolution
        device: Device to use for computation
        
    Returns:
        RGB numpy array showing predicted scene radiance from that point
    """
    width, height = resolution
    
    # Generate directions on sphere (spherical sensor)
    i = dr.linspace(mi.Float, 0, width - 1, width)
    j = dr.linspace(mi.Float, 0, height - 1, height)

    ii, jj = dr.meshgrid(i, j)
    film_size = mi.Vector2f(width, height)
    position_sample = (mi.Vector2f(ii, jj) + 0.5) / film_size

    half_pixel_offset = 0.5 / film_size
    corrected_position_sample = position_sample - half_pixel_offset

    d = mi.warp.square_to_uniform_sphere(corrected_position_sample)
    d = dr.normalize(d)
    
    # Convert to world space directions (these are the view directions - towards the "camera" in each direction)
    dirs_world = si.to_world(d)
    view_dirs = dirs_world.torch().to(device)  # (H*W, 3)
    
    num_dirs = view_dirs.shape[0]
    
    # Expand position and normal to match directions
    position_expanded = position.expand(num_dirs, -1).to(device)
    normal_expanded = normal.expand(num_dirs, -1).to(device)
    roughness = torch.ones((num_dirs, 1), device=device)
    
    # Query NRC for all directions - get actual RGB radiance
    # Pass bbox for proper position normalization
    with torch.no_grad():
        nrc_radiance = nrc_model.query(
            position_expanded, 
            normal_expanded, 
            view_dirs, 
            roughness,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )  # (H*W, 3) RGB radiance
    
    # Reshape to image (H, W, 3)
    rgb_image = nrc_radiance.reshape(height, width, 3).detach().cpu().numpy().astype(np.float64)
    
    # Apply tone mapping + gamma correction (numba-accelerated if available)
    if NUMBA_AVAILABLE:
        rgb_image = fast_tone_map_and_gamma(rgb_image, 2.2)
    else:
        # Fallback to numpy
        rgb_image = rgb_image / (1.0 + rgb_image)  # Reinhard
        rgb_image = np.power(np.clip(rgb_image, 0, 1), 1.0 / 2.2)  # Gamma
    
    return rgb_image

def numpy_to_qpixmap(numpy_array):
    if numpy_array is None:
        return QPixmap()
    if numpy_array.ndim == 2:
        h, w = numpy_array.shape
        ch = 1
    else:
        h, w, ch = numpy_array.shape
    if numpy_array.dtype != np.uint8:
        # Use numba-accelerated conversion if available and 3-channel
        if NUMBA_AVAILABLE and ch == 3 and numpy_array.ndim == 3:
            numpy_array = fast_float_to_uint8(np.ascontiguousarray(numpy_array, dtype=np.float64))
        else:
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