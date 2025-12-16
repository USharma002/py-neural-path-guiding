import numpy as np
import torch
import matplotlib.cm as cm
import plotly.graph_objects as go

import drjit as dr
import mitsuba as mi

from PyQt6.QtGui import QImage, QPixmap


# Robust import across your code variants
try:
    from guiding.config import (
        NUMBA_AVAILABLE,
        fast_tone_map_and_gamma,
        fast_float_to_uint8,
    )
except Exception:
    # Older naming used elsewhere in your project
    from guiding.config import (  # type: ignore
        NUMBAAVAILABLE as NUMBA_AVAILABLE,
        fasttonemapandgamma as fast_tone_map_and_gamma,
        fastfloattouint8 as fast_float_to_uint8,
    )


def add_rgb_axes_to_fig(fig, length=1.1, cone_size=0.07):
    """Adds colored RGB arrows to a Plotly 3D figure to represent the axes."""
    axis_names = ["X", "Y", "Z"]
    colors = ["red", "lime", "blue"]
    directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for i in range(3):
        x_end, y_end, z_end = [d * length for d in directions[i]]

        fig.add_trace(
            go.Scatter3d(
                x=[0, x_end],
                y=[0, y_end],
                z=[0, z_end],
                mode="lines",
                line=dict(color=colors[i], width=5),
                name=axis_names[i],
            )
        )
        fig.add_trace(
            go.Cone(
                x=[x_end],
                y=[y_end],
                z=[z_end],
                u=[directions[i][0]],
                v=[directions[i][1]],
                w=[directions[i][2]],
                colorscale=[[0, colors[i]], [1, colors[i]]],
                sizemode="absolute",
                sizeref=cone_size,
                showscale=False,
            )
        )
    return fig


def generate_sphere_mesh(res=100):
    theta = torch.linspace(0, 2 * np.pi, res)
    phi = torch.linspace(0, np.pi, res)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    xyz = torch.stack([x, y, z], dim=-1)
    return xyz.reshape(-1, 3), x, y, z


def visualize_pdf_on_sphere_html(model, res=100):
    """Visualize PDF on unit sphere as 3D interactive plot."""
    xyz, xgrid, ygrid, zgrid = generate_sphere_mesh(res)
    xyz = xyz.to("cuda")  # (N, 3)

    with torch.no_grad():
        # Handle new interface used by your guiding distributions
        if hasattr(model, "_vis_position") and model._vis_position is not None:
            n_dirs = xyz.shape[0]
            pos = model._vis_position.reshape(1, 3).expand(n_dirs, -1)
            wi = model._vis_wi.reshape(1, 3).expand(n_dirs, -1)
            roughness = model._vis_roughness.reshape(1, 1).expand(n_dirs, -1)
            pdf_vals = model.pdf(pos, wi, xyz, roughness)
        else:
            pdf_vals = model.pdf(xyz)

        pdf_vals = pdf_vals.reshape(res, res)

    x = xgrid.cpu().numpy()
    y = ygrid.cpu().numpy()
    z = zgrid.cpu().numpy()
    pdf = pdf_vals.detach().cpu().numpy()

    fig = go.Figure(
        data=go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=pdf,
            colorscale="plasma",
            cmin=float(pdf.min()),
            cmax=float(pdf.max()),
            showscale=False,
            opacity=1.0,
        )
    )
    fig = add_rgb_axes_to_fig(fig)

    fig.update_layout(
        title="Guiding PDF on Unit Sphere",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig.to_html(include_plotlyjs="cdn")


def _to_n3(x: torch.Tensor) -> torch.Tensor:
    """DrJit often gives (3,N); convert to (N,3)."""
    if x.ndim == 2 and x.shape[0] == 3 and x.shape[1] != 3:
        return x.t().contiguous()
    return x.contiguous()


def _apply_display_mapping(
    rgb: np.ndarray,
    *,
    mode: str = "reinhard",
    gamma: float = 2.2,
    exposure: float = 0.0,
) -> np.ndarray:
    """
    Convert linear/HDR RGB to displayable [0,1] sRGB-ish.
    mode:
      - "clamp": clamp to [0,1] then gamma (matches your Hemisphere RGB view style)
      - "reinhard": reinhard tonemap then gamma
      - "none": only apply exposure and clamp (no gamma)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = rgb * (2.0 ** float(exposure))

    if mode == "none":
        return np.clip(rgb, 0.0, 1.0)

    if mode == "clamp":
        rgb = np.clip(rgb, 0.0, 1.0)
        return np.power(rgb, 1.0 / gamma)

    # default: reinhard
    rgb = rgb / (1.0 + rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.power(rgb, 1.0 / gamma)


def plot_guiding_distribution(
    guiding_dist,
    si,
    position,
    wi,
    roughness,
    resolution=(256, 256),
    device="cuda",
):
    """Plot any GuidingDistribution on sphere using its pdf() method."""
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

    # Local coords (this matches how your guiding is trained/queried)
    dirs_local = si.to_local(d)
    dirs_torch = _to_n3(dirs_local.torch()).to(device)

    n_dirs = dirs_torch.shape[0]
    pos_expanded = position.reshape(1, 3).expand(n_dirs, -1).to(device)
    wi_expanded = wi.reshape(1, 3).expand(n_dirs, -1).to(device)
    roughness_expanded = roughness.reshape(1, 1).expand(n_dirs, -1).to(device)

    pdf_values = guiding_dist.pdf(pos_expanded, wi_expanded, dirs_torch, roughness_expanded)

    if hasattr(pdf_values, "detach"):
        value_grid = torch.log1p(pdf_values.reshape(height, width).detach().cpu()).numpy()
    else:
        value_grid = np.log1p(np.array(pdf_values).reshape(height, width))

    max_val = float(np.max(value_grid))
    if max_val > 1e-6:
        value_grid /= max_val

    colored_rgb = cm.magma(value_grid)[..., :3]
    return colored_rgb


def plot_nrc_radiance(
    nrc_model,
    si,
    position,
    normal,
    bbox_min=None,
    bbox_max=None,
    resolution=(256, 256),
    device="cuda",
    *,
    display_mode: str = "clamp",
    exposure: float = 0.0,
):
    """
    Render hemisphere radiance using NRC predictions.

    IMPORTANT: Uses LOCAL directions (si.to_local(d)), so it matches the coordinate
    frame used by training data (batch.wi is local in your integrator).
    """
    width, height = resolution

    # Generate directions on sphere
    i = dr.linspace(mi.Float, 0, width - 1, width)
    j = dr.linspace(mi.Float, 0, height - 1, height)
    ii, jj = dr.meshgrid(i, j)

    film_size = mi.Vector2f(width, height)
    position_sample = (mi.Vector2f(ii, jj) + 0.5) / film_size

    half_pixel_offset = 0.5 / film_size
    corrected_position_sample = position_sample - half_pixel_offset

    d = mi.warp.square_to_uniform_sphere(corrected_position_sample)
    d = dr.normalize(d)

    # Local directions: align with training (fixes rotation/misalignment)
    dirs_local = si.to_local(d)
    view_dirs = _to_n3(dirs_local.torch()).to(device)

    num_dirs = view_dirs.shape[0]
    position_expanded = position.reshape(1, 3).expand(num_dirs, -1).to(device)
    normal_expanded = normal.reshape(1, 3).expand(num_dirs, -1).to(device)
    roughness = torch.ones((num_dirs, 1), device=device, dtype=torch.float32)

    with torch.no_grad():
        nrc_radiance = nrc_model.query(
            position_expanded,
            normal_expanded,
            view_dirs,
            roughness,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

    rgb = nrc_radiance.reshape(height, width, 3).detach().cpu().numpy().astype(np.float64)

    # Display mapping:
    # - "clamp" makes it comparable to your Hemisphere RGB sRGB panel (clip->gamma).
    # - switch to "reinhard" + exposure if you want to inspect HDR structure better.
    if NUMBA_AVAILABLE and display_mode == "reinhard":
        rgb_disp = fast_tone_map_and_gamma(rgb * (2.0 ** float(exposure)), 2.2)
    else:
        rgb_disp = _apply_display_mapping(rgb, mode=display_mode, exposure=exposure, gamma=2.2)

    return rgb_disp


def numpy_to_qpixmap(numpy_array):
    if numpy_array is None:
        return QPixmap()

    if numpy_array.ndim == 2:
        h, w = numpy_array.shape
        ch = 1
    else:
        h, w, ch = numpy_array.shape

    if numpy_array.dtype != np.uint8:
        if NUMBA_AVAILABLE and ch == 3 and numpy_array.ndim == 3:
            numpy_array = fast_float_to_uint8(np.ascontiguousarray(numpy_array, dtype=np.float64))
        else:
            numpy_array = (np.clip(numpy_array, 0, 1) * 255).astype(np.uint8)

    if ch == 1:
        fmt = QImage.Format.Format_Grayscale8
    elif ch == 3:
        fmt = QImage.Format.Format_RGB888
    elif ch == 4:
        fmt = QImage.Format.Format_RGBA8888
    else:
        return QPixmap()

    image_data = np.ascontiguousarray(numpy_array)
    bytes_per_line = ch * w
    q_image = QImage(image_data.data, w, h, bytes_per_line, fmt)
    return QPixmap.fromImage(q_image.copy())


def draw_marker_on_image(image_np, x, y, size=3, color=(0.0, 1.0, 0.0)):
    marked = image_np.copy()
    h, w = marked.shape[:2]
    x0, x1 = max(0, x - size), min(w, x + size + 1)
    y0, y1 = max(0, y - size), min(h, y + size + 1)
    marked[y0:y1, x0:x1] = color
    return marked


def view_debug_image(integrator, depth_to_view=0):
    """Utility: visualize a radiance buffer slice after rendering."""
    radiance_buffer = integrator.surfaceInteractionRecord.radiance
    rad = radiance_buffer.torch().T.cpu().numpy().reshape(256, 256, 5, 3)

    img_np = rad[..., depth_to_view, :3]

    import matplotlib.pyplot as plt

    plt.imshow(np.clip(np.power(np.clip(img_np, 0, 1), 1 / 2.2), 0, 1))
    plt.title(f"Reconstructed Radiance at Depth {depth_to_view}")
    plt.show()
