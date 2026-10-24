"""Images for the viewer's inspection panels."""
import numpy as np
import torch
import matplotlib.cm as cm
import plotly.graph_objects as go

import drjit as dr
import mitsuba as mi

from PyQt6.QtGui import QImage, QPixmap


def _expand(x: torch.Tensor, n: int, dims: int) -> torch.Tensor:
    return x.reshape(1, dims).expand(n, -1).cuda()


def _sphere_directions(si, resolution):
    """Local-frame directions (H*W, 3) for an equal-area image of the whole sphere at si."""
    width, height = resolution
    ii, jj = dr.meshgrid(dr.linspace(mi.Float, 0, width - 1, width), dr.linspace(mi.Float, 0, height - 1, height))
    film_size = mi.Vector2f(width, height)
    d = mi.warp.square_to_uniform_sphere((mi.Vector2f(ii, jj) + 0.5) / film_size - 0.5 / film_size)
    return si.to_local(dr.normalize(d)).torch().T.contiguous().cuda()


def plot_guiding_distribution(guide, si, position, wi, normal, roughness, resolution=(256, 256)):
    """log1p(pdf) of the guide over all directions, magma-coloured, (H, W, 3)."""
    dirs = _sphere_directions(si, resolution)
    n = dirs.shape[0]
    pdf = guide.pdf(_expand(position, n, 3), _expand(wi, n, 3), _expand(normal, n, 3), _expand(roughness, n, 1), dirs)
    grid = torch.log1p(pdf.reshape(resolution[1], resolution[0])).cpu().numpy()
    if grid.max() > 1e-6:
        grid /= grid.max()
    return cm.magma(grid)[..., :3]


def plot_nrc_radiance(nrc, si, position, normal, resolution=(256, 256), exposure=0.0):
    """NRC radiance over all directions, Reinhard tone mapped with gamma 2.2, (H, W, 3)."""
    dirs = _sphere_directions(si, resolution)
    n = dirs.shape[0]
    rgb = nrc.query(_expand(position, n, 3), _expand(normal, n, 3), dirs, torch.ones((n, 1), device="cuda"))
    rgb = rgb.reshape(resolution[1], resolution[0], 3).cpu().numpy() * 2.0 ** exposure
    return np.clip(rgb / (1.0 + rgb), 0.0, 1.0) ** (1 / 2.2)


def visualize_pdf_on_sphere_html(guide, position, wi, normal, roughness, res=100):
    """Interactive Plotly HTML of the guide's pdf on the unit sphere."""
    theta, phi = torch.meshgrid(torch.linspace(0, 2 * np.pi, res), torch.linspace(0, np.pi, res), indexing="ij")
    x, y, z = torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)
    xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cuda()
    n = xyz.shape[0]
    with torch.no_grad():
        pdf = guide.pdf(_expand(position, n, 3), _expand(wi, n, 3), _expand(normal, n, 3),
                        _expand(roughness, n, 1), xyz).reshape(res, res).cpu().numpy()

    fig = go.Figure(go.Surface(x=x.numpy(), y=y.numpy(), z=z.numpy(), surfacecolor=pdf, colorscale="plasma",
                               cmin=float(pdf.min()), cmax=float(pdf.max()), showscale=False))
    for axis, color in zip(np.eye(3) * 1.1, ["red", "lime", "blue"]):
        fig.add_trace(go.Scatter3d(x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]], mode="lines",
                                   line=dict(color=color, width=5), showlegend=False))
        fig.add_trace(go.Cone(x=[axis[0]], y=[axis[1]], z=[axis[2]], u=[axis[0]], v=[axis[1]], w=[axis[2]],
                              colorscale=[[0, color], [1, color]], sizemode="absolute", sizeref=0.07, showscale=False))
    hidden = dict(visible=False)
    fig.update_layout(
        title="Guiding PDF on Unit Sphere",
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, bgcolor="black"),
        scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=1.25)),
        paper_bgcolor="black", font=dict(color="white"), margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig.to_html(include_plotlyjs="cdn")


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    """(H, W) or (H, W, 3) image in [0, 1] (or uint8) -> QPixmap."""
    if image is None:
        return QPixmap()
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    gray = image.ndim == 2
    fmt = QImage.Format.Format_Grayscale8 if gray else QImage.Format.Format_RGB888
    return QPixmap.fromImage(QImage(image.data, w, h, w * (1 if gray else 3), fmt).copy())


def draw_marker_on_image(image_np, x, y, size=3, color=(0.0, 1.0, 0.0)):
    marked = image_np.copy()
    h, w = marked.shape[:2]
    marked[max(0, y - size):min(h, y + size + 1), max(0, x - size):min(w, x + size + 1)] = color
    return marked
