import torch
import math
import torch.nn.functional as F
import mitsuba as mi
import drjit as dr

M_EPSILON = 1e-5
M_2PI = 2 * math.pi
M_INV_4PI = 1.0 / (4 * math.pi)

def uniform_sample_sphere(u):
    """Uniform sampling on unit sphere using 2D uniform sample u ∈ [0,1]^2"""
    z = 1.0 - 2.0 * u[..., 0]
    r = safe_sqrt(1.0 - z * z)
    phi = M_2PI * u[..., 1]
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)

def spherical_to_cartesian(theta, phi):
    theta = torch.as_tensor(theta)
    phi = torch.as_tensor(phi)

    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)

def cartesian_to_spherical(dir):
    dir = torch.nn.functional.normalize(dir, dim=-1)  
    z = torch.clamp(dir[..., 2], -1.0, 1.0)

    theta = torch.acos(z)
    phi = torch.atan2(dir[..., 1], dir[..., 0])
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    return torch.stack([theta, phi], dim=-1)

def cartesian_to_spherical_normalized(dir):
    sph = cartesian_to_spherical(dir)
    sph[..., 0] /= math.pi
    sph[..., 1] /= 2*math.pi
    return sph

def canonical_to_dir(p):
    p = torch.tensor(p)
    cos_theta = 2 * p[..., 1] - 1
    sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, min=0.0))
    phi = 2 * math.pi * p[..., 0]

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    return torch.stack([x, y, z], dim=-1)

def dir_to_canonical(dir):
    cos_theta = torch.clamp(dir[..., 2], -1.0, 1.0)
    v = (cos_theta + 1.0) * 0.5

    phi = torch.atan2(dir[..., 1], dir[..., 0])
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    u = phi / (2 * math.pi)

    return torch.stack([u, v], dim=-1)

def safe_sqrt(x):
    """Computes torch.sqrt(x) with a clamp to avoid NaNs for negative inputs."""
    return torch.sqrt(torch.clamp(x, min=0.0))

def get_perpendicular(u: torch.Tensor) -> torch.Tensor:
    a = u.abs()  # (..., 3)
    
    # Compare components to find smallest axis to cross with
    uyx = (a[..., 0] < a[..., 1]).int()
    uzx = (a[..., 0] < a[..., 2]).int()
    uzy = (a[..., 1] < a[..., 2]).int()

    xm = uyx & uzx
    ym = (1 ^ xm) & uzy
    zm = 1 ^ (xm | ym)

    # Create basis vector to cross with
    basis = torch.stack([xm, ym, zm], dim=-1).float()

    # Compute perpendicular vector via cross product
    perp = torch.linalg.cross(u, basis, dim=-1)
    perp = F.normalize(perp, dim=-1)
    return perp

def to_world(mu, local_dir):
    # Ensure mu is unit-length
    mu = F.normalize(mu, dim=-1)

    # Create tangent frame (u, v, w) where w = mu
    w = mu
    up = torch.tensor([0.0, 1.0, 0.0], device=mu.device)
    u = F.normalize(torch.linalg.cross(up.expand_as(w), w), dim=-1)
    v = torch.linalg.cross(w, u)

    # Combine local_dir = (x, y, z) with frame
    return (
        local_dir[..., 0:1] * u +
        local_dir[..., 1:2] * v +
        local_dir[..., 2:3] * w
    )

def to_local(world_dir, mu):
    mu = torch.nn.functional.normalize(mu, dim=-1)
    T = get_perpendicular(mu)
    B = torch.nn.functional.normalize(torch.linalg.cross(mu, T, dim=-1), dim=-1)
    return torch.stack([
        torch.dot(world_dir, T),
        torch.dot(world_dir, B),
        torch.dot(world_dir, mu)
    ])


# === Test cases ===

def canonicalToDir(p: mi.Vector2f) -> mi.Vector3f:
    """
        Input: Vector2f: x = Phi, y = CosTheta
            Phi: In normalized range [0, 1]
            CosTheta: In normalized range [0, 1]
        Output: Normalized Vector3f
    """
    # Unnormalized
    cosTheta = 2 * p.y - 1
    # Compute sin(theta)
    sinTheta = dr.sqrt( 1 - cosTheta * cosTheta )
    
    # Unnormalize
    phi = dr.two_pi * p.x

    sinPhi, cosPhi = dr.sincos(phi)

    dir = mi.Vector3f()

    #   Phi: xy, CosTheta: z
    dir.x = sinTheta * cosPhi
    dir.y = sinTheta * sinPhi
    dir.z = cosTheta

    #   Phi: xz, CosTheta: y
    # dir.x = sinTheta * cosPhi
    # dir.z = sinTheta * sinPhi
    # dir.y = cosTheta

    return dir


def dirToCanonical(d: mi.Vector3f) -> mi.Vector2f:
    """
        Input: Normalized Vector3f
        Output: Vector2f: x = Phi, y = CosTheta
            Phi: In normalized range [0, 1]
            CosTheta: In normalized range [0, 1]
    """

    #   Phi: xy, CosTheta: z
    cosTheta = dr.clip( d.z, -1, 1 )
    phi = dr.atan2(d.y, d.x)

    #   Phi: xz, CosTheta: y
    # cosTheta = dr.clip( d.y, -1, 1 )
    # phi = dr.atan2(d.z, d.x)

    loop = mi.Loop("rotate phi", lambda: (phi))
    while loop( phi < 0 ):
        phi += 2.0 * dr.pi

    p = mi.Vector2f(0)
    p.x = phi / dr.two_pi
    p.y = ( cosTheta + 1 ) / 2

    # Have to check first if the item isfinite, if not then return [0,0]
    flag = dr.isfinite(d.x) & dr.isfinite(d.y) & dr.isfinite(d.z)
    return dr.select( flag, p, mi.Vector2f(0) )
