"""Pure-torch quaternion / geometry utilities.

Provides the subset of ``isaaclab.utils.math`` that the Play2Perfect pipeline
uses, implemented without any dependency on Isaac Lab or Isaac Sim so that the
same helpers can be used inside the Genesis simulator backend.

All quaternions are in **wxyz** order (w first), matching both Genesis and
Isaac Lab conventions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic quaternion operations
# ---------------------------------------------------------------------------


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two unit quaternions (wxyz order).

    Args:
        q1: (..., 4) tensor [w, x, y, z]
        q2: (..., 4) tensor [w, x, y, z]

    Returns:
        (..., 4) product quaternion, not renormalised.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return conjugate (= inverse for unit quaternion) of q (wxyz)."""
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) *v* by unit quaternion(s) *q* (wxyz order).

    Supports broadcasting: q can be (4,) or (N, 4); v can be (3,) or (N, 3).

    Returns a tensor of the same shape as *v*.
    """
    # Rodrigues formula: v' = v + 2*w*(xyz × v) + 2*(xyz × (xyz × v))
    xyz = q[..., 1:]          # (..., 3)
    w = q[..., 0:1]            # (..., 1)
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Build unit quaternions from angle (radians, scalar/N) and axis (3 or N×3).

    Returns:
        (N, 4) or (4,) tensor in wxyz order.
    """
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def random_orientation(n: int, device: torch.device) -> torch.Tensor:
    """Sample *n* uniformly random unit quaternions (wxyz, N×4).

    Uses the method from Shoemake (1992): draw 4 standard normals, normalise.
    """
    q = torch.randn(n, 4, device=device)
    return F.normalize(q, dim=-1)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """L2-normalise quaternions to unit length along the last dimension."""
    return F.normalize(q, dim=-1)


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)


def keypoints_world(
    center_pos: torch.Tensor,   # (N, 3)
    center_rot: torch.Tensor,   # (N, 4) wxyz
    kp_offsets: torch.Tensor,   # (N, K, 3)
) -> torch.Tensor:
    """Rotate and translate object-frame keypoints to world frame."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)


# ---------------------------------------------------------------------------
# Misc geometry
# ---------------------------------------------------------------------------


def perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply a random rotation noise of at most *max_deg* degrees to *q_wxyz*."""
    n = q_wxyz.shape[0]
    axis = F.normalize(torch.randn(n, 3, device=q_wxyz.device), dim=-1)
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (
        math.pi / 180.0
    )
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)


__all__ = [
    "quat_mul",
    "quat_conjugate",
    "quat_apply",
    "quat_from_angle_axis",
    "random_orientation",
    "normalize_quat",
    "keypoints_world",
    "perturb_quat",
    "KEYPOINT_CORNERS",
]
