import torch


@torch.jit.script
def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for batched tensors [w, x, y, z]."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert euler angles (xyz/roll-pitch-yaw) to quaternion (wxyz).

    Args:
        euler: (*, 3) roll, pitch, yaw in radians

    Returns:
        (*, 4) quaternion in wxyz format
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to euler angles (xyz/roll-pitch-yaw).

    Args:
        quat: (*, 4) quaternion in wxyz format

    Returns:
        (*, 3) roll, pitch, yaw in radians

    At the gimbal-lock poles (pitch = ±90°) roll and yaw are not independently
    recoverable: only their sum/difference is. The naive formulas degenerate to
    ``atan2(0, 0) == 0`` for *both* angles there, which silently discards the
    rotation about the vertical axis (e.g. euler (0, 90, -90) would come back as
    (0, 90, 0) — a genuinely different orientation, not an equivalent gauge
    choice). At the poles we therefore set roll = 0 and fold the whole rotation
    into yaw, which round-trips through ``euler_to_quat`` exactly.

    The fold must fire only where the naive formulas actually break down. Just
    off the pole roll and yaw *are* separately recoverable, so folding there
    destroys real information: at ``1e-6`` this branch covered a band ~0.08°
    wide in pitch and produced euler angles that no longer round-trip (6 of 4000
    random rotations). ``1e-9`` still catches the exact pole and round-trips
    everywhere, verified from 1e-8 off-pole outward.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid numerical issues at poles
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Gimbal lock: with roll folded into yaw, atan2(x, w) == -sign(pitch) * yaw / 2,
    # so yaw = -sign(pitch) * 2 * atan2(x, w).
    gimbal = sinp.abs() > 1.0 - 1e-9
    if bool(gimbal.any()):
        lock_yaw = -torch.copysign(torch.ones_like(sinp), sinp) * 2.0 * torch.atan2(x, w)
        roll = torch.where(gimbal, torch.zeros_like(roll), roll)
        yaw = torch.where(gimbal, lock_yaw, yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to axis-angle (rotation vector).

    The quaternion is canonicalised to w >= 0 first, so the result is always the
    shorter of the two equivalent rotations -- the form the RoboWits EE_ABS
    actions expect.

    Args:
        quat: (*, 4) quaternion in wxyz format

    Returns:
        (*, 3) axis-angle, magnitude = rotation angle in radians
    """
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    w, xyz = quat[..., 0:1], quat[..., 1:4]
    norm = torch.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-8)
    angle = 2.0 * torch.atan2(norm, w)
    return (xyz / norm) * angle


def axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rotation vector) to quaternion (wxyz).

    Args:
        axis_angle: (*, 3) axis-angle, magnitude = rotation angle in radians

    Returns:
        (*, 4) quaternion in wxyz format
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle
    half = angle * 0.5
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to the 6D rotation representation.

    Uses the rot6d layout of the RoboWits datasets: the first two *rows* of the rotation
    matrix, flattened row-major -> [R00, R01, R02, R10, R11, R12]. Verified against
    converted_lerobot_0410/18 (same recordings, axis-angle) — note this is the transpose of
    the column layout used by gs-core's ``RotationMatrix.to_rot6d``.

    Args:
        quat: (*, 4) quaternion in wxyz format

    Returns:
        (*, 6) 6D rotation
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ],
        dim=-1,
    )


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert the 6D rotation representation to a rotation matrix via Gram-Schmidt.

    The 6D values are the first two rows of the matrix (see ``quat_to_rot6d``).

    Args:
        rot6d: (*, 6) 6D rotation [a1(3), a2(3)] (non-orthonormal rows are allowed)

    Returns:
        (*, 3, 3) rotation matrix
    """
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]

    b1 = a1 / torch.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 / torch.norm(a2, dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)  # rows


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert the 6D rotation representation to a quaternion (wxyz).

    Args:
        rot6d: (*, 6) 6D rotation [a1(3), a2(3)] (non-orthonormal rows are allowed)

    Returns:
        (*, 4) quaternion in wxyz format, with non-negative scalar part
    """
    rot = rot6d_to_rotation_matrix(rot6d)

    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    # Magnitudes of the four quaternion components (Shepperd's method)
    q_abs = torch.sqrt(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22],
            dim=-1,
        ).clamp(min=0.0)
    )

    # Four candidate quaternions, each numerically stable when its own component is largest
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs.unsqueeze(-1).clamp(min=1e-4))

    best = q_abs.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(*q_abs.shape[:-1], 1, 4)
    quat = candidates.gather(-2, best).squeeze(-2)

    return torch.where(quat[..., :1] < 0, -quat, quat)
