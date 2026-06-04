"""Shared utilities for RoboWits environments.

Polygon operations used in success criteria, plus entity inspection and
geometry helpers moved from RoboWitsEnv.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Task registry helpers
# ---------------------------------------------------------------------------


def resolve_task(task_id: str) -> str:
    """Convert a short task ID to its full registry name.

    Accepts mutation IDs ("06_01" -> "robowits/06_01-v0"), base IDs
    ("06" -> unique TASK_REGISTRY match), or already-full names (pass-through).
    """
    import gs_gym

    if "/" in task_id:
        return task_id
    if "_" in task_id:
        return f"robowits/{task_id}-v0"
    tid = task_id.zfill(2)
    matches = [t for t in gs_gym.TASK_REGISTRY if f"/{tid}-" in t]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous task ID '{task_id}'. Matches: {matches}")
    raise ValueError(f"Unknown task ID '{task_id}'.")


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------


def polygon_signed_area(poly: NDArray) -> float:
    """Compute signed area of polygon using shoelace formula.

    Positive for counter-clockwise, negative for clockwise.
    """
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: NDArray) -> float:
    """Compute absolute area of polygon using shoelace formula."""
    return abs(polygon_signed_area(poly))


def ensure_ccw(poly: NDArray) -> NDArray:
    """Ensure polygon vertices are in counter-clockwise order."""
    if poly is None or len(poly) < 3:
        return poly
    if polygon_signed_area(poly) < 0:
        return np.flipud(poly.copy())
    return poly


def sutherland_hodgman_clip(subject: NDArray, clip: NDArray) -> NDArray:
    """Clip subject polygon by convex clip polygon using Sutherland-Hodgman algorithm.

    Both polygons should be in counter-clockwise order.

    Returns:
        (K, 2) array of clipped polygon vertices, or empty array if no intersection.
    """
    if subject is None or clip is None:
        return np.zeros((0, 2), dtype=float)
    if len(subject) < 3 or len(clip) < 3:
        return np.zeros((0, 2), dtype=float)

    output_list = subject.tolist()
    clip_pts = clip.tolist()

    def is_inside(p, a, b):
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

    def compute_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return [px, py]

    for i in range(len(clip_pts)):
        if not output_list:
            break
        input_list = output_list
        output_list = []
        A = clip_pts[i]
        B = clip_pts[(i + 1) % len(clip_pts)]
        S = input_list[-1]
        for E in input_list:
            if is_inside(E, A, B):
                if not is_inside(S, A, B):
                    output_list.append(compute_intersection(S, E, A, B))
                output_list.append(E)
            elif is_inside(S, A, B):
                output_list.append(compute_intersection(S, E, A, B))
            S = E

    if not output_list or len(output_list) < 3:
        return np.zeros((0, 2), dtype=float)
    return np.array(output_list, dtype=float)


def intersection_area(poly_a: NDArray, poly_b: NDArray) -> float:
    """Compute intersection area of two 2D polygons using Sutherland-Hodgman clipping."""
    if poly_a is None or poly_b is None:
        return 0.0
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    poly_a = ensure_ccw(np.asarray(poly_a, dtype=float))
    poly_b = ensure_ccw(np.asarray(poly_b, dtype=float))
    clipped = sutherland_hodgman_clip(poly_a, poly_b)
    if len(clipped) < 3:
        return 0.0
    return float(polygon_area(np.array(clipped)))


# ---------------------------------------------------------------------------
# Grasp / held detection
# ---------------------------------------------------------------------------


def check_being_held(
    obj_info: dict,
    robot,
    tol: float = 0.03,
) -> bool:
    """Check if an object is being held by either gripper.

    An object is considered "held" if either gripper's end-effector position
    is within the object's axis-aligned bounding box, expanded by a tolerance
    to account for the gripper tip not being perfectly centered inside the object.

    Args:
        obj_info: Object info dict from collect_objs_info(), must contain 'bounds' key
            with shape (2, 3) as [[xmin, ymin, zmin], [xmax, ymax, zmax]].
        robot: BimanualMarvinRobot instance with right_ee_pose and left_ee_pose attributes.
        tol: Tolerance in meters to expand the bounding box on each side.

    Returns:
        True if the object is being held by either gripper, False otherwise.
    """
    bounds = obj_info.get("bounds")
    if bounds is None:
        return False

    bounds = np.array(bounds)
    if bounds.shape != (2, 3):
        return False

    xmin, ymin, zmin = bounds[0] - tol
    xmax, ymax, zmax = bounds[1] + tol

    right_ee_pos = robot.right_ee_pose[0, :3].cpu().numpy()
    left_ee_pos = robot.left_ee_pose[0, :3].cpu().numpy()

    def point_in_bounds(pos):
        return (xmin <= pos[0] <= xmax) and (ymin <= pos[1] <= ymax) and (zmin <= pos[2] <= zmax)

    right_holding = point_in_bounds(right_ee_pos)
    left_holding = point_in_bounds(left_ee_pos)

    return right_holding or left_holding


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------


def is_particle_entity(entity) -> bool:
    """Return True if entity is a particle-based (MPM) entity."""
    try:
        from genesis.engine.entities.particle_entity import ParticleEntity

        if isinstance(entity, ParticleEntity):
            return True
    except ImportError:
        pass

    try:
        class_name = type(entity).__name__.lower()
        module_name = getattr(type(entity), "__module__", "").lower()
        if "particle" in class_name or "mpm" in class_name:
            return True
        if "particle" in module_name or "mpm" in module_name:
            return True
        if hasattr(entity, "material"):
            mat = entity.material
            mat_name = type(mat).__name__.lower()
            if "mpm" in mat_name or "particle" in mat_name:
                return True
            if "mpm" in getattr(mat.__class__, "__module__", "").lower():
                return True
        if hasattr(entity, "n_particles") and entity.n_particles > 0:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------


def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to axis-angle."""
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    w, xyz = quat[..., 0:1], quat[..., 1:4]
    norm = torch.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-8)
    angle = 2.0 * torch.atan2(norm, w)
    return (xyz / norm) * angle


def axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to quaternion (wxyz)."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle
    half = angle * 0.5
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw] in radians.

    Args:
        quat: (4,) quaternion tensor in wxyz format

    Returns:
        (3,) euler angles tensor in radians
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(np.pi / 2, device=quat.device),
        torch.asin(sinp),
    )

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Geometry helpers (Genesis entity operations)
# ---------------------------------------------------------------------------


def get_AABB(entity, device) -> torch.Tensor:
    """Get axis-aligned bounding box for a Genesis RigidEntity.

    Handles rotation correctly by transforming corners for Box, Cylinder, and
    Sphere morphs. Falls back to entity.get_AABB() if collision geometry is available.

    Returns:
        (n_envs, 2, 3) tensor: [[min_x, min_y, min_z], [max_x, max_y, max_z]] per env
    """
    from genesis.options.morphs import Box, Cylinder, Sphere

    if entity.n_geoms > 0:
        return entity.get_AABB()

    from trimesh.transformations import quaternion_matrix, transform_points

    all_pos = entity.get_pos().cpu().numpy()  # (n_envs, 3)
    all_quat = entity.get_quat().cpu().numpy()  # (n_envs, 4)
    n_envs = all_pos.shape[0]

    aabb_list = []
    for env_idx in range(n_envs):
        pos = all_pos[env_idx]
        quat = all_quat[env_idx]

        T = quaternion_matrix(quat)
        T[:3, 3] = pos

        if isinstance(entity.morph, Box):
            sx, sy, sz = entity.morph.size
            corners_local = np.array(
                [
                    [-sx / 2, -sy / 2, -sz / 2],
                    [sx / 2, -sy / 2, -sz / 2],
                    [sx / 2, sy / 2, -sz / 2],
                    [-sx / 2, sy / 2, -sz / 2],
                    [-sx / 2, -sy / 2, sz / 2],
                    [sx / 2, -sy / 2, sz / 2],
                    [sx / 2, sy / 2, sz / 2],
                    [-sx / 2, sy / 2, sz / 2],
                ]
            )
            corners_world = transform_points(corners_local, T)
            aabb_list.append([corners_world.min(axis=0), corners_world.max(axis=0)])

        elif isinstance(entity.morph, Cylinder):
            radius = entity.morph.radius
            height = entity.morph.height
            n_pts = 16
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            bottom_circle = np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.full(n_pts, -height / 2)]
            )
            top_circle = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_pts, height / 2)])
            vertices_world = transform_points(np.vstack([bottom_circle, top_circle]), T)
            aabb_list.append([vertices_world.min(axis=0), vertices_world.max(axis=0)])

        elif isinstance(entity.morph, Sphere):
            radius = entity.morph.radius
            aabb_list.append([pos - radius, pos + radius])

        else:
            try:
                env_aabb = entity.get_AABB()[env_idx].cpu().numpy()
                aabb_list.append(env_aabb)
            except Exception:
                aabb_list.append([pos - 0.01, pos + 0.01])

    return torch.tensor(np.array(aabb_list), dtype=torch.float32, device=device)


def compute_2d_convex_hull(entity, env_idx: int = 0) -> NDArray | None:
    """Compute 2D convex hull of a Genesis RigidEntity projected to the XY plane.

    Returns:
        (N, 2) array of hull vertices in CCW order, or None if computation fails.
    """
    from genesis.options.morphs import Box, Cylinder, Sphere
    from scipy.spatial import ConvexHull

    try:
        pos = entity.get_pos()[env_idx].cpu().numpy()
        quat = entity.get_quat()[env_idx].cpu().numpy()

        from trimesh.transformations import quaternion_matrix, transform_points

        T = quaternion_matrix(quat)
        T[:3, 3] = pos

        vertices_3d = None

        if entity.n_geoms > 0 and hasattr(entity, "geoms"):
            try:
                import trimesh

                meshes = []
                for geom in entity.geoms:
                    mesh_copy = geom.get_trimesh().copy()
                    mesh_copy.apply_transform(T)
                    meshes.append(mesh_copy)
                if meshes:
                    vertices_3d = trimesh.util.concatenate(meshes).vertices
            except Exception:
                pass

        if vertices_3d is None:
            if isinstance(entity.morph, Box):
                sx, sy, sz = entity.morph.size
                corners_local = np.array(
                    [
                        [-sx / 2, -sy / 2, -sz / 2],
                        [sx / 2, -sy / 2, -sz / 2],
                        [sx / 2, sy / 2, -sz / 2],
                        [-sx / 2, sy / 2, -sz / 2],
                        [-sx / 2, -sy / 2, sz / 2],
                        [sx / 2, -sy / 2, sz / 2],
                        [sx / 2, sy / 2, sz / 2],
                        [-sx / 2, sy / 2, sz / 2],
                    ]
                )
                vertices_3d = transform_points(corners_local, T)

            elif isinstance(entity.morph, Cylinder):
                radius = entity.morph.radius
                height = entity.morph.height
                n_pts = min(max(16, int(2 * np.pi * radius / 0.01)), 64)
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                bottom = np.column_stack(
                    [radius * np.cos(angles), radius * np.sin(angles), np.full(n_pts, -height / 2)]
                )
                top = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_pts, height / 2)])
                vertices_3d = transform_points(np.vstack([bottom, top]), T)

            elif isinstance(entity.morph, Sphere):
                radius = entity.morph.radius
                n_pts = 32
                phi = np.pi * (3.0 - np.sqrt(5.0))
                points = []
                for i in range(n_pts):
                    y = 1 - (i / (n_pts - 1)) * 2
                    r = np.sqrt(1 - y * y)
                    theta = phi * i
                    points.append([np.cos(theta) * r * radius, y * radius, np.sin(theta) * r * radius])
                vertices_3d = transform_points(np.array(points), T)

        if vertices_3d is None:
            return None

        vertices_2d = vertices_3d[:, :2]
        if len(vertices_2d) < 3:
            return vertices_2d

        hull = ConvexHull(vertices_2d)
        return vertices_2d[hull.vertices]

    except Exception:
        return None
