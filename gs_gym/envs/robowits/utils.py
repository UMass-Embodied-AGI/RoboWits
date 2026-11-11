"""Shared utilities for RoboWits environments.

Polygon operations used in success criteria, plus entity inspection and
geometry helpers moved from RoboWitsEnv.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray

# Re-exported: these used to be defined here, duplicating math_utils. They are
# general rotation math, so they live there now and are imported for the task
# modules that already reach for them via this module.
from gs_gym.common.utils.math_utils import axis_angle_to_quat, quat_to_axis_angle, quat_to_euler

logger = logging.getLogger(__name__)

__all__ = ["axis_angle_to_quat", "quat_to_axis_angle", "quat_to_euler"]

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
# Reward / success prerequisites
# ---------------------------------------------------------------------------


def require_objs(env, objs_info: dict, names: list[str]) -> bool:
    """Assert that a reward or success check can see the entities it needs.

    A name absent from ``env._entities`` is a scene-definition bug — typically a
    mutation variant that renamed or dropped an object its inherited reward still
    references. That pins the progress score at exactly 0.0 for every episode of
    that variant, which is indistinguishable from a policy that never moves, so it
    raises rather than scoring 0.

    A name that *is* registered but missing from ``objs_info`` means state
    collection skipped it this step (``collect_objs_info`` swallows per-entity
    errors). That is transient, so it warns and returns False, letting the caller
    score 0.0 for the step.
    """
    missing = [n for n in names if n not in objs_info]
    if not missing:
        return True

    registered = set(getattr(env, "_entities", {}))
    unregistered = [n for n in missing if n not in registered]
    if unregistered:
        raise RuntimeError(
            f"{type(env).__name__}: reward/success requires entities {unregistered} that this scene never "
            f"registers. Registered entities: {sorted(registered)}. Either add them in _add_custom_entities() "
            f"under these exact names, or override get_reward()/_check_success() for this variant."
        )

    logger.warning(
        f"{type(env).__name__}: entities {missing} are registered but absent from collect_objs_info() this "
        f"step; scoring 0.0 for this step."
    )
    return False


# ---------------------------------------------------------------------------
# Grasp / held detection
# ---------------------------------------------------------------------------


def check_released(
    obj_info: dict,
    robot,
    tol: float = 0.03,
) -> bool:
    """Check that neither gripper is holding an object.

    The object counts as held while either gripper's end-effector position is within the object's axis-aligned
    bounding box, expanded by a tolerance to account for the gripper tip not being perfectly centered inside the
    object. Stateless, like check_still.

    Args:
        obj_info: Object info dict from collect_objs_info(), must contain 'bounds' key
            with shape (2, 3) as [[xmin, ymin, zmin], [xmax, ymax, zmax]].
        robot: BimanualRobot instance with right_ee_pose and left_ee_pose attributes.
        tol: Tolerance in meters to expand the bounding box on each side.

    Returns:
        True if neither gripper holds the object (also when 'bounds' is missing or malformed), False otherwise.
    """
    bounds = obj_info.get("bounds")
    if bounds is None:
        return True

    bounds = np.array(bounds)
    if bounds.shape != (2, 3):
        return True

    xmin, ymin, zmin = bounds[0] - tol
    xmax, ymax, zmax = bounds[1] + tol

    right_ee_pos = robot.right_ee_pose[0, :3].cpu().numpy()
    left_ee_pos = robot.left_ee_pose[0, :3].cpu().numpy()

    def point_in_bounds(pos):
        return (xmin <= pos[0] <= xmax) and (ymin <= pos[1] <= ymax) and (zmin <= pos[2] <= zmax)

    return not (point_in_bounds(right_ee_pos) or point_in_bounds(left_ee_pos))


def check_upright(obj_info: dict, max_tilt_deg: float = 15.0) -> bool:
    """Check that an object stands upright: its body z-axis within max_tilt_deg of world up.

    Exact for any orientation (computed from the quaternion, not from roll and pitch separately), and stateless
    like check_released and check_still. "Upright" is the object's spawn orientation, so it applies to objects
    whose upright pose is their identity rotation.

    Args:
        obj_info: Rigid object info dict from collect_objs_info(), with 'quat' (4,) in wxyz order.
        max_tilt_deg: Largest angle between the body z-axis and world z, in degrees.

    Returns:
        True if the tilt is within max_tilt_deg; False if 'quat' is missing.
    """
    quat = obj_info.get("quat")
    if quat is None:
        return False
    w, x, y, z = (float(q) for q in np.asarray(quat, dtype=float).reshape(4))
    n2 = w * w + x * x + y * y + z * z
    if n2 <= 0.0:
        return False
    # World z component of the body z-axis, for a possibly non-unit quaternion
    up_z = 1.0 - 2.0 * (x * x + y * y) / n2
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))) <= max_tilt_deg


# Stillness limits shared by every task's check_still call. Resting objects measure <= 0.005 m/s; task 04's
# card sliding off or tumbling measured 0.15-0.42 m/s.
STILL_LIN_SPEED = 0.02  # m/s
STILL_ANG_SPEED = 0.5  # rad/s


def check_still(
    obj_info: dict,
    lin_tol: float = STILL_LIN_SPEED,
    ang_tol: float = STILL_ANG_SPEED,
) -> bool:
    """Check if a rigid object is still: neither moving nor turning.

    Stateless, like check_released: it reads only the current velocities, so it cannot tell an object at rest
    from one that is motionless for an instant (just released on a perch it will roll off, or at the top of a
    bounce).

    Args:
        obj_info: Rigid object info dict from collect_objs_info(), with 'vel' (3,) in m/s and 'ang' (3,) in
            rad/s. Snapshots recorded before 'ang' existed (eval objs_info.json) are checked on 'vel' alone.
        lin_tol: Largest linear speed, m/s.
        ang_tol: Largest angular speed, rad/s.

    Returns:
        True if both speeds are within their limits; False if 'vel' is missing.
    """
    vel = obj_info.get("vel")
    if vel is None:
        return False
    vel = np.asarray(vel, dtype=float)
    if vel.shape != (3,):
        raise ValueError(f"check_still takes a rigid object's (3,) velocity, got shape {vel.shape}")
    if float(np.linalg.norm(vel)) > lin_tol:
        return False
    ang = obj_info.get("ang")
    return ang is None or float(np.linalg.norm(np.asarray(ang, dtype=float))) <= ang_tol


# ---------------------------------------------------------------------------
# Dense-reward stages
# ---------------------------------------------------------------------------

# Tasks whose dense reward follows their success check share one ladder, so a score means the same thing in each:
#   0 -> STAGE_APPROACH          a gripper approaches the object
#   STAGE_APPROACH -> STAGE_PROGRESS   the object moves toward the goal (added to the approach score, so an
#                                      untouched object at spawn scores 0), then partly meets the goal geometry
#   STAGE_HELD_AT_GOAL           the goal geometry is met but the object is still held
#   STAGE_MOVING_AT_GOAL         the goal geometry is met, released, but the object is still moving
#   1.0                          success
STAGE_APPROACH = 0.2
STAGE_PROGRESS = 0.65
STAGE_HELD_AT_GOAL = 0.7
STAGE_MOVING_AT_GOAL = 0.85


def approach_score(robot, points, approach: float = 0.3, reach: float = 0.15, touch: float = 0.05) -> float:
    """First rung of the ladder: 0 -> STAGE_APPROACH as the nearer gripper closes on the nearest of `points`.

    Args:
        robot: BimanualRobot instance with right_ee_pose and left_ee_pose attributes.
        points: World positions (3,) of the objects worth approaching; None entries are skipped.
        approach, reach, touch: Distances (m) at which the score starts, reaches half, and reaches full.
    """
    ees = [robot.right_ee_pose[0, :3].cpu().numpy(), robot.left_ee_pose[0, :3].cpu().numpy()]
    pts = [np.asarray(p, dtype=float)[:3] for p in points if p is not None]
    if not pts:
        return 0.0
    d = min(float(np.linalg.norm(ee - p)) for ee in ees for p in pts)
    if d <= touch:
        return STAGE_APPROACH
    if d <= reach:
        return STAGE_APPROACH * (0.5 + 0.5 * (reach - d) / (reach - touch))
    if d < approach:
        return STAGE_APPROACH * 0.5 * (approach - d) / (approach - reach)
    return 0.0


def progress_frac(pos, spawn, goal, done_dist: float = 0.0, fallback_scale: float = 0.3) -> float:
    """Fraction of the way, in the table plane, from where the object spawned to within done_dist of goal.

    0 at the spawn distance (or farther), 1 within done_dist. Measured from this episode's spawn position, so an
    idle policy scores 0 wherever randomization put the object. Without a spawn position (e.g. a replayed
    snapshot), the distance is scored on a fixed fallback_scale instead.
    """
    d = float(np.linalg.norm(np.asarray(pos, dtype=float)[:2] - np.asarray(goal, dtype=float)[:2]))
    if spawn is not None:
        d0 = float(np.linalg.norm(np.asarray(spawn, dtype=float)[:2] - np.asarray(goal, dtype=float)[:2]))
        if d0 > done_dist + 1e-6:
            return float(np.clip((d0 - d) / (d0 - done_dist), 0.0, 1.0))
    return float(np.clip(1.0 - (d - done_dist) / fallback_scale, 0.0, 1.0))


def at_goal_score(released: bool) -> float:
    """Score for a met goal geometry while success is still pending: held, or released but still moving."""
    return STAGE_MOVING_AT_GOAL if released else STAGE_HELD_AT_GOAL


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------


def is_particle_entity(entity) -> bool:
    """Return True if entity is a particle-based (MPM or SPH) entity.

    Both ``MPMEntity`` and ``SPHEntity`` subclass ``ParticleEntity``, which has
    ``set_particles_pos``/``set_position`` but no ``set_pos``/``set_quat``.
    """
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

    all_pos = entity.get_pos(relative=False).cpu().numpy()  # (n_envs, 3)
    all_quat = entity.get_quat(relative=False).cpu().numpy()  # (n_envs, 4)
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
        pos = entity.get_pos(relative=False)[env_idx].cpu().numpy()
        quat = entity.get_quat(relative=False)[env_idx].cpu().numpy()

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
