"""Placement utilities for RoboWits environments.

Provides functions for random object placement with collision avoidance.
"""

from collections.abc import Set as AbstractSet

import numpy as np
import torch

from gs_gym.common.utils.math_utils import euler_to_quat, quat_apply, quat_to_euler


class PlacementError(RuntimeError):
    """Raised when no collision-free random placement can be found within the
    allotted attempts. The caller is expected to catch this and re-roll (e.g.
    re-rotating + re-placing) up to a bounded number of times."""


def _is_fixed_entity(entity) -> bool:
    """Return True if ``entity`` was constructed with ``fixed=True``.

    Genesis' rigid solver pre-builds contact data for ``fixed`` bodies at scene
    build time. Calling ``set_pos`` / ``set_quat`` on such a body during reset
    moves the dynamic state while the cached collision acceleration structures
    stay at the original pose — the next solver step then produces invalid
    constraint forces and raises ``GenesisException: Invalid constraint forces
    causing 'nan'``. Placement therefore must *not* teleport fixed entities,
    even when they appear inside a randomized placement group (where the
    expected semantics are "support-surface / landmark that stays put while
    its dynamic companions are optionally re-arranged around it").
    """
    morph = getattr(entity, "morph", None)
    return bool(getattr(morph, "fixed", False))


def sample_euler_delta_rad(
    rng: np.random.Generator,
    rotation_range: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Sample independent uniform Euler increments (radians) from ``rotation_range``."""
    rx, ry, rz = rotation_range
    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        return (0.0, 0.0, 0.0)
    d_roll = float(rng.uniform(-rx, rx)) if rx != 0.0 else 0.0
    d_pitch = float(rng.uniform(-ry, ry)) if ry != 0.0 else 0.0
    d_yaw = float(rng.uniform(-rz, rz)) if rz != 0.0 else 0.0
    return (d_roll, d_pitch, d_yaw)


def apply_additive_euler_delta_to_quat(
    initial_quat: np.ndarray,
    d_roll: float,
    d_pitch: float,
    d_yaw: float,
) -> np.ndarray:
    """Apply orientation randomization in Euler space (same convention as ``euler_to_quat``).

    ``initial_quat`` is converted with ``quat_to_euler`` (roll, pitch, yaw rad),
    increments are added component-wise, then converted back with ``euler_to_quat``.
    So an object authored at euler (0, 0, π/2) with yaw noise ±0.53 rad ends near
    yaw ∈ [π/2−0.53, π/2+0.53], matching intuition from morph euler angles.

    This replaces the legacy ``delta_quat ⊗ initial_quat`` compose, which did not
    match additive Euler yaw when reporting orientation via ``quat_to_euler``.
    """
    if d_roll == 0.0 and d_pitch == 0.0 and d_yaw == 0.0:
        return np.asarray(initial_quat, dtype=np.float64).copy()

    iq = torch.from_numpy(np.asarray(initial_quat, dtype=np.float64)).unsqueeze(0)
    euler_i = quat_to_euler(iq)[0]
    euler_new = euler_i + torch.tensor([d_roll, d_pitch, d_yaw], dtype=torch.float64, device=euler_i.device)
    q = euler_to_quat(euler_new.unsqueeze(0))[0]
    return q.detach().cpu().numpy().astype(np.float64)


def quat_mul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (``q ⊗ r``)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotate_xy_by_yaw(xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 2D vector by yaw (radians) about the origin."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def transform_particles_like_group_anchor(
    pos: np.ndarray,
    anchor_initial: np.ndarray,
    anchor_pos: np.ndarray,
    delta_quat: np.ndarray,
) -> np.ndarray:
    """Apply group anchor translation plus the same ``delta_quat`` as rigid members.

    Each particle offset ``p - anchor_initial`` is rotated in 3D by ``delta_quat``
    (Hamilton wxyz, same as ``quat_mul_np(delta_quat, …)`` on orientations), then
    anchored at ``anchor_pos``. Roll/pitch/yaw in ``rotation_range`` all affect sand.

    ``pos`` is (N, 3) world positions; returns (N, 3) new world positions.
    """
    if pos.size == 0 or pos.ndim != 2 or pos.shape[1] != 3:
        return pos
    a0 = np.asarray(anchor_initial, dtype=np.float64).reshape(3)
    ap = np.asarray(anchor_pos, dtype=np.float64).reshape(3)
    raw = pos.astype(np.float64, copy=False) - a0.reshape(1, 3)
    n = int(raw.shape[0])
    dq = torch.tensor(np.asarray(delta_quat, dtype=np.float64).reshape(4), dtype=torch.float64).view(1, 4).expand(n, -1)
    rv = torch.tensor(raw, dtype=torch.float64)
    rot = quat_apply(dq, rv).detach().cpu().numpy()
    return (ap.reshape(1, 3) + rot).astype(np.float32, copy=False)


def compute_2d_convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull using Graham scan algorithm.

    Args:
        points: (N, 2) array of 2D points

    Returns:
        (M, 2) array of hull vertices in counter-clockwise order
    """
    if len(points) < 3:
        return points

    # Find the point with lowest y (and leftmost if tied)
    start_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    start = points[start_idx]

    # Sort by polar angle with respect to start point
    def polar_angle(p):
        return np.arctan2(p[1] - start[1], p[0] - start[0])

    sorted_points = sorted(points, key=polar_angle)

    # Graham scan
    hull = []
    for p in sorted_points:
        while len(hull) >= 2:
            # Check if we make a left turn
            o = hull[-2]
            a = hull[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    return np.array(hull)


def check_collision_2d(
    hull1: np.ndarray,
    hull2: np.ndarray,
    margin: float = 0.01,
) -> bool:
    """Check if two 2D convex hulls collide using separating axis theorem.

    Args:
        hull1: (M, 2) array of first hull vertices
        hull2: (N, 2) array of second hull vertices
        margin: Safety margin to add around objects

    Returns:
        True if hulls collide (including margin), False otherwise
    """
    if len(hull1) < 2 or len(hull2) < 2:
        return False

    def get_axes(hull):
        """Get perpendicular axes from hull edges."""
        axes = []
        for i in range(len(hull)):
            edge = hull[(i + 1) % len(hull)] - hull[i]
            # Perpendicular axis (normalized)
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm > 1e-10:
                axes.append(axis / norm)
        return axes

    def project(hull, axis):
        """Project hull onto axis and return min/max."""
        projections = np.dot(hull, axis)
        return projections.min(), projections.max()

    # Check all axes from both hulls
    for axis in get_axes(hull1) + get_axes(hull2):
        min1, max1 = project(hull1, axis)
        min2, max2 = project(hull2, axis)

        # Add margin
        min1 -= margin
        max1 += margin

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # Separated on this axis

    return True  # No separating axis found, hulls collide


def get_object_bounds_2d(
    pos: np.ndarray,
    size: tuple[float, float, float],
    quat: np.ndarray | None = None,
) -> np.ndarray:
    """Get 2D bounding box corners for an object, optionally rotated.

    Args:
        pos: (3,) position of object center
        size: (width, depth, height) of object
        quat: Optional (4,) quaternion (wxyz) for rotation. If None, uses identity.

    Returns:
        (4, 2) array of corner positions in XY plane
    """
    half_w = size[0] / 2
    half_d = size[1] / 2
    x, y = pos[0], pos[1]

    # Local corners relative to center
    local_corners = np.array(
        [
            [-half_w, -half_d],
            [half_w, -half_d],
            [half_w, half_d],
            [-half_w, half_d],
        ]
    )

    if quat is not None:
        # Extract yaw angle from quaternion for 2D rotation
        # For small tilts, we only care about z-axis rotation
        yaw = quat_to_yaw(quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate corners by yaw
        rotated_corners = np.zeros_like(local_corners)
        for i, (cx, cy) in enumerate(local_corners):
            rotated_corners[i, 0] = cx * cos_yaw - cy * sin_yaw
            rotated_corners[i, 1] = cx * sin_yaw + cy * cos_yaw

        local_corners = rotated_corners

    # Translate to world position
    world_corners = local_corners + np.array([x, y])

    return world_corners


def quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw angle (z-axis rotation) from quaternion.

    Args:
        quat: (4,) quaternion in wxyz format

    Returns:
        Yaw angle in radians
    """
    w, x, y, z = quat
    # Yaw from quaternion
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_oriented_hull_2d(
    pos: np.ndarray,
    size: tuple[float, float, float],
    quat: np.ndarray | None = None,
) -> np.ndarray:
    """Get 2D convex hull for an oriented object.

    Args:
        pos: (3,) position of object center
        size: (width, depth, height) of object
        quat: Optional (4,) quaternion (wxyz) for rotation

    Returns:
        (M, 2) array of convex hull vertices
    """
    corners = get_object_bounds_2d(pos, size, quat)
    return compute_2d_convex_hull(corners)


def place_objects_randomly(
    object_sizes: list[tuple[float, float, float]],
    reachable_bounds: tuple[tuple[float, float], tuple[float, float]],
    table_height: float = 0.0,
    margin: float = 0.02,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None,
    object_quats: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Place objects randomly within bounds without collision (single placement).

    Args:
        object_sizes: List of (width, depth, height) for each object
        reachable_bounds: ((x_min, x_max), (y_min, y_max)) placement area
        table_height: Height of table surface
        margin: Minimum separation between objects
        max_attempts: Maximum placement attempts per object
        rng: Random number generator
        object_quats: Optional list of (4,) quaternions (wxyz) for each object.
                     If provided, uses oriented convex hulls for collision detection.

    Returns:
        List of (3,) position arrays for each object
    """
    if rng is None:
        rng = np.random.default_rng()

    x_range = reachable_bounds[0]
    y_range = reachable_bounds[1]

    placed_positions: list[np.ndarray] = []
    placed_hulls: list[np.ndarray] = []

    for i, size in enumerate(object_sizes):
        # Get orientation for this object (if provided)
        quat = object_quats[i] if object_quats is not None else None
        # Account for object size in placement bounds
        x_min = x_range[0] + size[0] / 2
        x_max = x_range[1] - size[0] / 2
        y_min = y_range[0] + size[1] / 2
        y_max = y_range[1] - size[1] / 2

        if x_max < x_min or y_max < y_min:
            # Object too large for bounds, place at center
            pos = np.array(
                [
                    (x_range[0] + x_range[1]) / 2,
                    (y_range[0] + y_range[1]) / 2,
                    table_height + size[2] / 2,
                ]
            )
            placed_positions.append(pos)
            placed_hulls.append(get_oriented_hull_2d(pos, size, quat))
            continue

        for _attempt in range(max_attempts):
            # Sample random position
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            z = table_height + size[2] / 2
            pos = np.array([x, y, z])

            # Get hull for this position (oriented if quat provided)
            new_hull = get_oriented_hull_2d(pos, size, quat)

            # Check collision with all placed objects
            collision = any(check_collision_2d(new_hull, hull, margin=margin) for hull in placed_hulls)

            if not collision:
                placed_positions.append(pos)
                placed_hulls.append(new_hull)
                break
        else:
            # Failed to place after max attempts, use last attempted position
            placed_positions.append(pos)
            placed_hulls.append(new_hull)

    return placed_positions


def place_objects_randomly_batched(
    object_sizes: list[tuple[float, float, float]],
    reachable_bounds: tuple[tuple[float, float], tuple[float, float]],
    n_envs: int,
    table_height: float = 0.0,
    margin: float = 0.02,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None,
    object_quats: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Place objects randomly with different positions per environment.

    Args:
        object_sizes: List of (width, depth, height) for each object
        reachable_bounds: ((x_min, x_max), (y_min, y_max)) placement area
        n_envs: Number of parallel environments
        table_height: Height of table surface
        margin: Minimum separation between objects
        max_attempts: Maximum placement attempts per object
        rng: Random number generator
        object_quats: Optional list of (4,) quaternions (wxyz) for each object.
                     If provided, uses oriented convex hulls for collision detection.

    Returns:
        List of (n_envs, 3) position arrays for each object
    """
    if rng is None:
        rng = np.random.default_rng()

    n_objects = len(object_sizes)

    # Generate placements for each environment
    all_positions = []
    for _ in range(n_objects):
        all_positions.append(np.zeros((n_envs, 3)))

    for env_idx in range(n_envs):
        # Generate one set of positions for this environment
        positions = place_objects_randomly(
            object_sizes=object_sizes,
            reachable_bounds=reachable_bounds,
            table_height=table_height,
            margin=margin,
            max_attempts=max_attempts,
            rng=rng,
            object_quats=object_quats,
        )
        for obj_idx, pos in enumerate(positions):
            all_positions[obj_idx][env_idx] = pos

    return all_positions


def check_objects_on_table(
    positions: torch.Tensor,
    table_height: float = 0.0,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Check if objects are on the table (haven't fallen).

    Args:
        positions: (n_envs, n_objects, 3) object positions
        table_height: Height of table surface
        tolerance: How far below table is considered "fallen"

    Returns:
        (n_envs,) boolean tensor, True if all objects on table
    """
    z_coords = positions[..., 2]
    on_table = z_coords >= (table_height - tolerance)
    return on_table.all(dim=-1)


def check_collinearity(
    positions: torch.Tensor,
    tolerance: float = 0.01,
) -> torch.Tensor:
    """Check if 3 points are collinear within tolerance.

    Uses the area of triangle formed by the three points.
    Area = 0.5 * |cross product of two edge vectors|

    Args:
        positions: (n_envs, 3, 3) positions of 3 objects (x, y, z per object)
        tolerance: Maximum deviation from perfect collinearity in meters

    Returns:
        (n_envs,) boolean tensor, True if collinear within tolerance
    """
    # Extract 2D positions (x, y only)
    p1 = positions[:, 0, :2]  # (n_envs, 2)
    p2 = positions[:, 1, :2]
    p3 = positions[:, 2, :2]

    # Compute vectors from p1 to p2 and p1 to p3
    v1 = p2 - p1  # (n_envs, 2)
    v2 = p3 - p1

    # 2D cross product (gives signed area * 2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    area = torch.abs(cross) / 2

    # Compute base length (longest edge) for normalization
    d12 = torch.norm(v1, dim=1)
    d13 = torch.norm(v2, dim=1)
    d23 = torch.norm(p3 - p2, dim=1)
    base = torch.max(torch.max(d12, d13), d23)

    # Height of triangle = 2 * area / base
    # This gives the perpendicular distance of the furthest point from the line
    height = 2 * area / (base + 1e-6)

    return height <= tolerance


def compute_group_placement_bounds(
    offsets: dict[str, tuple[float, float, float]],
    sizes: dict[str, tuple[float, float, float]],
    reachable_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute anchor placement bounds that keep entire group within reachable area.

    This function computes where the anchor object can be placed such that all
    objects in the group (defined by their offsets from anchor and sizes) stay
    within the reachable bounds.

    Args:
        offsets: Dict mapping object name to (x, y, z) offset from anchor.
                 The anchor should have offset (0, 0, 0).
        sizes: Dict mapping object name to (sx, sy, sz) bounding box size.
        reachable_bounds: ((x_min, x_max), (y_min, y_max)) reachable area.

    Returns:
        ((anchor_x_min, anchor_x_max), (anchor_y_min, anchor_y_max)) bounds
        for valid anchor placement. If the group is too large to fit, the
        bounds are clamped to the center of the reachable area.
    """
    if not offsets:
        return reachable_bounds

    x_range, y_range = reachable_bounds
    names = list(offsets.keys())

    # Compute group bounding box using offsets ± size/2 for true extents
    group_x_min = min(offsets[n][0] - sizes[n][0] / 2 for n in names)
    group_x_max = max(offsets[n][0] + sizes[n][0] / 2 for n in names)
    group_y_min = min(offsets[n][1] - sizes[n][1] / 2 for n in names)
    group_y_max = max(offsets[n][1] + sizes[n][1] / 2 for n in names)

    # Compute anchor placement bounds:
    # When anchor is at x_min, leftmost object edge should be at x_range[0]
    # When anchor is at x_max, rightmost object edge should be at x_range[1]
    anchor_x_min = x_range[0] - group_x_min
    anchor_x_max = x_range[1] - group_x_max
    anchor_y_min = y_range[0] - group_y_min
    anchor_y_max = y_range[1] - group_y_max

    # Clamp to valid range if group is too large
    if anchor_x_max < anchor_x_min:
        anchor_x_min = anchor_x_max = (x_range[0] + x_range[1]) / 2
    if anchor_y_max < anchor_y_min:
        anchor_y_min = anchor_y_max = (y_range[0] + y_range[1]) / 2

    return ((anchor_x_min, anchor_x_max), (anchor_y_min, anchor_y_max))


def check_minimum_spread(
    positions: torch.Tensor,
    min_distance: float = 0.03,
) -> torch.Tensor:
    """Check if objects have minimum spread (not too close together).

    Args:
        positions: (n_envs, n_objects, 3) object positions
        min_distance: Minimum distance between furthest objects

    Returns:
        (n_envs,) boolean tensor, True if spread is sufficient
    """
    n_envs, n_objects = positions.shape[:2]

    # Compute pairwise distances
    max_dist = torch.zeros(n_envs, device=positions.device)

    for i in range(n_objects):
        for j in range(i + 1, n_objects):
            dist = torch.norm(positions[:, i, :2] - positions[:, j, :2], dim=1)
            max_dist = torch.max(max_dist, dist)

    return max_dist >= min_distance


def parse_reachable_bounds(bounds) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse reachable_bounds from config to tuple format.

    Supports both formats:
    - Tuple format: ((x_min, x_max), (y_min, y_max))
    - Dict format: {"x": [x_min, x_max], "y": [y_min, y_max]}

    Returns:
        Tuple format: ((x_min, x_max), (y_min, y_max))
    """
    if isinstance(bounds, dict):
        x_range = bounds.get("x", [0.3, 0.7])
        y_range = bounds.get("y", [-0.25, 0.25])
        return (tuple(x_range), tuple(y_range))
    return bounds


def get_object_reachable_area(
    name: str,
    default_bounds: tuple[tuple[float, float], tuple[float, float]],
    object_reachable_areas: dict[str, dict[str, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get reachable area for an object, with per-object override support."""
    if name in object_reachable_areas:
        area = object_reachable_areas[name]
        return (
            (
                area.get("x_min", default_bounds[0][0]),
                area.get("x_max", default_bounds[0][1]),
            ),
            (
                area.get("y_min", default_bounds[1][0]),
                area.get("y_max", default_bounds[1][1]),
            ),
        )
    return default_bounds


def _compute_free_space_grid_single(
    size: tuple[float, float, float],
    quat: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    placed_hulls: list[np.ndarray],
    resolution: int = 40,
    margin: float = 0.02,
) -> dict:
    """Sample a grid over ``bounds`` and record which centers are collision-free.

    Used by the placement debugger to visualize why a single-object placement
    failed. Returns a dict describing the effective (size-shrunk) bounds, the
    grid axes, and a boolean ``free`` mask (True = collision-free).
    """
    (x_min_raw, x_max_raw), (y_min_raw, y_max_raw) = bounds
    x_min = x_min_raw + size[0] / 2
    x_max = x_max_raw - size[0] / 2
    y_min = y_min_raw + size[1] / 2
    y_max = y_max_raw - size[1] / 2

    valid_bounds = x_max >= x_min and y_max >= y_min
    xs = np.linspace(x_min, x_max, resolution) if valid_bounds else np.array([])
    ys = np.linspace(y_min, y_max, resolution) if valid_bounds else np.array([])
    free = np.zeros((len(ys), len(xs)), dtype=bool)

    if valid_bounds:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                pos = np.array([x, y, 0.0])
                hull = get_oriented_hull_2d(pos, size, quat)
                collides = any(check_collision_2d(hull, h, margin=margin) for h in placed_hulls)
                free[iy, ix] = not collides

    return {
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "free": free.tolist(),
        "effective_bounds": [[x_min, x_max], [y_min, y_max]] if valid_bounds else None,
    }


def _hulls_to_list(hulls: list[np.ndarray]) -> list[list[list[float]]]:
    return [np.asarray(h, dtype=float).tolist() for h in hulls]


def place_single_object(
    name: str,
    entities: dict,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    placed_hulls: list[np.ndarray],
    rng: np.random.Generator,
    device,
    is_particle_fn,
    rotation_range: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_attempts: int = 100,
    debug: dict | None = None,
    placement_target_names: AbstractSet[str] | None = None,
) -> None:
    """Place a single object randomly, avoiding collisions with placed objects.

    Preserves the entity's initial Z position. Orientation randomization applies
    uniform Euler increments (roll/pitch/yaw in radians, independent axes from
    ``rotation_range``) **on top of** the morph's initial quaternion in Euler
    space (``quat_to_euler`` → add deltas → ``euler_to_quat``), so e.g. yaw noise
    for an object authored at euler z = 90° is centered on 90°, not on 0°.

    Only X and Y are randomized.

    If ``placement_target_names`` is set and ``name`` is not in it, returns
    immediately without moving the entity (rigid / MPM / SPH stay at reset pose).

    Raises:
        PlacementError: If no collision-free position can be found within
            ``max_attempts`` samples.
    """
    if name not in entities:
        return

    if placement_target_names is not None and name not in placement_target_names:
        return

    entity_info = entities[name]
    entity = entity_info["entity"]
    if is_particle_fn(entity):
        return

    size = entity_info["size"]
    initial_pos = entity_info.get("initial_pos")
    initial_quat = entity_info.get("initial_quat")
    if initial_pos is None or initial_quat is None:
        initial_pos = entity.get_pos()[0].cpu().numpy()
        initial_quat = entity.get_quat()[0].cpu().numpy()

    d_roll, d_pitch, d_yaw = sample_euler_delta_rad(rng, rotation_range)
    quat = apply_additive_euler_delta_to_quat(initial_quat, d_roll, d_pitch, d_yaw)

    initial_z = initial_pos[2]
    x_range, y_range = bounds
    x_min = x_range[0] + size[0] / 2
    x_max = x_range[1] - size[0] / 2
    y_min = y_range[0] + size[1] / 2
    y_max = y_range[1] - size[1] / 2

    # Raise a clear PlacementError if the declared footprint does not fit in
    # the reachable bounds — otherwise ``rng.uniform(x_min, x_max)`` throws a
    # cryptic ``high - low < 0`` from numpy a few frames deep.
    if x_max < x_min or y_max < y_min:
        if debug is not None:
            debug.update(
                {
                    "kind": "single",
                    "name": name,
                    "bounds": [list(bounds[0]), list(bounds[1])],
                    "effective_bounds": [[x_min, x_max], [y_min, y_max]],
                    "size": list(size),
                    "quat": quat.tolist(),
                    "yaw": float(quat_to_yaw(quat)),
                    "initial_pos": [float(v) for v in initial_pos],
                    "initial_quat": [float(v) for v in initial_quat],
                    "placed_hulls": _hulls_to_list(placed_hulls),
                    "tried_positions": [],
                    "max_attempts": max_attempts,
                    "reason": "footprint_larger_than_bounds",
                }
            )
        raise PlacementError(
            f"place_single_object: footprint of '{name}' ({size[0]:.3f} x {size[1]:.3f}) "
            f"does not fit in reachable bounds "
            f"x[{x_range[0]:.3f},{x_range[1]:.3f}] y[{y_range[0]:.3f},{y_range[1]:.3f}] "
            f"(needs >= {size[0]:.3f} along x and >= {size[1]:.3f} along y, "
            f"effective=x[{x_min:.3f},{x_max:.3f}] y[{y_min:.3f},{y_max:.3f}], "
            f"initial_pos=({initial_pos[0]:.3f},{initial_pos[1]:.3f},{initial_pos[2]:.3f}), "
            f"yaw={float(quat_to_yaw(quat)):.3f} rad). "
            f"Either shrink the declared size or widen the reachable area."
        )

    pos = None
    new_hull = None
    tried_positions: list[list[float]] = []
    for _ in range(max_attempts):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        tried_positions.append([float(x), float(y)])
        pos = np.array([x, y, initial_z])
        new_hull = get_oriented_hull_2d(pos, size, quat)
        collision = any(check_collision_2d(new_hull, h, margin=0.02) for h in placed_hulls)
        if not collision:
            placed_hulls.append(new_hull)
            break
    else:
        if debug is not None:
            debug.update(
                {
                    "kind": "single",
                    "name": name,
                    "bounds": [list(bounds[0]), list(bounds[1])],
                    "effective_bounds": [[x_min, x_max], [y_min, y_max]],
                    "size": list(size),
                    "quat": quat.tolist(),
                    "yaw": float(quat_to_yaw(quat)),
                    "initial_pos": [float(v) for v in initial_pos],
                    "initial_quat": [float(v) for v in initial_quat],
                    "placed_hulls": _hulls_to_list(placed_hulls),
                    "tried_positions": tried_positions,
                    "max_attempts": max_attempts,
                    "grid": _compute_free_space_grid_single(
                        size=size, quat=quat, bounds=bounds, placed_hulls=placed_hulls
                    ),
                }
            )
        tried_summary = ""
        if tried_positions:
            arr = np.asarray(tried_positions, dtype=float)
            tried_summary = (
                f", sampled x in [{arr[:, 0].min():.3f}, {arr[:, 0].max():.3f}] "
                f"y in [{arr[:, 1].min():.3f}, {arr[:, 1].max():.3f}]"
            )
        raise PlacementError(
            f"place_single_object: no collision-free placement for '{name}' "
            f"within {max_attempts} attempts "
            f"(raw_bounds=x[{x_range[0]:.3f},{x_range[1]:.3f}] y[{y_range[0]:.3f},{y_range[1]:.3f}], "
            f"effective_bounds=x[{x_min:.3f},{x_max:.3f}] y[{y_min:.3f},{y_max:.3f}], "
            f"footprint=({size[0]:.3f}x{size[1]:.3f}), "
            f"initial_pos=({initial_pos[0]:.3f},{initial_pos[1]:.3f},{initial_pos[2]:.3f}), "
            f"yaw={float(quat_to_yaw(quat)):.3f} rad, "
            f"n_placed_hulls={len(placed_hulls)}{tried_summary})"
        )

    pos_tensor = torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
    quat_tensor = torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0)
    entity.set_pos(pos_tensor)
    entity.set_quat(quat_tensor)


def _compute_free_space_grid_group(
    valid_names: list[str],
    sizes: dict[str, tuple[float, float, float]],
    rotated_offsets: dict[str, np.ndarray],
    rotated_quats: dict[str, np.ndarray],
    anchor_bounds: tuple[tuple[float, float], tuple[float, float]],
    placed_hulls: list[np.ndarray],
    resolution: int = 40,
    margin: float = 0.02,
) -> dict:
    """Grid of collision-free anchor positions for a group placement."""
    (x_min, x_max), (y_min, y_max) = anchor_bounds
    valid = x_max >= x_min and y_max >= y_min
    xs = np.linspace(x_min, x_max, resolution) if valid else np.array([])
    ys = np.linspace(y_min, y_max, resolution) if valid else np.array([])
    free = np.zeros((len(ys), len(xs)), dtype=bool)

    if valid:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                anchor = np.array([x, y, 0.0])
                ok = True
                for name in valid_names:
                    hull = get_oriented_hull_2d(anchor + rotated_offsets[name], sizes[name], rotated_quats[name])
                    if any(check_collision_2d(hull, h, margin=margin) for h in placed_hulls):
                        ok = False
                        break
                free[iy, ix] = ok

    return {
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "free": free.tolist(),
    }


def place_group_together(
    group: tuple[str, ...],
    entities: dict,
    default_bounds: tuple[tuple[float, float], tuple[float, float]],
    placed_hulls: list[np.ndarray],
    rng: np.random.Generator,
    device,
    is_particle_fn,
    rotation_range: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_attempts: int = 100,
    debug: dict | None = None,
    ignored_hulls: set[str] | frozenset[str] | None = None,
    placement_target_names: AbstractSet[str] | None = None,
) -> None:
    """Place a group of objects together, maintaining relative positions.

    The first object in the group is the anchor. A single rotation delta is
    sampled from ``rotation_range`` and applied to every member's orientation,
    and the followers' XY offsets from the anchor are rotated by the same yaw.
    Z offsets are preserved exactly.

    Particle / MPM / SPH members are only transformed when listed in
    ``placement_target_names`` (when that set is provided).

    Raises:
        PlacementError: If no collision-free anchor placement can be found
            within ``max_attempts`` samples.
    """
    valid_names = [n for n in group if n in entities]
    if not valid_names:
        return

    initial_positions: dict[str, np.ndarray] = {}
    initial_quats: dict[str, np.ndarray] = {}
    rigid_names = []
    for name in valid_names:
        entity_info = entities[name]
        entity = entity_info["entity"]
        if is_particle_fn(entity):
            continue
        rigid_names.append(name)
        pos = entity_info.get("initial_pos")
        quat = entity_info.get("initial_quat")
        if pos is None or quat is None:
            pos = entity.get_pos()[0].cpu().numpy()
            quat = entity.get_quat()[0].cpu().numpy()
        initial_positions[name] = pos
        initial_quats[name] = quat
    valid_names = rigid_names
    if not valid_names:
        return

    anchor_name = valid_names[0]
    anchor_initial = initial_positions[anchor_name]

    # Sample one rotation delta for the whole group: build ``delta_quat`` from the
    # same Euler triple used elsewhere, then left-multiply each member's initial
    # orientation (rigid-group semantics). Single-object placement instead adds
    # the Euler delta to each body's ``quat_to_euler(initial)`` so reported yaw
    # matches morph euler intuition.
    d_roll, d_pitch, d_yaw = sample_euler_delta_rad(rng, rotation_range)
    if d_roll == 0.0 and d_pitch == 0.0 and d_yaw == 0.0:
        delta_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        de = torch.tensor([d_roll, d_pitch, d_yaw], dtype=torch.float64)
        delta_quat = euler_to_quat(de.unsqueeze(0))[0].detach().cpu().numpy().astype(np.float64)
    delta_yaw = quat_to_yaw(delta_quat)
    rotated_quats = {n: quat_mul_np(delta_quat, initial_quats[n]) for n in valid_names}
    rotated_offsets: dict[str, np.ndarray] = {}
    for name in valid_names:
        raw_off = initial_positions[name] - anchor_initial
        rot_xy = rotate_xy_by_yaw(raw_off[:2], delta_yaw) if delta_yaw != 0.0 else raw_off[:2]
        rotated_offsets[name] = np.array([rot_xy[0], rot_xy[1], raw_off[2]])

    offset_dict = {n: tuple(rotated_offsets[n]) for n in valid_names}
    sizes_dict = {n: entities[n]["size"] for n in valid_names}
    (x_min, x_max), (y_min, y_max) = compute_group_placement_bounds(offset_dict, sizes_dict, default_bounds)

    anchor_z = anchor_initial[2]
    anchor_pos = None
    group_hulls: list[np.ndarray] = []
    tried_positions: list[list[float]] = []

    for _ in range(max_attempts):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        tried_positions.append([float(x), float(y)])
        anchor_pos = np.array([x, y, anchor_z])

        collision = False
        group_hulls = []
        for name in valid_names:
            size = entities[name]["size"]
            obj_pos = anchor_pos + rotated_offsets[name]
            hull = get_oriented_hull_2d(obj_pos, size, rotated_quats[name])
            group_hulls.append(hull)
            for placed_hull in placed_hulls:
                if check_collision_2d(hull, placed_hull, margin=0.02):
                    collision = True
                    break
            if collision:
                break

        if not collision:
            placed_hulls.extend(group_hulls)
            break
    else:
        if debug is not None:
            sizes_dict_native = {n: list(entities[n]["size"]) for n in valid_names}
            offsets_native = {n: rotated_offsets[n].tolist() for n in valid_names}
            quats_native = {n: rotated_quats[n].tolist() for n in valid_names}
            debug.update(
                {
                    "kind": "group",
                    "group": list(valid_names),
                    "default_bounds": [list(default_bounds[0]), list(default_bounds[1])],
                    "anchor_bounds": [[x_min, x_max], [y_min, y_max]],
                    "sizes": sizes_dict_native,
                    "rotated_offsets": offsets_native,
                    "rotated_quats": quats_native,
                    "delta_yaw": float(delta_yaw),
                    "initial_positions": {n: initial_positions[n].tolist() for n in valid_names},
                    "placed_hulls": _hulls_to_list(placed_hulls),
                    "tried_positions": tried_positions,
                    "max_attempts": max_attempts,
                    "grid": _compute_free_space_grid_group(
                        valid_names=valid_names,
                        sizes={n: entities[n]["size"] for n in valid_names},
                        rotated_offsets=rotated_offsets,
                        rotated_quats=rotated_quats,
                        anchor_bounds=((x_min, x_max), (y_min, y_max)),
                        placed_hulls=placed_hulls,
                    ),
                }
            )
        tried_summary = ""
        if tried_positions:
            arr = np.asarray(tried_positions, dtype=float)
            tried_summary = (
                f", sampled x in [{arr[:, 0].min():.3f}, {arr[:, 0].max():.3f}] "
                f"y in [{arr[:, 1].min():.3f}, {arr[:, 1].max():.3f}]"
            )
        sizes_str = ", ".join(f"{n}({entities[n]['size'][0]:.3f}x{entities[n]['size'][1]:.3f})" for n in valid_names)
        raise PlacementError(
            f"place_group_together: no collision-free placement for group "
            f"{tuple(valid_names)} within {max_attempts} attempts "
            f"(default_bounds=x[{default_bounds[0][0]:.3f},{default_bounds[0][1]:.3f}] "
            f"y[{default_bounds[1][0]:.3f},{default_bounds[1][1]:.3f}], "
            f"anchor_bounds=x[{x_min:.3f},{x_max:.3f}] y[{y_min:.3f},{y_max:.3f}], "
            f"delta_yaw={float(delta_yaw):.3f} rad, footprints=[{sizes_str}], "
            f"n_placed_hulls={len(placed_hulls)}{tried_summary})"
        )

    for name in valid_names:
        if placement_target_names is not None and name not in placement_target_names:
            continue
        entity = entities[name]["entity"]
        obj_pos = anchor_pos + rotated_offsets[name]
        pos_tensor = torch.tensor(obj_pos, dtype=torch.float32, device=device).unsqueeze(0)
        quat_tensor = torch.tensor(rotated_quats[name], dtype=torch.float32, device=device).unsqueeze(0)
        entity.set_pos(pos_tensor)
        entity.set_quat(quat_tensor)

    # Particle (e.g. MPM / SPH) members: anchor translation + full ``delta_quat`` on offsets from anchor.
    for name in group:
        if name not in entities:
            continue
        if placement_target_names is not None and name not in placement_target_names:
            continue
        entity = entities[name]["entity"]
        if not is_particle_fn(entity):
            continue
        if not hasattr(entity, "get_particles_pos") or not hasattr(entity, "set_particles_pos"):
            continue
        t = entity.get_particles_pos().detach()
        pos_np = t[0].cpu().numpy() if t.dim() == 3 else t.cpu().numpy()
        if pos_np.size == 0:
            continue
        new_np = transform_particles_like_group_anchor(pos_np, anchor_initial, anchor_pos, delta_quat)
        poss = torch.as_tensor(new_np, dtype=torch.float32, device=device)
        entity.set_particles_pos(poss)
        if hasattr(entity, "set_particles_vel"):
            entity.set_particles_vel(torch.zeros_like(poss))
