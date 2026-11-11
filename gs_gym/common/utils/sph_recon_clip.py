"""Keep Genesis's reconstructed SPH water surface on the fluid's side of rigid walls.

The rasterizer meshes an SPH liquid (``vis_mode="recon"``) with splashsurf from the particle positions alone. Its
kernel reaches about two particle radii, so water pressed against a wall thinner than that is meshed straight through
it: in task 25 a 5 mm mug wall against 15 mm particles showed water inside an empty mug, and a VLM policy acted on it.
Physics is unaffected (no particle crosses the wall); only the picture is wrong.

After each reconstruction, every surface vertex is traced back to its nearest particle. A vertex that lies in open
space while the segment to that particle passes through a rigid collision geom is water drawn on the far side of a
wall, so it and its faces are dropped. Vertices inside a geom are kept: the wall itself hides them. On a task 25 dunk
the dropped surface sat 1.8 cm (median) from any particle against 0.64 cm for the kept surface, i.e. it was never
water.

The wall test keeps anything inside a collision geom, and a collision geom can be thicker than the visual wall, so a
sliver of surface can still show at a wall's inner edge. Surface inside a geom that is real water lies within about
one particle radius of the fluid (p90 0.9 cm at radius 0.75 cm), so a vertex inside a geom and farther than
``MAX_DIST_RADII`` radii from every particle is dropped too. Open-water surface is never judged by distance: water
pressed against the outside of a wall legitimately sits farther from its particles, and dropping it cut holes in real
water. Only the rasterizer path is changed, and only for SPH entities.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

_SAMPLES = np.linspace(0.1, 0.9, 9)  # interior points along each particle -> vertex segment
_MAX_GEOM_EXTENT = 0.6  # m; larger geoms (table, floor) never separate water from where it is drawn
MAX_DIST_RADII = 1.6  # inside a wall, drop surface farther than this many particle radii from every particle

_installed = False
_planes_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
_pending: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _local_planes(geom) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Half-space form (normals, offsets, AABB corners) of a collision geom in its own frame, cached by index."""
    if geom.idx not in _planes_cache:
        planes = None
        verts = np.asarray(geom.get_trimesh().vertices, dtype=float)
        if len(verts) >= 4 and np.ptp(verts, axis=0).max() <= _MAX_GEOM_EXTENT:
            try:
                eq = ConvexHull(verts).equations
                lo, hi = verts.min(0), verts.max(0)
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                planes = (eq[:, :3], eq[:, 3], corners)
            except Exception:  # degenerate (flat) geom: nothing to hide behind
                planes = None
        _planes_cache[geom.idx] = planes
    return _planes_cache[geom.idx]


def _world_planes(
    rigid, env_idx: int, offset: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(normals, offsets, aabb_lo, aabb_hi) of every small rigid collision geom in world frame for one env."""
    geoms = [g for ent in rigid.entities for g in ent.geoms]
    local = [(g, _local_planes(g)) for g in geoms]
    local = [(g, p) for g, p in local if p is not None]
    if not local:
        return []
    idx = [g.idx for g, _ in local]
    pos = rigid.get_geoms_pos(geoms_idx=idx, relative=False)
    quat = rigid.get_geoms_quat(geoms_idx=idx, relative=False)
    if pos.ndim == 3:
        pos, quat = pos[env_idx], quat[env_idx]
    pos = pos.cpu().numpy() + offset
    rot = _quat_to_rot(quat.cpu().numpy())
    out = []
    for (_, (n, d, corners)), p, R in zip(local, pos, rot, strict=True):
        n_w = n @ R.T
        world_corners = corners @ R.T + p
        out.append((n_w, d - n_w @ p, world_corners.min(0), world_corners.max(0)))
    return out


def clip_to_rigid(mesh, positions: np.ndarray, planes, max_dist: float | None = None) -> tuple[int, int]:
    """Drop, in place, vertices (and their faces) that are not water.

    A vertex goes if it is drawn on the far side of a rigid geom from its nearest particle, or, when ``max_dist`` is
    given, if it lies inside a rigid geom and is farther than ``max_dist`` from every particle. Returns the number dropped by each rule (a vertex
    caught by both counts once, under the wall rule).
    """
    if len(mesh.vertices) == 0 or len(positions) == 0:
        return 0, 0
    verts = np.asarray(mesh.vertices)
    dist, nearest = cKDTree(positions).query(verts)
    src = positions[nearest]
    seg_lo, seg_hi = np.minimum(src, verts), np.maximum(src, verts)
    in_wall = np.zeros(len(verts), dtype=bool)
    crosses = np.zeros(len(verts), dtype=bool)
    for n, d, lo, hi in planes:
        near = np.flatnonzero(np.all(seg_hi >= lo, axis=1) & np.all(seg_lo <= hi, axis=1))
        if len(near) == 0:
            continue
        s, v = src[near], verts[near]
        in_wall[near[np.all(v @ n.T + d <= 0.0, axis=-1)]] = True
        pts = s[:, None, :] + _SAMPLES[None, :, None] * (v - s)[:, None, :]
        crosses[near[np.all(pts @ n.T + d <= 0.0, axis=-1).any(axis=1)]] = True
    behind_wall = crosses & ~in_wall
    # Only surface inside a wall is judged by distance: open-water surface pressed against a wall legitimately sits
    # farther from its particles, and dropping it cuts holes in real water.
    too_far = (dist > max_dist) & in_wall & ~behind_wall if max_dist is not None else np.zeros(len(verts), dtype=bool)
    drop = behind_wall | too_far
    if drop.any():
        mesh.update_faces(~drop[mesh.faces].any(axis=1))
        mesh.remove_unreferenced_vertices()
    return int(behind_wall.sum()), int(too_far.sum())


def install() -> None:
    """Patch Genesis's rasterizer so SPH ``recon`` surfaces are clipped against rigid geoms. Idempotent."""
    global _installed
    if _installed:
        return
    import genesis.utils.particle as pu
    import genesis.vis.rasterizer_context as rc

    orig_update_sph = rc.RasterizerContext.update_sph
    orig_particles_to_mesh = pu.particles_to_mesh

    def update_sph(self):
        # update_sph reconstructs once per (SPH entity, rendered env), in that order; queue the matching walls so
        # the patched particles_to_mesh below clips each call against the right env. Anything outside this method
        # (e.g. the ray tracer) finds the queue empty and is left untouched.
        if self.sim.sph_solver.is_active:
            offsets = np.asarray(self.scene.envs_offset)
            per_env = {i: _world_planes(self.sim.rigid_solver, i, offsets[i]) for i in self.rendered_envs_idx}
            for ent in self.sim.sph_solver.entities:
                if ent.surface.vis_mode == "recon":
                    _pending.extend(per_env[i] for i in self.rendered_envs_idx)
        try:
            orig_update_sph(self)
        finally:
            _pending.clear()

    def particles_to_mesh(positions, radius, backend):
        mesh = orig_particles_to_mesh(positions, radius, backend)
        if _pending:
            clip_to_rigid(mesh, np.asarray(positions), _pending.pop(0), max_dist=MAX_DIST_RADII * float(radius))
        return mesh

    rc.RasterizerContext.update_sph = update_sph
    pu.particles_to_mesh = particles_to_mesh  # rasterizer_context calls it as ``pu.particles_to_mesh``
    _installed = True
