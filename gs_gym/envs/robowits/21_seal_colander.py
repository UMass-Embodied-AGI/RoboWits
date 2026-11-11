"""SealColander environment for RoboWits.

Task: Make the perforated container hold water for a short time.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.scenes.schema import SPHOptionsArgs


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N, 3) by the unit quaternion q = (w, x, y, z), in numpy.

    Plain numpy: Genesis makes CUDA torch's default device, so torch here would land on the GPU.
    """
    q = np.asarray(q, dtype=float).reshape(4)
    w, u = q[0], q[1:]
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float).reshape(4) * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(a, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


def _yaw_quat(yaw_deg: float) -> np.ndarray:
    h = np.deg2rad(yaw_deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


# Perforated ladle (m): a square bowl with one square hole in the middle of its floor and a handle raised 40 deg.
# Body frame: origin at the centre of the bowl's underside, z up out of the bowl, the handle along +x.
LADLE_WALL = 0.010  # wall thickness: the runny SPH water creeps into, and through, thinner walls
LADLE_INNER = 0.06  # inner width of the bowl (square)
LADLE_DEPTH = 0.03  # inner depth, floor top to rim
LADLE_HOLE = 0.020  # width across flats of the octagonal hole: an open ladle empties in ~2 s; the dome seals it
LADLE_HANDLE = (0.15, 0.015, 0.008, 40.0)  # length, width, thickness, rise (deg)


def _ladle_geoms(
    floor: float = LADLE_WALL, hole: float = LADLE_HOLE
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """(half sizes, centre, pitch deg) of each box: four floor boxes framing a square hole, four walls, the handle.

    ``_ladle_corner_prisms`` fills the square's corners, turning it into an octagon.
    """
    t, f, inner, depth = LADLE_WALL, floor, LADLE_INNER, LADLE_DEPTH
    outer = inner + 2 * t
    band = (outer - hole) / 2  # floor strip on either side of the hole
    geoms = [
        ((outer / 2, band / 2, f / 2), (0.0, +(hole / 2 + band / 2), f / 2), 0.0),
        ((outer / 2, band / 2, f / 2), (0.0, -(hole / 2 + band / 2), f / 2), 0.0),
        ((band / 2, hole / 2, f / 2), (+(hole / 2 + band / 2), 0.0, f / 2), 0.0),
        ((band / 2, hole / 2, f / 2), (-(hole / 2 + band / 2), 0.0, f / 2), 0.0),
        ((t / 2, outer / 2, depth / 2), (+(inner / 2 + t / 2), 0.0, f + depth / 2), 0.0),
        ((t / 2, outer / 2, depth / 2), (-(inner / 2 + t / 2), 0.0, f + depth / 2), 0.0),
        ((inner / 2, t / 2, depth / 2), (0.0, +(inner / 2 + t / 2), f + depth / 2), 0.0),
        ((inner / 2, t / 2, depth / 2), (0.0, -(inner / 2 + t / 2), f + depth / 2), 0.0),
    ]
    length, width, thick, rise = LADLE_HANDLE
    d = np.array([np.cos(np.deg2rad(rise)), 0.0, np.sin(np.deg2rad(rise))])
    start = np.array([outer / 2 - t, 0.0, f + depth - thick / 2])  # overlaps the rim so the joint is solid
    centre = start + d * length / 2
    geoms.append(((length / 2, width / 2, thick / 2), tuple(float(c) for c in centre), -rise))
    return geoms


def _ladle_corner_prisms(floor: float = LADLE_WALL, hole: float = LADLE_HOLE) -> list[np.ndarray]:
    """Triangular prisms (6 vertices each) that cut the square hole's corners into a regular octagon.

    A sphere cap on a square hole touches only the edge midpoints, leaving ~3-4 mm gaps at the corners that water
    creeps through unless its surface tension is high (which made it look like gel); on the octagon the gap at a
    vertex is ~0.5 mm.
    """
    r, k = hole / 2, np.tan(np.deg2rad(22.5))
    prisms = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            tri = [(sx * r, sy * r * k), (sx * r * k, sy * r), (sx * r, sy * r)]
            prisms.append(np.array([(x, y, z) for z in (0.0, floor) for x, y in tri]))
    return prisms


def _ladle_mjcf_path(fixed: bool = False, floor: float = LADLE_WALL, hole: float = LADLE_HOLE) -> str:
    """Write the ladle's MJCF: one free body of boxes and corner prisms, so every collision shape is exactly convex."""
    prisms = _ladle_corner_prisms(floor, hole)
    assets = "\n".join(
        f'    <mesh name="corner{i}" vertex="{" ".join(f"{v:.5f}" for v in pts.ravel())}"/>'
        for i, pts in enumerate(prisms)
    )
    corners = "\n".join(f'      <geom type="mesh" mesh="corner{i}" density="500"/>' for i in range(len(prisms)))
    boxes = "\n".join(
        f'      <geom type="box" size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
        f'euler="0 {pitch:.3f} 0" density="500"/>'
        for h, c, pitch in _ladle_geoms(floor, hole)
    )
    joint = "" if fixed else '      <joint type="free"/>\n'
    xml = (
        '<mujoco model="perforated_ladle">\n  <compiler angle="degree"/>\n'
        f"  <asset>\n{assets}\n  </asset>\n  <worldbody>\n"
        f'    <body name="ladle">\n{joint}{boxes}\n{corners}\n    </body>\n'
        "  </worldbody>\n</mujoco>\n"
    )
    path = Path(tempfile.gettempdir()) / f"gs_gym_ladle_{hashlib.sha1(xml.encode()).hexdigest()[:12]}.xml"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(xml)
        tmp.replace(path)
    return str(path)


DOME_RADIUS = 0.02  # the curved holder: a hemisphere, curved side down


def _dome_mesh_path(radius: float = DOME_RADIUS) -> str:
    """Write a hemisphere mesh (flat face at z=0, curved side toward -z). Convex, so its collision shape is exact."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius).vertices
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    rim = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], 1)
    dome = trimesh.convex.convex_hull(np.concatenate([sphere[sphere[:, 2] <= 0.0], rim]))
    path = Path(tempfile.gettempdir()) / f"gs_gym_dome_{radius:.4f}.obj"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp.obj")
        dome.export(tmp)
        tmp.replace(path)
    return str(path)


@register_task("robowits/21-seal-colander-v0")
class SealColanderEnv(RoboWitsEnv):
    """Make a perforated ladle hold water.

    The ladle (``perforated_container``) has one octagonal hole, 2.0 cm across, in the middle of its floor,
    so anything it scoops drains. The dome (``curved_holder``, a 4 cm hemisphere, curved side
    down) is wider than the hole: set in it, its curved underside seals it, and the sealed ladle
    can scoop water from the pitcher and hold it. A dome, not a ball: it does not roll, and a
    knocked-over dome rocks back flat side up. The ladle is built from boxes (an MJCF written at build time) so its
    collision shapes are exact; the toolbench slotted spoon it replaces was a thin open shell
    that neither held SPH water nor caught the ball, and its slot sat too close to its edge for
    the ball to cover.

    The pitcher, its invisible containment cage and its water are task 25's (verified to keep
    the water in, since the pitcher mesh alone leaks SPH), with the cage's spout gap closed --
    nothing should overflow here -- and finer SPH particles to resolve the hole. The pitcher is
    fixed: full of water it is too heavy to lift, so the water is scooped.

    Success criteria, held for ``HOLD_STEPS`` consecutive steps (~1 s at 30 Hz):
    - at least ``MIN_HELD_ML`` of water inside the ladle's bowl, never more than ``MAX_HOLD_LOSS`` below the amount
      the hold started with (an open hole keeps draining; a sealed one does not)
    - the ladle is lifted off the table (the table under the hole would seal it)
    - the bowl is clear of the pitcher -- off to the side, or held above its rim (water around a ladle still in
      the pitcher does not count)
    """

    # SPH liquid: far faster on the GPU backend.
    preferred_genesis_backend = "cuda"

    TABLE_Z = 0.76
    #: SPH particle size (m). Task 25's default 0.015 is about the hole's width; at 6 mm a sealed ladle keeps ~99%
    #: of its water over 2 s while an open one loses ~28% (still 10-20x slower than real water drains).
    WATER_PARTICLE_SIZE = 0.006
    #: Task 25's water but runnier: its gamma 0.02 / mu 0.01 beaded into a gel-like lump. Low surface tension needs the
    #: octagonal hole to seal (the square one leaked past the dome), and mu 0.002 is also within the SPH stable step.
    WATER_SPH = {"rho": 100.0, "stiffness": 5000.0, "exponent": 7.0, "mu": 0.002, "gamma": 0.005}
    #: Pitcher centre on the table (task 25's pitcher is at (0.50, -0.03); the cage follows it).
    PITCHER_XY = (0.61, 0.0)
    #: Ladle floor thickness (m). A variant whose particles couple through a grid (MPM) needs it thicker than a cell.
    #: The runny water (mu 0.002) sank into a 4 mm floor and some passed right through it; 1 cm stops that.
    LADLE_FLOOR = 0.010
    #: Width across flats of the ladle's octagonal hole (m).
    LADLE_HOLE_SIDE = LADLE_HOLE
    LADLE_LIFT_MIN = 0.02  # ladle AABB bottom must clear the table by this much (m)
    PITCHER_CLEAR_R = 0.14  # bowl centre's horizontal distance from the pitcher axis that counts as clear (m)
    MIN_HELD_ML = 10.0
    HOLD_STEPS = 30
    #: The water may not drop more than this fraction below the hold's starting amount, or the hold restarts. SPH
    #: drains an open hole too slowly for MIN_HELD_ML alone to tell a leaking ladle from a sealed one.
    MAX_HOLD_LOSS = 0.05

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Make the container hold water for a short time."

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        self._hold_count = 0
        self._hold_step = -1
        self._hold_ref_ml = 0.0
        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

    def _config_to_env_args(self, config):
        args = super()._config_to_env_args(config)
        scene_args = args.scene_args.model_copy(
            update={"substeps": 100, "sph_options": SPHOptionsArgs(particle_size=self.WATER_PARTICLE_SIZE)}
        )
        return args.model_copy(update={"scene_args": scene_args})

    @property
    def placement_groups(self) -> PlacementGroups:
        """Pitcher and water are fixed; only the ladle and the dome are randomized."""
        return ("perforated_container", "curved_holder")

    def _add_pitcher_with_water(self, px: float, py: float) -> None:
        """Task 25's fixed pitcher, invisible cage and water, centred on (px, py).

        Offsets are task 25's (pitcher at (0.50, -0.03)), except the cage box at 90 deg: task 25
        lowers it to open the spout for its overflow; here it is full height like the rest.
        """
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )
        pitcher = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pitcher.glb", pattern_is_dir=False),
                scale=1.525,
                pos=(px, py, 0.9515 - 0.08),
                euler=(0.0, 0.0, -90.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=800.0, sdf_min_res=128, sdf_max_res=256),
            surface=gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True, vis_mode="visual"),
        )
        self._entities["pitcher"] = {"entity": pitcher}
        self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(px, py + 0.02, 0.82), radius=0.11, height=0.05, fixed=True, visualization=False, collision=True
            ),
        )
        for rot in range(0, 360, 30):
            sin_rot, cos_rot = np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))
            self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=(px + 0.09 * cos_rot, py + 0.015 + 0.09 * sin_rot, 0.97 - 0.08),
                    euler=(0.0, 0.0, float(rot)),
                    size=(0.01, 0.05, 0.15),
                    fixed=True,
                    visualization=False,
                    collision=True,
                ),
            )
        water = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(px, py + 0.01, 0.8715 + 0.2 - 0.08),
                euler=(0.0, 0.0, 0.0),
                radius=0.06,
                height=0.3,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(**self.WATER_SPH, sampler="pbs"),
            # Nyx has no refractive glass (its exporter keeps colour, roughness and opacity), so the water is a glossy,
            # partly transparent blue rather than task 25's opaque pale Glass.
            surface=gs.surfaces.Glass(
                color=(0.25, 0.55, 0.85), opacity=0.55, roughness=0.05, double_sided=True, vis_mode="recon"
            ),
        )
        self._entities["water"] = {"entity": water}

    def _add_ladle(self, xy: tuple[float, float], yaw_deg: float = 0.0, fixed: bool = False) -> None:
        """Add the perforated ladle resting on the table at ``xy``, handle along its +x turned by ``yaw_deg``."""
        pos = (float(xy[0]), float(xy[1]), self.TABLE_Z + 0.001)
        ladle = self._scene.scene.add_entity(
            gs.morphs.MJCF(
                file=_ladle_mjcf_path(fixed, self.LADLE_FLOOR, self.LADLE_HOLE_SIDE),
                pos=pos,
                euler=(0.0, 0.0, float(yaw_deg)),
            ),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(color=(0.15, 0.4, 0.85), double_sided=True),  # blue plastic
        )
        self._entities["perforated_container"] = {"entity": ladle, "authored_pose": (np.array(pos), _yaw_quat(yaw_deg))}

    def _add_curved_holder(self, xy: tuple[float, float]) -> None:
        """Add the dome resting curved side down on the table at ``xy``."""
        dome = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                file=_dome_mesh_path(),
                pos=(float(xy[0]), float(xy[1]), self.TABLE_Z + DOME_RADIUS),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Smooth(color=(0.85, 0.15, 0.12), double_sided=True),  # red rubber
        )
        self._entities["curved_holder"] = {"entity": dome}

    def _add_custom_entities(self) -> None:
        """Add the perforated ladle, the dome, and the pitcher with its water."""
        self._add_ladle((0.42, -0.16), yaw_deg=180.0)

        self._add_curved_holder((0.52, -0.25))

        self._add_pitcher_with_water(*self.PITCHER_XY)

    # ------------------------------------------------------------------ success

    def reset_env(self) -> None:
        super().reset_env()
        self._hold_count = 0
        self._hold_step = -1
        self._hold_ref_ml = 0.0

    def _ladle_pose(self, ladle: dict) -> tuple[np.ndarray, np.ndarray]:
        """The ladle's pose in its authored frame (bowl underside centre, z up, handle along +x).

        Genesis places a free MJCF body's link at its centre of mass with principal axes -- for this ladle 3 cm up
        and tilted ~34 deg by the raised handle. The link-to-authored offset is constant, so it is read once from
        the build-time link pose (``initial_pos``/``initial_quat``) and the pose the ladle was authored at.
        """
        info = self._entities["perforated_container"]
        if "link_to_authored" not in info:
            p0, q0 = np.asarray(info["initial_pos"], dtype=float), np.asarray(info["initial_quat"], dtype=float)
            pa, qa = info["authored_pose"]
            info["link_to_authored"] = (_quat_rotate(_quat_conj(q0), pa - p0)[0], _quat_mul(_quat_conj(q0), qa))
        c_p, c_q = info["link_to_authored"]
        p, q = np.asarray(ladle["pos"], dtype=float), np.asarray(ladle["quat"], dtype=float)
        return p + _quat_rotate(q, c_p)[0], _quat_mul(q, c_q)

    def _bowl_body(self) -> tuple[np.ndarray, np.ndarray]:
        """The bowl's inside in the ladle's body frame; the top allows 1 cm of water above the rim.

        The box reaches 3 mm into the floor and walls: SPH water resting against them settles a fraction of a
        millimetre in, and that water is still held. Water draining out of the hole falls well below this.
        """
        f, tol = self.LADLE_FLOOR, 0.003
        return (
            np.array([-LADLE_INNER / 2 - tol, -LADLE_INNER / 2 - tol, f - tol]),
            np.array([LADLE_INNER / 2 + tol, LADLE_INNER / 2 + tol, f + LADLE_DEPTH + 0.01]),
        )

    def _to_ladle_body(self, pts: np.ndarray, ladle: dict) -> np.ndarray:
        pos, quat = self._ladle_pose(ladle)
        return _quat_rotate(_quat_conj(quat), np.asarray(pts, dtype=float) - pos)

    def _held_state(self, objs_info: dict) -> dict:
        """Water in the bowl (ml), whether the ladle is lifted, and whether its bowl is clear of the pitcher."""
        ladle, water = objs_info.get("perforated_container"), objs_info.get("water")
        out = {"held_ml": 0.0, "lifted": False, "clear": False}
        if ladle is None or water is None or water.get("material") != "particle":
            return out
        wpos = np.asarray(water.get("pos"))
        if wpos.ndim != 2 or len(wpos) == 0:
            return out
        lo, hi = self._bowl_body()
        body = self._to_ladle_body(wpos, ladle)
        n_in = int(np.all((body >= lo) & (body <= hi), axis=1).sum())
        out["held_ml"] = n_in * 0.8 * self.WATER_PARTICLE_SIZE**3 * 1e6
        out["lifted"] = float(np.asarray(ladle["bounds"])[0][2]) >= self.TABLE_Z + self.LADLE_LIFT_MIN
        pos, quat = self._ladle_pose(ladle)
        head_c = pos + _quat_rotate(quat, (lo + hi) / 2)[0]
        pitcher = objs_info.get("pitcher")
        axis = np.asarray(pitcher["pos"][:2] if pitcher is not None else self.PITCHER_XY, dtype=float)
        aside = float(np.linalg.norm(head_c[:2] - axis)) >= self.PITCHER_CLEAR_R
        # Or lifted out over it: the bowl's whole underside above the rim (a full ladle held over the pitcher is a
        # natural place to hold it; astra did, and the horizontal test alone never counted that)
        above = False
        if pitcher is not None and pitcher.get("bounds") is not None:
            half = LADLE_INNER / 2 + LADLE_WALL
            under = np.array([[x, y, 0.0] for x in (-half, half) for y in (-half, half)])
            bottom_z = float((pos + _quat_rotate(quat, under))[:, 2].min())
            above = bottom_z >= float(np.asarray(pitcher["bounds"])[1][2])
        out["clear"] = aside or above
        return out

    def _check_success(self) -> torch.Tensor:
        """Water held, without draining, in the lifted ladle clear of the pitcher, for HOLD_STEPS consecutive steps.

        The counter advances once per env step however often this is called (get_reward calls it too).
        """
        if self._hold_step != self._step_count:
            self._hold_step = self._step_count
            s = self._held_state(self.collect_objs_info())
            holding = s["lifted"] and s["clear"] and s["held_ml"] >= self.MIN_HELD_ML
            if not holding:
                self._hold_count = 0
            elif self._hold_count == 0 or s["held_ml"] < (1.0 - self.MAX_HOLD_LOSS) * self._hold_ref_ml:
                self._hold_count, self._hold_ref_ml = 1, s["held_ml"]  # start (or restart) the hold from here
            else:
                self._hold_count += 1
        return torch.tensor([self._hold_count >= self.HOLD_STEPS], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """1.0 on success; else gripper reach (<=0.2), dome seated in the hole (0.4), held water (0.4 -> 0.9)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}
        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        objs_info = self.collect_objs_info()
        ladle, dome = objs_info.get("perforated_container"), objs_info.get("curved_holder")
        if ladle is None:
            return zero, {"success": zero}

        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        ref = np.asarray(ladle["pos"][:3], dtype=float)
        d = min(
            float(np.linalg.norm(self._robot.right_ee_pose[0, :3].cpu().numpy() - ref)),
            float(np.linalg.norm(self._robot.left_ee_pose[0, :3].cpu().numpy() - ref)),
        )
        if d <= TOUCH:
            score = 0.2
        elif d <= REACH:
            score = 0.1 + 0.1 * (REACH - d) / (REACH - TOUCH)
        elif d < APPROACH:
            score = 0.1 * (APPROACH - d) / (APPROACH - REACH)
        else:
            score = 0.0

        if dome is not None:
            b = self._to_ladle_body(np.asarray(dome["pos"], dtype=float)[None], ladle)[0]
            # Seated: over the hole, resting on its edges. Genesis reports a free body at its centre of mass, which for
            # the hemisphere is 3R/8 below its flat face; the face sits ~1.8 cm above the floor top.
            hole = self.LADLE_HOLE_SIDE
            seat = self.LADLE_FLOOR + np.sqrt(DOME_RADIUS**2 - (hole / 2) ** 2) - 3.0 * DOME_RADIUS / 8.0
            if np.hypot(b[0], b[1]) < hole / 2 and b[2] < seat + 0.003:
                score = max(score, 0.4)
        s = self._held_state(objs_info)
        if s["lifted"] and s["clear"] and s["held_ml"] > 0:
            score = max(score, 0.4 + 0.5 * min(1.0, s["held_ml"] / self.MIN_HELD_ML))
        return torch.tensor([score], dtype=torch.float32, device=self._device), {"success": zero}
