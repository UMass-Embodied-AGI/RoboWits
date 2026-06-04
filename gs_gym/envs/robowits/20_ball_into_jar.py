"""BallIntoJar environment for RoboWits.

Task: Put the foam ball fully into the jar.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.placement import parse_reachable_bounds
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/20-ball-into-jar-v0")
class BallIntoJarEnv(RoboWitsEnv):
    """Put the foam ball fully into the jar.

    The foam ball is larger than the glass jar mouth when undeformed.
    The task leverages the elasticity and compressibility of the foam
    to change its shape and pass through the restriction.

    Success criteria:
    - At least 90% of foam ball particles lie inside the glass jar's axis-aligned
      bounding box (rigid fallback: the ball's AABB is fully contained in the jar's)
    - Objects haven't fallen off the table
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Put the foam ball fully into the jar."

    TABLE_Z = 0.76

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
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
        args = args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """Jar/ball rigid placement (if applicable); foam MPM cloud gets XY jitter in ``_place_objects``."""
        return ("glass jar", "foam ball")

    def _add_custom_entities(self) -> None:
        """Add glass jar and foam ball."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Glass jar (mesh, fixed)
        jar = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/f4777d21-c966-40dd-872c-8bf28e00d3ee/obj.glb", pattern_is_dir=False),
                scale=0.8,
                pos=(0.64, 0.0, 0.8350),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=250.0, friction=0.5, coup_friction=0.15),
            surface=gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        )
        self._entities["glass jar"] = {
            "entity": jar,
        }

        # Foam ball (MPM elastic)
        foam_ball = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.46, -0.10, 0.8420),
                radius=0.06,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Elastic(E=5e4, nu=0.2, rho=200.0, sampler="pbs"),
            surface=gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        )
        self._entities["foam ball"] = {
            "entity": foam_ball,
        }

    def _capture_initial_entity_poses(self) -> None:
        super()._capture_initial_entity_poses()
        info = self._entities.get("foam ball")
        if info is None:
            return
        entity = info["entity"]
        if not hasattr(entity, "get_particles_pos"):
            return
        raw = entity.get_particles_pos().detach().cpu().numpy()
        if raw.ndim == 3:
            template = np.asarray(raw[0], dtype=np.float64, order="C")
        else:
            template = np.asarray(raw, dtype=np.float64, order="C")
        if template.size == 0 or template.shape[-1] != 3:
            return
        info["initial_particles_pos"] = template

    def _place_objects(self) -> None:
        super()._place_objects()
        if self._eval_particles.get("foam ball") is not None:
            return
        info = self._entities.get("foam ball")
        if info is None:
            return
        template = info.get("initial_particles_pos")
        if template is None:
            return
        entity = info["entity"]
        if not hasattr(entity, "set_particles_pos"):
            return

        default_bounds = parse_reachable_bounds(self._config.get("reachable_bounds", self.DEFAULT_REACHABLE_BOUNDS))
        (x0, x1), (y0, y1) = default_bounds
        xy_half = 0.5 * float(
            max(
                float(np.ptp(template[:, 0])),
                float(np.ptp(template[:, 1])),
            )
        )
        margin = 0.02
        r = xy_half + margin
        cx_min, cx_max = x0 + r, x1 - r
        cy_min, cy_max = y0 + r, y1 - r
        if cx_max < cx_min or cy_max < cy_min:
            return

        centroid = template.mean(axis=0)
        new_cx = float(self._random.uniform(cx_min, cx_max))
        new_cy = float(self._random.uniform(cy_min, cy_max))
        dx, dy = new_cx - float(centroid[0]), new_cy - float(centroid[1])

        shifted = template + np.array([dx, dy, 0.0], dtype=np.float64)
        poss = torch.as_tensor(shifted, dtype=torch.float32, device=self._device)
        entity.set_particles_pos(poss)
        if hasattr(entity, "set_particles_vel"):
            z = torch.zeros_like(poss)
            entity.set_particles_vel(z)

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: foam ball mostly inside the jar's AABB."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        Z_FLOOR_THRESH = self.TABLE_Z - 0.15
        PAD = 0.02

        jar = objs_info.get("glass jar")
        ball = objs_info.get("foam ball")

        if jar is None or ball is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_not_fallen(obj):
            x_min, x_max = TABLE_X_MIN - PAD, TABLE_X_MAX + PAD
            y_min, y_max = TABLE_Y_MIN - PAD, TABLE_Y_MAX + PAD

            mat = obj.get("material", "rigid")
            if mat == "rigid":
                hull = obj.get("convex_hull_2d")
                if hull is not None and np.size(hull) > 0:
                    xs, ys = hull[:, 0], hull[:, 1]
                    within_xy = (
                        np.all(xs >= x_min) and np.all(xs <= x_max) and np.all(ys >= y_min) and np.all(ys <= y_max)
                    )
                else:
                    b = obj.get("bounds")
                    if b is None:
                        p = obj.get("pos")
                        p = np.array([np.inf, np.inf, np.inf]) if p is None else p
                        within_xy = (x_min <= p[0] <= x_max) and (y_min <= p[1] <= y_max)
                    else:
                        within_xy = b[0][0] >= x_min and b[1][0] <= x_max and b[0][1] >= y_min and b[1][1] <= y_max
                b = obj.get("bounds")
                if b is not None:
                    min_z = float(b[0][2])
                else:
                    p = obj.get("pos")
                    p = np.array([0.0, 0.0, 10.0]) if p is None else p
                    min_z = float(p[2])
                return within_xy and (min_z > Z_FLOOR_THRESH)
            else:
                pos = obj.get("pos")
                if pos is None or np.size(pos) == 0:
                    return False
                pos = np.asarray(pos)
                xy = pos[:, :2]
                xs, ys = xy[:, 0], xy[:, 1]
                within = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
                frac_within = np.mean(within) if within.size > 0 else 0.0
                min_z = float(np.min(pos[:, 2])) if pos.size > 0 else 10.0
                return (frac_within >= 0.8) and (min_z > Z_FLOOR_THRESH)

        if not (is_not_fallen(jar) and is_not_fallen(ball)):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        jar_bounds = jar.get("bounds")
        if jar_bounds is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        jb = np.asarray(jar_bounds, dtype=float).reshape(2, 3)
        j_lo, j_hi = jb[0], jb[1]

        ball_mat = ball.get("material", "rigid")
        if ball_mat == "particle":
            ball_pos = ball.get("pos")
            if ball_pos is None or np.size(ball_pos) == 0:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            ball_pos = np.asarray(ball_pos, dtype=float)
            inside = (
                (ball_pos[:, 0] >= j_lo[0])
                & (ball_pos[:, 0] <= j_hi[0])
                & (ball_pos[:, 1] >= j_lo[1])
                & (ball_pos[:, 1] <= j_hi[1])
                & (ball_pos[:, 2] >= j_lo[2])
                & (ball_pos[:, 2] <= j_hi[2])
            )
            frac_inside = float(np.mean(inside))
            return torch.tensor([frac_inside >= 0.90], dtype=torch.bool, device=self._device)

        b = ball.get("bounds")
        if b is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        bb = np.asarray(b, dtype=float).reshape(2, 3)
        b_lo, b_hi = bb[0], bb[1]
        ball_fully_inside = bool(np.all(b_lo >= j_lo) and np.all(b_hi <= j_hi))
        return torch.tensor([ball_fully_inside], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else milestone progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info()
        jar = objs_info.get("glass jar")
        ball = objs_info.get("foam ball")

        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if jar is None or ball is None:
            return zero, {"success": zero}

        def get_centroid(obj):
            p = obj.get("pos")
            if p is None:
                return None
            p = np.asarray(p, dtype=float)
            return p.mean(axis=0) if p.ndim > 1 else p

        ball_centroid = get_centroid(ball)
        jar_centroid = get_centroid(jar)

        # EE proximity to ball centroid (stage 1: 0 → 0.2)
        APPROACH, REACH, TOUCH = 0.3, 0.15, 0.05
        right_ee = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee = self._robot.left_ee_pose[0, :3].cpu().numpy()
        if ball_centroid is not None:
            ref = ball_centroid[:3]
            min_dist = min(float(np.linalg.norm(right_ee - ref)), float(np.linalg.norm(left_ee - ref)))
            if min_dist <= TOUCH:
                ee_score = 0.2
            elif min_dist <= REACH:
                ee_score = 0.1 + 0.1 * (REACH - min_dist) / (REACH - TOUCH)
            elif min_dist < APPROACH:
                ee_score = 0.1 * (APPROACH - min_dist) / (APPROACH - REACH)
            else:
                ee_score = 0.0
        else:
            ee_score = 0.0

        # Ball XY distance to jar center (stage 2: 0.2 → 0.6)
        if ball_centroid is not None and jar_centroid is not None:
            dist_xy = float(np.linalg.norm(ball_centroid[:2] - jar_centroid[:2]))
            dist_frac = max(0.0, min(1.0, 1.0 - dist_xy / 0.3))
            dist_score = 0.2 + 0.4 * dist_frac
        else:
            dist_score = 0.0

        # Fraction of ball particles inside jar AABB (stage 3: 0.6 → 1.0)
        jar_bounds = jar.get("bounds")
        inside_score = 0.0
        if jar_bounds is not None:
            jb = np.asarray(jar_bounds, dtype=float).reshape(2, 3)
            j_lo, j_hi = jb[0], jb[1]
            ball_mat = ball.get("material", "rigid")
            if ball_mat == "particle":
                pos_arr = ball.get("pos")
                if pos_arr is not None and np.size(pos_arr) > 0:
                    pos_arr = np.asarray(pos_arr, dtype=float)
                    inside = (
                        (pos_arr[:, 0] >= j_lo[0])
                        & (pos_arr[:, 0] <= j_hi[0])
                        & (pos_arr[:, 1] >= j_lo[1])
                        & (pos_arr[:, 1] <= j_hi[1])
                        & (pos_arr[:, 2] >= j_lo[2])
                        & (pos_arr[:, 2] <= j_hi[2])
                    )
                    frac_inside = float(np.mean(inside))
                    if frac_inside > 1e-3:
                        inside_score = 0.6 + 0.4 * min(1.0, frac_inside / 0.90)

        score = max(ee_score, dist_score, inside_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
