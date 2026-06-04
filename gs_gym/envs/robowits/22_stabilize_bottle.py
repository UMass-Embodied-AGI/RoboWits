"""StabilizeBottle environment for RoboWits.

Task: Make the tube stand upright in the bowl using sand for support.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.scenes.schema import MPMOptionsArgs


@register_task("robowits/22-stabilize-bottle-v0")
class StabilizeBottleEnv(RoboWitsEnv):
    """Stabilize a tube using dry sand in a bowl.

    The narrow base of the tube makes it unstable. Granular packing of the
    dry sand around the base provides distributed support and friction,
    stabilizing the tube without direct clamping by the gripper.

    Success criteria:
    - Tube's base is inside the bowl's footprint
    - Tube is upright (tilt <= 5 degrees from vertical)
    - Tube is stationary (low velocity)
    - Objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Make the tube stand upright in the bowl without falling for a short time."

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
        args = args.model_copy(
            update={
                "scene_args": args.scene_args.model_copy(
                    update={
                        "substeps": 200,
                        "mpm_options": MPMOptionsArgs(
                            grid_density=64,
                            particle_size=0.002,
                            enable_CPIC=True,
                        ),
                    }
                )
            }
        )
        return args

    @property
    def placement_groups(self) -> PlacementGroups:
        """Tube and bowl placed independently."""
        return ("tube", "bowl", ("sand_container", "dry_sand"))

    def _add_custom_entities(self) -> None:
        """Add bowl, tube, sand container, and dry sand."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Bowl (mesh, fixed)
        bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=1.76,
                pos=(0.52, 0.0, 0.76 + 0.024 * 1.76 + 0.001),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["bowl"] = {
            "entity": bowl,
        }

        # Tube (mesh)
        tube = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e620c6a5-0a69-4a70-a9a7-62c91931715e/obj.glb", pattern_is_dir=False),
                scale=(2.0, 2.0, 2.2),
                pos=(0.4, 0.25, 0.8),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=50.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tube"] = {
            "entity": tube,
        }

        # Sand container (mesh)
        sand_container = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.35, -0.16, 0.8202),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, coup_friction=0.3, coup_softness=0.001),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["sand_container"] = {
            "entity": sand_container,
        }

        # Dry sand (MPM)
        dry_sand = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.025,
                height=0.22,
                pos=(0.35, -0.14, 0.905),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.Sand(rho=500.0, sampler="pbs", friction_angle=30),
            surface=gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle"),
        )
        self._entities["dry_sand"] = {
            "entity": dry_sand,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: tube standing upright in bowl."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69
        TILT_DEG_MAX = 5.0
        SPEED_MAX = 0.01
        MARGIN = 1e-3

        tube = objs_info.get("tube")
        bowl = objs_info.get("bowl")

        if tube is None or bowl is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_on_table(obj):
            hull = obj.get("convex_hull_2d")
            if hull is not None:
                xs = [float(p[0]) for p in hull]
                ys = [float(p[1]) for p in hull]
                return (
                    min(xs) >= TABLE_X_MIN - MARGIN
                    and max(xs) <= TABLE_X_MAX + MARGIN
                    and min(ys) >= TABLE_Y_MIN - MARGIN
                    and max(ys) <= TABLE_Y_MAX + MARGIN
                )
            bounds = obj.get("bounds")
            if bounds is not None:
                x_min, y_min = float(bounds[0][0]), float(bounds[0][1])
                x_max, y_max = float(bounds[1][0]), float(bounds[1][1])
                return (
                    x_min >= TABLE_X_MIN - MARGIN
                    and x_max <= TABLE_X_MAX + MARGIN
                    and y_min >= TABLE_Y_MIN - MARGIN
                    and y_max <= TABLE_Y_MAX + MARGIN
                )
            return False

        if not is_on_table(tube) or not is_on_table(bowl):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        def point_in_polygon(pt, poly):
            if poly is None or len(poly) < 3:
                return False
            x, y = float(pt[0]), float(pt[1])
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = float(poly[i][0]), float(poly[i][1])
                x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
                if (y1 > y) != (y2 > y):
                    denom = y2 - y1
                    if abs(denom) < 1e-12:
                        continue
                    xinters = (x2 - x1) * (y - y1) / denom + x1
                    if x < xinters:
                        inside = not inside
            return inside

        bowl_hull = bowl.get("convex_hull_2d")
        tube_pos = tube.get("pos")
        if tube_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        if bowl_hull is not None and len(bowl_hull) >= 3:
            if not point_in_polygon((float(tube_pos[0]), float(tube_pos[1])), bowl_hull):
                return torch.tensor([False], dtype=torch.bool, device=self._device)
        else:
            bowl_bounds = bowl.get("bounds")
            if bowl_bounds is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            x_min, y_min = float(bowl_bounds[0][0]), float(bowl_bounds[0][1])
            x_max, y_max = float(bowl_bounds[1][0]), float(bowl_bounds[1][1])
            if not (
                x_min + MARGIN <= float(tube_pos[0]) <= x_max - MARGIN
                and y_min + MARGIN <= float(tube_pos[1]) <= y_max - MARGIN
            ):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        euler = tube.get("euler")
        if euler is not None:
            rx = float(euler[0])
            ry = float(euler[1])
            cx, _sx = math.cos(rx), math.sin(rx)
            cy, _sy = math.cos(ry), math.sin(ry)
            up_z = cy * cx
            up_z = max(-1.0, min(1.0, up_z))
            tilt = math.degrees(math.acos(up_z))
            if tilt > TILT_DEG_MAX:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        tube_vel = tube.get("vel")
        if tube_vel is not None:
            speed = math.sqrt(float(tube_vel[0]) ** 2 + float(tube_vel[1]) ** 2 + float(tube_vel[2]) ** 2)
            if speed > SPEED_MAX:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        bowl_vel = bowl.get("vel")
        if bowl_vel is not None:
            speed = math.sqrt(float(bowl_vel[0]) ** 2 + float(bowl_vel[1]) ** 2 + float(bowl_vel[2]) ** 2)
            if speed > SPEED_MAX:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward based on milestone-based progress score."""
        objs_info = self.collect_objs_info()

        TUBE = "tube"
        BOWL = "bowl"
        SAND_CONTAINER = "sand_container"
        SAND = "dry_sand"

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.4
        TILT_DEG_MAX = 5.0

        INITIAL_CONTAINER_POS = np.array([0.35, -0.16, 0.8202])
        INITIAL_CONTAINER_EULER = np.array([0.0, 0.0, 0.0])
        BOWL_POS = np.array([0.52, 0.0, 0.76 + 0.024 * 1.76 + 0.001])

        if TUBE not in objs_info or BOWL not in objs_info:
            score_t = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        def _to_vec3(arr):
            """Convert array to 1D vec3, handling batch/particle dimensions."""
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2:
                # Batched (n_envs, 3) or particles (n_particles, 3) - take mean
                arr = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
            return arr.flatten()[:3]

        def get_pos(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            p = obj.get("pos")
            if p is None:
                return None
            return _to_vec3(p)

        def get_euler(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            e = obj.get("euler")
            if e is None:
                return None
            return _to_vec3(e)

        def get_bounds(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            b = obj.get("bounds")
            if b is None:
                return None
            b = np.asarray(b, dtype=float)
            # Handle batch dimension (n_envs, 2, 3) -> (2, 3)
            if b.ndim == 3:
                b = b[0]
            return b

        def get_vel(name):
            obj = objs_info.get(name)
            if obj is None:
                return None
            v = obj.get("vel")
            if v is None:
                return None
            return _to_vec3(v)

        def is_point_in_aabb_xy(point, bounds):
            if bounds is None or point is None:
                return False
            x, y = float(point[0]), float(point[1])
            x_min, x_max = float(bounds[0, 0]), float(bounds[1, 0])
            y_min, y_max = float(bounds[0, 1]), float(bounds[1, 1])
            return x_min <= x <= x_max and y_min <= y <= y_max

        def tilt_from_vertical_deg(euler_rad):
            """Compute tilt angle from vertical in degrees (euler in radians)."""
            if euler_rad is None:
                return 90.0
            rx = float(euler_rad[0])
            ry = float(euler_rad[1])
            cx = math.cos(rx)
            cy = math.cos(ry)
            up_z = cy * cx
            up_z = max(-1.0, min(1.0, up_z))
            return math.degrees(math.acos(up_z))

        def is_tube_upright():
            tube_euler = get_euler(TUBE)
            tilt = tilt_from_vertical_deg(tube_euler)
            tube_vel = get_vel(TUBE)
            is_stable = True
            if tube_vel is not None:
                speed = float(np.linalg.norm(tube_vel))
                is_stable = speed < 0.05
            return tilt <= TILT_DEG_MAX and is_stable

        def is_tube_in_bowl():
            tube_pos = get_pos(TUBE)
            bowl_bounds = get_bounds(BOWL)
            if tube_pos is None or bowl_bounds is None:
                return False
            return is_point_in_aabb_xy(tube_pos, bowl_bounds)

        def is_sand_in_bowl():
            sand_pos = get_pos(SAND)
            bowl_bounds = get_bounds(BOWL)
            if sand_pos is None or bowl_bounds is None:
                return False
            return is_point_in_aabb_xy(sand_pos, bowl_bounds)

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        has_sand_container = SAND_CONTAINER in objs_info and SAND in objs_info

        if has_sand_container:
            container_pos = get_pos(SAND_CONTAINER)
            container_euler = get_euler(SAND_CONTAINER)

            if container_pos is not None:
                dist_right_to_container = float(np.linalg.norm(right_ee_pos - container_pos))
                dist_left_to_container = float(np.linalg.norm(left_ee_pos - container_pos))
                min_dist_to_container = min(dist_right_to_container, dist_left_to_container)
            else:
                min_dist_to_container = float("inf")

            container_score = 0.0
            if min_dist_to_container <= TOUCH_THRESHOLD:
                container_score = 0.2
            elif min_dist_to_container <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_to_container - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                container_score = 0.1 + reach_progress * 0.1
            elif min_dist_to_container < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_to_container - REACH_THRESHOLD) / (
                    APPROACH_THRESHOLD - REACH_THRESHOLD
                )
                approach_progress = max(0.0, min(1.0, approach_progress))
                container_score = approach_progress * 0.1

            if container_pos is not None:
                height_gain = max(0.0, container_pos[2] - INITIAL_CONTAINER_POS[2])
                height_bonus = min(0.1, height_gain / 0.1 * 0.1)
                container_score += height_bonus

            if container_euler is not None:
                euler_diff = np.abs(container_euler - INITIAL_CONTAINER_EULER)
                max_rotation = float(np.max(euler_diff))
                rotation_bonus = min(0.1, max_rotation / 45.0 * 0.1)
                container_score += rotation_bonus

            container_score = min(0.4, container_score)

            sand_in_bowl = is_sand_in_bowl()

            tube_pos = get_pos(TUBE)
            if tube_pos is not None:
                dist_right_to_tube = float(np.linalg.norm(right_ee_pos - tube_pos))
                dist_left_to_tube = float(np.linalg.norm(left_ee_pos - tube_pos))
                min_dist_to_tube = min(dist_right_to_tube, dist_left_to_tube)
            else:
                min_dist_to_tube = float("inf")

            tube_in_bowl = is_tube_in_bowl()
            tube_upright = is_tube_upright()

            if tube_upright and tube_in_bowl:
                score = 1.0
            elif tube_in_bowl:
                tube_euler = get_euler(TUBE)
                tilt = tilt_from_vertical_deg(tube_euler)
                upright_progress = max(0.0, 1.0 - tilt / 45.0)
                score = 0.95 + upright_progress * 0.05
            elif sand_in_bowl:
                if min_dist_to_tube <= TOUCH_THRESHOLD:
                    tube_toward_bowl = (
                        max(0.0, 1.0 - float(np.linalg.norm(tube_pos[:2] - BOWL_POS[:2])) / 0.3)
                        if tube_pos is not None
                        else 0.0
                    )
                    score = 0.85 + tube_toward_bowl * 0.1
                elif min_dist_to_tube <= REACH_THRESHOLD:
                    reach_progress = 1.0 - (min_dist_to_tube - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                    reach_progress = max(0.0, min(1.0, reach_progress))
                    score = 0.7 + reach_progress * 0.15
                else:
                    score = 0.7
            else:
                sand_pos = get_pos(SAND)
                if sand_pos is not None:
                    sand_toward_bowl = max(0.0, 1.0 - float(np.linalg.norm(sand_pos[:2] - BOWL_POS[:2])) / 0.3)
                    score = container_score + sand_toward_bowl * (0.7 - container_score)
                else:
                    score = container_score
        else:
            tube_pos = get_pos(TUBE)
            if tube_pos is not None:
                dist_right_to_tube = float(np.linalg.norm(right_ee_pos - tube_pos))
                dist_left_to_tube = float(np.linalg.norm(left_ee_pos - tube_pos))
                min_dist_to_tube = min(dist_right_to_tube, dist_left_to_tube)
            else:
                min_dist_to_tube = float("inf")

            tube_in_bowl = is_tube_in_bowl()

            if tube_in_bowl:
                score = 0.3
            elif min_dist_to_tube <= TOUCH_THRESHOLD:
                tube_toward_bowl = (
                    max(0.0, 1.0 - float(np.linalg.norm(tube_pos[:2] - BOWL_POS[:2])) / 0.3)
                    if tube_pos is not None
                    else 0.0
                )
                score = 0.2 + tube_toward_bowl * 0.1
            elif min_dist_to_tube <= REACH_THRESHOLD:
                reach_progress = 1.0 - (min_dist_to_tube - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
                reach_progress = max(0.0, min(1.0, reach_progress))
                score = 0.1 + reach_progress * 0.1
            elif min_dist_to_tube < APPROACH_THRESHOLD:
                approach_progress = 1.0 - (min_dist_to_tube - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
                approach_progress = max(0.0, min(1.0, approach_progress))
                score = approach_progress * 0.1
            else:
                score = 0.0

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
