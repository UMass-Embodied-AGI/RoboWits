"""StandPages environment for RoboWits.

Task: Assemble two pages using a stabilizing bar to create a standing hinge structure.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/07-stand-pages-v0")
class StandPagesEnv(RoboWitsEnv):
    """Assemble two pages using a stabilizing bar to create a standing hinge structure.

    Individually, the pages lack the base width to overcome tipping forces.
    The robot must transform these 2D objects into a stable 3D structure by
    utilizing a pin-joint mechanism. By inserting the bar through the holes,
    the robot creates a mechanical constraint that links the two objects.

    Success criteria:
    - Both pages are standing upright (rotation about x or y close to 90 degrees)
    - Both pages are on the table surface
    - Both pages are within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Assemble the two pages using the stabilizing bar to create a standing hinge structure."

    TABLE_Z = 0.76
    ANGLE_TOL_DEG = 15.0
    Z_CONTACT_TOL = 0.03

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

    @property
    def placement_groups(self) -> PlacementGroups:
        """Pages and bar grouped together."""
        return (("page A with eyelets", "page B with eyelets", "stabilizing bar"),)

    def _add_custom_entities(self) -> None:
        """Add pages and stabilizing bar."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Page A with eyelets (mesh)
        page_a = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pageA.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.4109999930848301, -0.23420582942490897, 0.7712195122934971),
                euler=(-2.8615428539401147, -1.4288504244225462, -0.4266647550127507),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page A with eyelets"] = {
            "entity": page_a,
        }

        # Page B with eyelets (mesh)
        page_b = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/pageB.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.4302705964101475, -0.0851732580583382, 0.7722011664765699),
                euler=(-0.35271044915980393, -1.4069400650183481, 0.18154524081515175),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        )
        self._entities["page B with eyelets"] = {
            "entity": page_b,
        }

        # Stabilizing bar (box approximation of cylinder)
        bar = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.43074102707827927, -0.16217637726016496, 0.7733424749166471),
                euler=(-0.34595464718138697, -0.07688387659809946, 3.0052565839351866),
                size=(0.2, 0.015, 0.015),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.2, ior=1.5),
        )
        self._entities["stabilizing bar"] = {
            "entity": bar,
        }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: both pages standing upright."""
        objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        def angle_close_to_right_angle(angle_deg: float) -> bool:
            a = angle_deg
            if a < 0:
                a += 360.0
            if a > 360:
                a -= 360.0
            d = min(abs(a), abs(180 - a), abs(360 - a))
            return d <= self.ANGLE_TOL_DEG

        def is_upright(euler) -> bool:
            if euler is None:
                return False
            ex = float(euler[0]) * 180.0 / np.pi
            ey = float(euler[1]) * 180.0 / np.pi
            return angle_close_to_right_angle(ex) and angle_close_to_right_angle(ey)

        def hull_within_table(hull2d) -> bool:
            if hull2d is None:
                return False
            for pt in hull2d:
                x, y = float(pt[0]), float(pt[1])
                if not (TABLE_X_MIN <= x <= TABLE_X_MAX and TABLE_Y_MIN <= y <= TABLE_Y_MAX):
                    return False
            return True

        def is_on_table(bounds) -> bool:
            if bounds is None:
                return False
            z_min = float(bounds[0][2])
            return abs(z_min - self.TABLE_Z) <= self.Z_CONTACT_TOL

        required_pages = ["page A with eyelets", "page B with eyelets"]

        for name in required_pages:
            info = objs_info.get(name)
            if info is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            euler = info.get("euler")
            if not is_upright(euler):
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            hull = info.get("convex_hull_2d")
            if not hull_within_table(hull):
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            bounds = info.get("bounds")
            if not is_on_table(bounds):
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            if _utils.check_being_held(info, self._robot):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dense reward: 1.0 on success, else tilt/EE progress in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": torch.ones(1, dtype=torch.float32, device=self._device)}

        objs_info = self.collect_objs_info()

        REACH_THRESHOLD = 0.15
        TOUCH_THRESHOLD = 0.05
        APPROACH_THRESHOLD = 0.3

        PAGE_A = "page A with eyelets"
        PAGE_B = "page B with eyelets"
        BAR = "stabilizing bar"
        ASSEMBLY = [PAGE_A, PAGE_B, BAR]

        def dist_to_right_angle(angle_deg: float) -> float:
            """Degrees away from the nearest upright target (0°/180°/360°), in [0, 90]."""
            a = angle_deg
            if a < 0:
                a += 360.0
            if a > 360:
                a -= 360.0
            return min(abs(a), abs(180.0 - a), abs(360.0 - a))

        def page_tilt_progress(name: str) -> float:
            """How close this page is to upright and self-standing: 0.0 = flat or held, 1.0 = at success boundary."""
            obj = objs_info.get(name)
            if obj is None:
                return 0.0
            if _utils.check_being_held(obj, self._robot):
                return 0.0
            euler = obj.get("euler")
            if euler is None:
                return 0.0
            ex = float(euler[0]) * 180.0 / math.pi
            ey = float(euler[1]) * 180.0 / math.pi
            # Distance from the nearest right angle for both axes; worst axis dominates.
            d = max(dist_to_right_angle(ex), dist_to_right_angle(ey))
            # Normalize: d=90 → 0.0, d=ANGLE_TOL_DEG → 1.0
            return max(0.0, min(1.0, (90.0 - d) / (90.0 - self.ANGLE_TOL_DEG)))

        tilt_score = max(page_tilt_progress(PAGE_A), page_tilt_progress(PAGE_B))

        right_ee_pos = self._robot.right_ee_pose[0, :3].cpu().numpy()
        left_ee_pos = self._robot.left_ee_pose[0, :3].cpu().numpy()

        max_dist = float("inf")
        if all(objs_info.get(n) is not None for n in ASSEMBLY):
            max_dist = 0.0
            for name in ASSEMBLY:
                pos = objs_info[name].get("pos")
                if pos is not None:
                    p = np.array(pos[:3], dtype=float)
                    max_dist = max(
                        max_dist,
                        min(
                            float(np.linalg.norm(right_ee_pos - p)),
                            float(np.linalg.norm(left_ee_pos - p)),
                        ),
                    )

        if max_dist <= TOUCH_THRESHOLD:
            ee_score = 0.3
        elif max_dist <= REACH_THRESHOLD:
            t = 1.0 - (max_dist - TOUCH_THRESHOLD) / (REACH_THRESHOLD - TOUCH_THRESHOLD)
            ee_score = 0.1 + 0.2 * t
        elif max_dist < APPROACH_THRESHOLD:
            t = 1.0 - (max_dist - REACH_THRESHOLD) / (APPROACH_THRESHOLD - REACH_THRESHOLD)
            ee_score = 0.1 * t
        else:
            ee_score = 0.0

        score = max(tilt_score, ee_score)
        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
