"""StackBowls environment for RoboWits.

Task: Nest the smaller bowl concentrically inside the larger one.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv
from gs_gym.envs.robowits.utils import (
    STAGE_PROGRESS,
    approach_score,
    at_goal_score,
    check_released,
    check_still,
    check_upright,
    progress_frac,
)


@register_task("robowits/19-stack-bowls-v0")
class StackBowlsEnv(RoboWitsEnv):
    """Nest the smaller bowl concentrically inside the larger one.

    Concentric alignment leverages circular geometry; matching rims
    provide a strong geometric signal for proper nesting.

    Success criteria:
    - Medium bowl's 2D center is within 1cm of large bowl's center
    - Medium bowl is inside the large one: its bottom above the large bowl's bottom (by NEST_MIN_LIFT, so not
      side by side on the table or with the large bowl turned over it) and below the large bowl's rim
    - Both bowls upright (within UPRIGHT_MAX_TILT_DEG)
    - Medium bowl released (utils.check_released) and still (utils.check_still)
    - Both bowls remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Nest the smaller bowl concentrically inside the larger one."

    TABLE_Z = 0.76
    # Nested, the medium bowl's bottom rests ~14-16 mm above the large bowl's bottom (the large bowl's floor alone
    # is 10-13 mm thick across the variants' scales); side by side on the table the two bottoms are level.
    NEST_MIN_LIFT = 0.005
    UPRIGHT_MAX_TILT_DEG = 15.0
    CONCENTRIC_TOL = 0.011  # 1 cm, plus 1 mm numerical slack
    TABLE_XY = (0.21, 1.00, -0.69, 0.69)  # x_min, x_max, y_min, y_max
    TABLE_TOL = 0.01

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int = 200,
        control_mode: str = "EE_ABS",
        **kwargs,
    ):
        # Initialize per-object reachable areas
        self._object_reachable_areas: dict[str, dict[str, float]] = {}

        super().__init__(
            config_name=config_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_mode=control_mode,
            **kwargs,
        )

        # Set reachable areas for objects
        smaller_area = {"x_min": 0.1, "x_max": 0.7, "y_min": -0.45, "y_max": 0.45}
        self._object_reachable_areas["large_bowl"] = smaller_area
        self._object_reachable_areas["medium_bowl"] = smaller_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """Both bowls placed independently."""
        return ("large_bowl", "medium_bowl")

    def _add_custom_entities(self) -> None:
        """Add large and medium bowls."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Large bowl (mesh)
        large_bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.688,
                pos=(0.605, 0.0, 0.8204),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["large_bowl"] = {
            "entity": large_bowl,
        }

        # Medium bowl (mesh)
        medium_bowl = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb", pattern_is_dir=False),
                scale=1.227,
                pos=(0.40, 0.10, 0.8042),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["medium_bowl"] = {
            "entity": medium_bowl,
        }

    def _medium_inside_large(self, large: dict, medium: dict) -> bool:
        """Whether the medium bowl's bottom lies between the large bowl's bottom (plus NEST_MIN_LIFT) and its rim."""
        large_b, medium_b = large.get("bounds"), medium.get("bounds")
        if large_b is None or medium_b is None:
            return False
        medium_bottom = float(medium_b[0][2])
        return float(large_b[0][2]) + self.NEST_MIN_LIFT < medium_bottom < float(large_b[1][2])

    def _center_2d(self, obj: dict) -> np.ndarray | None:
        """A bowl's center in the table plane: mean of its outline's vertices, else its box's center, else pos."""
        hull = obj.get("convex_hull_2d")
        if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
            c = np.mean(hull, axis=0)
            if np.all(np.isfinite(c)):
                return c
        bounds = obj.get("bounds")
        if bounds is not None and isinstance(bounds, np.ndarray) and bounds.shape == (2, 3):
            (xmin, ymin, _), (xmax, ymax, _) = bounds
            return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=float)
        pos = obj.get("pos")
        if pos is not None and isinstance(pos, np.ndarray) and pos.size >= 2:
            return pos[:2].astype(float)
        return None

    def _on_table(self, obj: dict) -> bool:
        """Whether a bowl's outline (else its box, else its pos) lies within the table."""
        x_min, x_max, y_min, y_max = self.TABLE_XY
        hull = obj.get("convex_hull_2d")
        if hull is not None and isinstance(hull, np.ndarray) and hull.size >= 2:
            xs, ys = hull[:, 0], hull[:, 1]
        else:
            bounds = obj.get("bounds")
            if bounds is not None and isinstance(bounds, np.ndarray) and bounds.shape == (2, 3):
                xs, ys = bounds[:, 0], bounds[:, 1]
            else:
                pos = obj.get("pos")
                if pos is None or not isinstance(pos, np.ndarray) or pos.size < 2:
                    return False
                xs, ys = pos[:1], pos[1:2]
        tol = self.TABLE_TOL
        return bool(
            xs.min() >= x_min - tol and xs.max() <= x_max + tol and ys.min() >= y_min - tol and ys.max() <= y_max + tol
        )

    def _nest_offset(self, large: dict, medium: dict) -> float | None:
        """Distance between the bowls' centers in the table plane; None if either is off the table."""
        if not (self._on_table(large) and self._on_table(medium)):
            return None
        c_large, c_medium = self._center_2d(large), self._center_2d(medium)
        if c_large is None or c_medium is None:
            return None
        return float(np.linalg.norm(c_large - c_medium))

    def _nested_upright(self, large: dict, medium: dict) -> bool:
        """The medium bowl inside the large one, both upright (the goal geometry, short of concentricity)."""
        return (
            self._medium_inside_large(large, medium)
            and check_upright(large, self.UPRIGHT_MAX_TILT_DEG)
            and check_upright(medium, self.UPRIGHT_MAX_TILT_DEG)
        )

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: bowls nested concentrically, upright, the medium one released and still."""
        objs_info = self.collect_objs_info(["large_bowl", "medium_bowl"])
        large, medium = objs_info.get("large_bowl"), objs_info.get("medium_bowl")
        if large is None or medium is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        offset = self._nest_offset(large, medium)
        success = (
            offset is not None
            and offset <= self.CONCENTRIC_TOL
            and self._nested_upright(large, medium)
            and check_released(medium, self._robot)
            and check_still(medium)
        )
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dense reward on the shared ladder (utils.STAGE_*), each stage a condition of the success check.

        0 -> 0.2     a gripper approaches the medium bowl
        0.2 -> 0.5   (added to the approach score) the medium bowl moves toward the large one's center, from where it spawned
        0.5 -> 0.65  the medium bowl is inside the large one, both upright, scored by how close to concentric
        0.7          the goal geometry is met (nested, upright, concentric) but the medium bowl is still held
        0.85         the goal geometry is met and the medium bowl released, but still moving
        1.0          success
        """
        if self._check_success().item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
            return reward, {"success": reward}

        objs_info = self.collect_objs_info(["large_bowl", "medium_bowl"])
        large, medium = objs_info.get("large_bowl"), objs_info.get("medium_bowl")
        zero = torch.zeros(1, dtype=torch.float32, device=self._device)
        if large is None or medium is None:
            return zero, {"success": zero}

        score = approach_score(self._robot, [medium.get("pos")])
        c_large = self._center_2d(large)
        if c_large is not None and medium.get("pos") is not None:
            spawn = self._entities.get("medium_bowl", {}).get("spawn_pos")
            frac = progress_frac(medium["pos"], spawn, c_large, done_dist=self.CONCENTRIC_TOL)
            score += 0.3 * frac  # on top of the approach score

        offset = self._nest_offset(large, medium)
        if offset is not None and self._nested_upright(large, medium):
            if offset <= self.CONCENTRIC_TOL:
                score = max(score, at_goal_score(check_released(medium, self._robot)))
            else:
                concentric = max(0.0, 1.0 - (offset - self.CONCENTRIC_TOL) / 0.05)
                score = max(score, 0.5 + (STAGE_PROGRESS - 0.5) * concentric)

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"success": zero}
