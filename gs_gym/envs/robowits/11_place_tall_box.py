"""PlaceTallBox environment for RoboWits.

Task: Move a tall box onto a target area under a beam obstacle.
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
    intersection_area,
    polygon_area,
    progress_frac,
)


@register_task("robowits/11-place-tall-box-v0")
class PlaceTallBoxEnv(RoboWitsEnv):
    """Move a tall box onto a target area under a beam obstacle.

    The obstacle enforces a geometric clearance constraint. The beam cannot
    be moved or raised. The solution relies on changing orientation to reduce
    the effective height, exploiting the different cross-sectional dimensions
    of the box to fit under the low beam.

    Success criteria:
    - Tall box covers the target area by more than 60%
    - Tall box rests on the target area (its bottom at the pad's top surface), not above it
    - Tall box is not held by either gripper
    - Both objects remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Move the rectangular box onto the green target area."

    TABLE_Z = 0.76
    # How far the box's lowest collision point may sit above the target pad's top and still count as resting on it.
    CONTACT_Z_TOL = 0.01
    COVERAGE_MIN = 0.6  # share of the smaller footprint (box or target) the two must overlap
    TABLE_XY = (0.21, 1.00, -0.69, 0.69)  # x_min, x_max, y_min, y_max

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
        """Tall box independent, obstacles grouped together."""
        return ("tall_cardboard_box", ("support_block_left", "support_block_right", "beam_plank", "target_area"))

    def _add_custom_entities(self) -> None:
        """Add tall box, support blocks, beam, and target area."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Tall cardboard box (mesh)
        box = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/8fb31a9b-33d5-4246-997f-84307520c1a0/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.42, 0.0, 0.8422),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["tall_cardboard_box"] = {
            "entity": box,
        }

        # Support block left (fixed)
        support_left = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, -0.15, 0.8025),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.06, 0.085),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support_block_left"] = {
            "entity": support_left,
        }

        # Support block right (fixed)
        support_right = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.15, 0.8025),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.06, 0.085),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6, ior=1.5),
        )
        self._entities["support_block_right"] = {
            "entity": support_right,
        }

        # Beam plank (fixed)
        beam = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.855),
                euler=(0.0, 0.0, 0.0),
                size=(0.12, 0.36, 0.02),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.7, 0.55, 0.3), roughness=0.7, ior=1.5),
        )
        self._entities["beam_plank"] = {
            "entity": beam,
        }

        # Target area
        target = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.55, 0.0, 0.761),
                euler=(0.0, 0.0, 0.0),
                size=(0.10, 0.12, 0.002),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        )
        self._entities["target_area"] = {
            "entity": target,
        }

    def _hull_on_table(self, hull: np.ndarray) -> bool:
        x_min, x_max, y_min, y_max = self.TABLE_XY
        return bool(
            hull[:, 0].min() >= x_min - 1e-3
            and hull[:, 0].max() <= x_max + 1e-3
            and hull[:, 1].min() >= y_min - 1e-3
            and hull[:, 1].max() <= y_max + 1e-3
        )

    def _coverage(self, box: dict, tgt: dict) -> float:
        """Overlap of box and target as a share of the smaller footprint; 0 if either is off the table."""
        box_hull, tgt_hull = box.get("convex_hull_2d"), tgt.get("convex_hull_2d")
        if box_hull is None or tgt_hull is None or len(box_hull) < 3 or len(tgt_hull) < 3:
            return 0.0
        box_hull, tgt_hull = np.asarray(box_hull, dtype=float), np.asarray(tgt_hull, dtype=float)
        if not (self._hull_on_table(box_hull) and self._hull_on_table(tgt_hull)):
            return 0.0
        smaller = min(polygon_area(box_hull), polygon_area(tgt_hull))
        if smaller < 1e-6:
            return 0.0
        return intersection_area(box_hull, tgt_hull) / smaller

    def _box_on_pad(self, box: dict, tgt: dict) -> bool:
        """Whether the box's lowest collision point is at the pad's top (the pad has no collision geometry)."""
        return float(box["bounds"][0][2]) <= float(tgt["bounds"][1][2]) + self.CONTACT_Z_TOL

    def _box_at_goal(self, box: dict, tgt: dict) -> bool:
        """The goal geometry: the box covers the target by more than COVERAGE_MIN, resting at the pad's height."""
        return self._coverage(box, tgt) > self.COVERAGE_MIN and self._box_on_pad(box, tgt)

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: box covers the target by 60%, rests on it, and is not held."""
        objs_info = self.collect_objs_info(["tall_cardboard_box", "target_area"])
        box, tgt = objs_info.get("tall_cardboard_box"), objs_info.get("target_area")
        if box is None or tgt is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)
        success = self._box_at_goal(box, tgt) and check_released(box, self._robot)
        return torch.tensor([success], dtype=torch.bool, device=self._device)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dense reward on the shared ladder (utils.STAGE_*), each stage a condition of the success check.

        0 -> 0.2     a gripper approaches the box
        0.2 -> 0.5   (added to the approach score) the box moves toward the target, from where it spawned
        0.5 -> 0.65  the box covers the target, up to COVERAGE_MIN
        0.7          the goal geometry is met (covering, at the pad's height) but the box is still held
        1.0          success
        """
        if self._check_success().item():
            score_t = torch.ones((), dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        objs_info = self.collect_objs_info(["tall_cardboard_box", "target_area"])
        box, tgt = objs_info.get("tall_cardboard_box"), objs_info.get("target_area")
        if box is None or tgt is None:
            score_t = torch.zeros((), dtype=torch.float32, device=self._device)
            return score_t, {"score": score_t}

        score = approach_score(self._robot, [box["pos"]])
        spawn = self._entities.get("tall_cardboard_box", {}).get("spawn_pos")
        frac = progress_frac(box["pos"], spawn, tgt["pos"])
        score += 0.3 * frac  # on top of the approach score
        coverage = self._coverage(box, tgt)
        if coverage > 0.0:
            score = max(score, 0.5 + (STAGE_PROGRESS - 0.5) * min(1.0, coverage / self.COVERAGE_MIN))
        if self._box_at_goal(box, tgt):
            score = max(score, at_goal_score(check_released(box, self._robot)))

        score_t = torch.tensor(score, dtype=torch.float32, device=self._device)
        return score_t, {"score": score_t}
