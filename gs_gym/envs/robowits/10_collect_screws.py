"""CollectScrews environment for RoboWits.

Task: Collect a screw into a container using a dustpan.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/10-collect-screws-v0")
class CollectScrewsEnv(RoboWitsEnv):
    """Collect a screw into a container using a dustpan.

    The screw is too small to be easily grasped directly. The robot must
    use the dustpan to collect the screw first and then put it into the
    container.

    Success criteria:
    - Screw is within the container's bounding box
    - Screw is still on the table (not fallen off)
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Collect the screw into the container"

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
        smaller_area = {"x_min": 0.2, "x_max": 0.4, "y_min": -0.3, "y_max": 0.3}
        self._object_reachable_areas["dustpan"] = smaller_area
        self._object_reachable_areas["screw"] = smaller_area

    @property
    def placement_groups(self) -> PlacementGroups:
        """All objects placed independently."""
        return ("screw", "dustpan", "container")

    def _add_custom_entities(self) -> None:
        """Add container, dustpan, and screw."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Container (mesh) - fixed
        container = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/382432e6-4f0d-44b6-98f9-2f3a013a47e2/obj.glb", pattern_is_dir=False),
                scale=1.2,
                pos=(0.58, 0.00, 0.81892),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["container"] = {
            "entity": container,
        }

        # Dustpan (mesh) - using dustpan.STL
        dustpan = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path(
                    "hf_assets/dustpan.STL",
                    pattern_is_dir=False,
                ),
                scale=0.001,
                pos=(0.67, 0.35, 0.9696),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.9, 0.8, 0.6), double_sided=True),
        )
        self._entities["dustpan"] = {
            "entity": dustpan,
        }

        # Screw (mesh)
        screw = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb", pattern_is_dir=False),
                scale=1.5,
                pos=(0.56, -0.16, 0.7674),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(color=(0.6, 0.7, 0.9), double_sided=True),
        )
        self._entities["screw"] = {
            "entity": screw,
        }

    def _check_success(self, objs_info: dict | None = None) -> torch.Tensor:
        """Check if task is successful: screw in container."""
        if objs_info is None:
            objs_info = self.collect_objs_info()

        TABLE_X_MIN, TABLE_X_MAX = 0.21, 1.00
        TABLE_Y_MIN, TABLE_Y_MAX = -0.69, 0.69

        container = objs_info.get("container")

        if container is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        container_bounds = container.get("bounds")
        if container_bounds is None:
            ch = container.get("convex_hull_2d")
            if ch is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            ch = np.array(ch).astype(float)
            x_min, y_min = np.min(ch[:, 0]), np.min(ch[:, 1])
            x_max, y_max = np.max(ch[:, 0]), np.max(ch[:, 1])
            z_min = self.TABLE_Z - 0.02
            z_max = self.TABLE_Z + 1.0
            cmin = np.array([x_min, y_min, z_min])
            cmax = np.array([x_max, y_max, z_max])
        else:
            cb = np.array(container_bounds).astype(float)
            cmin, cmax = cb[0], cb[1]

        eps = 1e-3

        screw = objs_info.get("screw")

        if screw is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        screw_pos = screw.get("pos")
        if screw_pos is None:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        screw_pos = np.asarray(screw_pos, dtype=float)

        if not (
            (TABLE_X_MIN - 0.02) <= screw_pos[0] <= (TABLE_X_MAX + 0.02)
            and (TABLE_Y_MIN - 0.02) <= screw_pos[1] <= (TABLE_Y_MAX + 0.02)
            and screw_pos[2] >= (self.TABLE_Z - 0.05)
        ):
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        inside = (
            (cmin[0] - eps) <= screw_pos[0] <= (cmax[0] + eps)
            and (cmin[1] - eps) <= screw_pos[1] <= (cmax[1] + eps)
            and (cmin[2] - 0.02) <= screw_pos[2] <= (cmax[2] + 0.02)
        )
        return torch.tensor([bool(inside)], dtype=torch.bool, device=self._device)

    def _check_screw_on_dustpan(self, objs_info: dict) -> bool:
        """Return True if screw is resting on the dustpan."""
        dustpan = objs_info.get("dustpan")
        screw = objs_info.get("screw")
        if dustpan is None or screw is None:
            return False
        screw_pos = screw.get("pos")
        if screw_pos is None:
            return False
        screw_pos = np.asarray(screw_pos, dtype=float)

        dp_bounds = dustpan.get("bounds")
        if dp_bounds is None:
            return False
        dp_bounds = np.asarray(dp_bounds, dtype=float)
        dp_min, dp_max = dp_bounds[0], dp_bounds[1]

        eps = 0.01
        in_xy = (dp_min[0] - eps) <= screw_pos[0] <= (dp_max[0] + eps) and (dp_min[1] - eps) <= screw_pos[1] <= (
            dp_max[1] + eps
        )
        near_z = dp_min[2] - 0.01 <= screw_pos[2] <= dp_max[2] + 0.05
        return bool(in_xy and near_z)

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: continuous progress 0→1.

        Phase 1 (0.0–0.5): dustpan approaching screw, scaled by proximity.
        Phase 2 (0.5–1.0): screw on dustpan, scaled by proximity to container.
        """
        objs_info = self.collect_objs_info()
        success = self._check_success(objs_info)
        screw_on_dustpan = self._check_screw_on_dustpan(objs_info)

        if success.item():
            progress = 1.0
        else:
            dustpan = objs_info.get("dustpan", {})
            screw = objs_info.get("screw", {})
            container = objs_info.get("container", {})

            dp_pos = np.asarray(dustpan.get("pos", [0, 0, 0]), dtype=float)
            screw_pos = np.asarray(screw.get("pos", [0, 0, 0]), dtype=float)
            container_pos = np.asarray(container.get("pos", [0, 0, 0]), dtype=float)

            if screw_on_dustpan:
                # Phase 2: how close is screw to container (xy only)
                d = float(np.linalg.norm(screw_pos[:2] - container_pos[:2]))
                progress = 0.5 + 0.5 * max(0.0, 1.0 - d / 0.5)
            else:
                # Phase 1: how close is dustpan to screw (xy only)
                d = float(np.linalg.norm(dp_pos[:2] - screw_pos[:2]))
                progress = 0.5 * max(0.0, 1.0 - d / 0.5)

        reward = torch.tensor([progress], dtype=torch.float32, device=self._device)
        return reward, {
            "success": success.float(),
            "screw_on_dustpan": torch.tensor([float(screw_on_dustpan)], device=self._device),
        }
