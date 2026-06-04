"""AlignChopsticks environment for RoboWits.

Task: Make all six chopsticks point in the same direction.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups, RoboWitsEnv


@register_task("robowits/26-align-chopsticks-v0")
class AlignChopsticksEnv(RoboWitsEnv):
    """Align six chopsticks so they all point in the same direction.

    Initially, the chopsticks lean on a board with mixed orientations. By pushing
    a ruler to slide the chopsticks further onto the board, the asymmetric mass
    distribution causes chopsticks of different orientations to behave differently.

    Success criteria:
    - All six chopsticks have their thin ends pointing in the same direction
    - Chopsticks remain within table bounds
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return (
            "Make all six chopsticks point in the same direction, with their small thin ends all facing the same way."
        )

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
            config_overrides={"scene": {"substeps": 5}},
            **kwargs,
        )

    @property
    def placement_groups(self) -> PlacementGroups:
        """Ruler independent, board and chopsticks grouped."""
        return (
            "ruler",
            (
                "board",
                "first chopstick",
                "second chopstick",
                "third chopstick",
                "fourth chopstick",
                "fifth chopstick",
                "sixth chopstick",
            ),
        )

    def _add_custom_entities(self) -> None:
        """Add board, ruler, and six chopsticks."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Board (mesh)
        board = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb", pattern_is_dir=False),
                scale=1.0,
                pos=(0.74, 0.0, 0.76 + 0.01),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=1.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["board"] = {
            "entity": board,
        }

        # Ruler (mesh)
        ruler = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.56, 0.33, 0.8),
                euler=(0.0, 0.0, 0.0),
                size=(0.2, 0.03, 0.03),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=50.0, friction=1.0),
            surface=gs.surfaces.Rough(color=(0.3, 0.3, 0.3), double_sided=True),
        )
        self._entities["ruler"] = {
            "entity": ruler,
        }

        # Chopsticks (6 mesh objects)
        chopstick_configs = [
            ("first chopstick", (0.535, -0.02, 0.85), (0.0, 0.0, 180.0)),
            ("second chopstick", (0.535, -0.02, 0.82), (0, 0, 0)),
            ("third chopstick", (0.535, -0.00, 0.85), (0, 0, 0)),
            ("fourth chopstick", (0.535, 0.0, 0.82), (0, 0, 0)),
            ("fifth chopstick", (0.535, 0.02, 0.82), (0, 0, 0)),
            ("sixth chopstick", (0.535, 0.02, 0.85), (0, 0, 0)),
        ]

        for name, pos, euler in chopstick_configs:
            chopstick = self._scene.scene.add_entity(
                gs.morphs.Mesh(
                    coacd_options=coacd_options,
                    file=get_asset_path("hf_assets/chopstick.glb", pattern_is_dir=False),
                    scale=(1.2, 2.8, 2.8),
                    pos=pos,
                    euler=euler,
                    fixed=False,
                    collision=True,
                ),
                material=gs.materials.Rigid(rho=200.0, friction=0.7),
                surface=gs.surfaces.Smooth(double_sided=True),
            )
            self._entities[name] = {
                "entity": chopstick,
            }

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: all chopsticks aligned."""
        objs_info = self.collect_objs_info()

        x_min, x_max = 0.21, 1.00
        y_min, y_max = -0.69, 0.69
        margin = 1e-3

        chopstick_names = [
            "first chopstick",
            "second chopstick",
            "third chopstick",
            "fourth chopstick",
            "fifth chopstick",
            "sixth chopstick",
        ]

        for name in chopstick_names:
            if name not in objs_info:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        def is_within_table_xy(obj: dict) -> bool:
            hull = obj.get("convex_hull_2d")
            if hull is not None and isinstance(hull, np.ndarray) and hull.size > 0:
                xs = hull[:, 0]
                ys = hull[:, 1]
            else:
                b = obj.get("bounds")
                if b is not None and isinstance(b, np.ndarray) and b.shape == (2, 3):
                    xs = np.array([b[0, 0], b[1, 0]])
                    ys = np.array([b[0, 1], b[1, 1]])
                else:
                    pos = obj.get("pos")
                    if pos is None:
                        return False
                    xs = np.array([pos[0]])
                    ys = np.array([pos[1]])
            return (
                xs.min() >= x_min - margin
                and xs.max() <= x_max + margin
                and ys.min() >= y_min - margin
                and ys.max() <= y_max + margin
            )

        for name in chopstick_names:
            if not is_within_table_xy(objs_info[name]):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        headings = []
        for name in chopstick_names:
            euler = objs_info[name].get("euler", None)
            if euler is None:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            try:
                yaw_rad = float(euler[2])
            except Exception:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            h = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=float)
            n = np.linalg.norm(h)
            if not np.isfinite(n) or n < 1e-8:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            headings.append(h / n)

        tol_deg = 12.0
        cos_tol = float(np.cos(np.deg2rad(tol_deg)))

        ref = headings[0]
        for h in headings[1:]:
            dot = float(np.dot(ref, h))
            if dot < cos_tol:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def _alignment_score(self) -> float:
        """Progress score in [0, 1]: mean heading alignment weighted by on-table fraction."""
        objs_info = self.collect_objs_info()

        x_min, x_max = 0.21, 1.00
        y_min, y_max = -0.69, 0.69
        margin = 1e-3

        chopstick_names = [
            "first chopstick",
            "second chopstick",
            "third chopstick",
            "fourth chopstick",
            "fifth chopstick",
            "sixth chopstick",
        ]

        on_table = 0
        headings = []
        for name in chopstick_names:
            if name not in objs_info:
                return 0.0
            obj = objs_info[name]

            hull = obj.get("convex_hull_2d")
            if hull is not None and isinstance(hull, np.ndarray) and hull.size > 0:
                xs, ys = hull[:, 0], hull[:, 1]
            else:
                b = obj.get("bounds")
                if b is not None and isinstance(b, np.ndarray) and b.shape == (2, 3):
                    xs = np.array([b[0, 0], b[1, 0]])
                    ys = np.array([b[0, 1], b[1, 1]])
                else:
                    pos = obj.get("pos")
                    xs = np.array([pos[0]] if pos is not None else [0.0])
                    ys = np.array([pos[1]] if pos is not None else [0.0])
            if (
                xs.min() >= x_min - margin
                and xs.max() <= x_max + margin
                and ys.min() >= y_min - margin
                and ys.max() <= y_max + margin
            ):
                on_table += 1

            euler = obj.get("euler")
            if euler is None:
                return 0.0
            try:
                yaw_rad = float(euler[2])
            except Exception:
                return 0.0
            h = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=float)
            n = np.linalg.norm(h)
            if not np.isfinite(n) or n < 1e-8:
                return 0.0
            headings.append(h / n)

        mean_vec = np.sum(headings, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-8:
            return 0.0
        mean_dir = mean_vec / norm

        alignment = float(np.mean([(float(np.dot(mean_dir, h)) + 1.0) / 2.0 for h in headings]))
        return (on_table / len(chopstick_names)) * alignment

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else alignment progress score in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
        else:
            reward = torch.tensor([self._alignment_score()], dtype=torch.float32, device=self._device)
        return reward, {"success": success.float()}
