"""AlignBlocks environment for RoboWits.

Task: Align 3 cubes in a straight line using a ruler.
"""

import genesis as gs
import torch

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import RoboWitsEnv


@register_task("robowits/01-align-blocks-v0")
class AlignBlocksEnv(RoboWitsEnv):
    """Align 3 cubes in a straight line using a ruler.

    Task success criteria:
    1. All objects still on table (not fallen)
    2. All cubes at table surface height
    3. Three cubes are collinear within 1cm tolerance
    4. Minimum 3cm separation between furthest cubes

    Observation: Dict with "agent_pos" (16D robot state) and "pixels" (camera images)
    Action: 18-DOF joint positions
    """

    @property
    def task_description(self) -> str:
        """Return a description of the current task."""
        return "Align the three cubes perfectly in a straight line."

    # Cube configuration
    CUBE_SIZE = (0.05, 0.05, 0.05)  # 5cm cubes
    NUM_CUBES = 3
    CUBE_NAMES = ("first target cube", "second target cube", "third target cube")

    # Ruler configuration
    RULER_SIZE = (0.30, 0.03, 0.005)  # 30cm x 3cm x 0.5cm (fallback box)
    RULER_SCALE = 1.2  # Scale for mesh asset

    # Success thresholds
    COLLINEARITY_TOLERANCE = 0.01  # 1cm
    MIN_SPREAD = 0.03  # 3cm minimum separation

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
        far_reachable_area = {"x_min": 0.45, "x_max": 0.6, "y_min": -0.25, "y_max": 0.25}
        close_reachable_area = {"x_min": 0.2, "x_max": 0.5, "y_min": -0.35, "y_max": 0.35}
        self._object_reachable_areas["first target cube"] = far_reachable_area
        self._object_reachable_areas["second target cube"] = far_reachable_area
        self._object_reachable_areas["third target cube"] = far_reachable_area
        self._object_reachable_areas["long rigid ruler"] = close_reachable_area

    def _add_custom_entities(self) -> None:
        """Add cubes and ruler to the scene."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Add 3 target cubes
        cube_positions = [
            (0.56, -0.05, 0.785),
            (0.62, 0.12, 0.785),
            (0.48, 0.03, 0.785),
        ]
        for name, pos in zip(self.CUBE_NAMES, cube_positions, strict=False):
            cube = self._scene.scene.add_entity(
                gs.morphs.Box(
                    pos=pos,
                    size=self.CUBE_SIZE,
                ),
                material=gs.materials.Rigid(rho=200.0),
                surface=gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
            )
            self._entities[name] = {
                "entity": cube,
            }

        # Add long rigid ruler (mesh)
        ruler = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                file=get_asset_path("blender_kit/33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb", pattern_is_dir=False),
                coacd_options=coacd_options,
                scale=self.RULER_SCALE,
                pos=(0.43, 0.0, 0.76 + 0.0097 * 1.2),
                euler=(0, 0, 90),
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["long rigid ruler"] = {
            "entity": ruler,
        }

    @property
    def placement_groups(self) -> tuple[str, ...]:
        """Objects to randomize on reset."""
        return ("first target cube", "second target cube", "third target cube", "long rigid ruler")

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward based on EE-to-cube proximity and cube alignment."""
        cube_positions = self._get_cube_positions()

        reach = self._compute_reach_reward(cube_positions)  # [0, 0.5]
        alignment = self._compute_alignment_reward(cube_positions)  # [0, 1]

        total_reward = torch.max(reach, alignment)

        reward_dict = {
            "reach": reach,
            "alignment": alignment,
        }

        return total_reward, reward_dict

    def _get_cube_positions(self) -> torch.Tensor:
        """Get positions of all cubes.

        Returns:
            (n_envs, 3, 3) tensor of cube positions
        """
        positions = []
        for name in self.CUBE_NAMES:
            cube = self._entities[name]["entity"]
            pos = cube.get_pos()  # (n_envs, 3)
            positions.append(pos)

        return torch.stack(positions, dim=1)  # (n_envs, 3, 3)

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        import numpy as np

        objs_info = self.collect_objs_info()

        TABLE_BOUNDS = {
            "x_min": 0.21,
            "x_max": 1.00,
            "y_min": -0.69,
            "y_max": 0.69,
            "z_surf": 0.76,
        }

        cube_names = [
            "first target cube",
            "second target cube",
            "third target cube",
        ]

        for n in cube_names:
            if n not in objs_info:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        def get_xy(name):
            p = objs_info[name]["pos"]
            return np.array([p[0], p[1]], dtype=float)

        def pairwise_dists(pts):
            d01 = np.linalg.norm(pts[0] - pts[1])
            d12 = np.linalg.norm(pts[1] - pts[2])
            d02 = np.linalg.norm(pts[0] - pts[2])
            return [(d01, (0, 1)), (d12, (1, 2)), (d02, (0, 2))]

        def point_line_distance(p, a, b):
            v = b - a
            w = p - a
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-9:
                return np.linalg.norm(p - a)
            cross_z = v[0] * w[1] - v[1] * w[0]
            return abs(cross_z) / (v_norm + 1e-12)

        def on_table_surface(name, z_tol=0.025):
            info = objs_info[name]
            b = info.get("bounds", None)
            if b is None:
                return True
            zmin = float(b[0][2])
            return abs(zmin - TABLE_BOUNDS["z_surf"]) <= z_tol

        margin = 0.03
        for name, info in objs_info.items():
            lname = name.lower()
            if "table" in lname or "robot" in lname:
                continue
            pos = info.get("pos", None)
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if (
                x < TABLE_BOUNDS["x_min"] - margin
                or x > TABLE_BOUNDS["x_max"] + margin
                or y < TABLE_BOUNDS["y_min"] - margin
                or y > TABLE_BOUNDS["y_max"] + margin
            ):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        for n in cube_names:
            if not on_table_surface(n, z_tol=0.025):
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        pts = [get_xy(n) for n in cube_names]

        dists = pairwise_dists(pts)
        max_pair = max(dists, key=lambda x: x[0])
        max_dist, (i, j) = max_pair
        if max_dist < 0.03:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        k = 3 - i - j
        line_tol = 0.010
        perp_dist = point_line_distance(pts[k], pts[i], pts[j])
        if perp_dist > line_tol:
            return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def _compute_reach_reward(self, cube_positions: torch.Tensor) -> torch.Tensor:
        """Reward for proximity of either EE to the nearest cube. Capped at 0.5.

        Args:
            cube_positions: (n_envs, 3, 3) cube positions

        Returns:
            (n_envs,) reward in [0, 0.5]
        """
        robot = self._robot
        right_ee = robot.ee_link_right.get_pos()  # (n_envs, 3)
        left_ee = robot.ee_link_left.get_pos()  # (n_envs, 3)

        min_dist = torch.full((self._n_envs,), float("inf"), device=self._device)
        for i in range(self.NUM_CUBES):
            cube_pos = cube_positions[:, i, :]  # (n_envs, 3)
            dist_r = torch.norm(right_ee - cube_pos, dim=1)
            dist_l = torch.norm(left_ee - cube_pos, dim=1)
            min_dist = torch.minimum(min_dist, torch.minimum(dist_r, dist_l))

        # dist=0 → reward=0.5; dist≥0.5m → reward≈0
        return torch.clamp(0.5 * (1.0 - min_dist / 0.5), 0.0, 0.5)

    def _compute_alignment_reward(self, cube_positions: torch.Tensor) -> torch.Tensor:
        """Compute reward for cube alignment.

        Args:
            cube_positions: (n_envs, 3, 3) cube positions

        Returns:
            (n_envs,) reward in [0, 1]
        """
        # Compute triangle area (collinearity measure)
        p1 = cube_positions[:, 0, :2]
        p2 = cube_positions[:, 1, :2]
        p3 = cube_positions[:, 2, :2]

        v1 = p2 - p1
        v2 = p3 - p1

        cross = torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        area = cross / 2

        # Reward is inverse of area (smaller area = better alignment)
        # Normalize: area of 0.01 (1cm height triangle) gives reward of ~0
        return torch.clamp(1 - area / 0.01, 0, 1)
