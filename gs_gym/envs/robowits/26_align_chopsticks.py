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

    CHOPSTICK_NAMES = (
        "first_chopstick",
        "second_chopstick",
        "third_chopstick",
        "fourth_chopstick",
        "fifth_chopstick",
        "sixth_chopstick",
    )

    # _check_success requires every chopstick inside the table bounds, so a sampled
    # scene that opens with one overhanging the front edge is unsolvable, and its
    # progress score is capped at on_table/6 no matter how well the policy aligns them.
    # sample_scenes.py reads this to reject such layouts.
    MUST_STAY_ON_TABLE = CHOPSTICK_NAMES

    # Pairwise tolerance for success: every stick within TOL_DEG of the reference.
    TOL_DEG = 12.0

    # Progress ramps toward *perfect* alignment rather than toward the loosest
    # arrangement that would pass TOL_DEG. Alignment is a resultant length, which is
    # not monotone in the max-pairwise-deviation that success tests: five tight sticks
    # plus one 30° out scores 0.9906, above the 0.9891 of six sticks each sitting
    # exactly on the 12° limit -- so ramping to the latter let progress reach 1.0 in
    # states that do not pass. Ramping to 1.0 keeps a full score for real success,
    # which get_reward returns directly.
    PERFECT_ALIGNMENT = 1.0

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
            # Thin chopsticks resting on a board generate stiff contacts: at 4, 5
            # and 10 the solver still blew up with "Invalid constraint forces
            # causing 'nan'" on some eval scenes (scene 41 of the seed set at 10).
            # 20 is what stabilised the comparably thin rubber band in 16_06.
            config_overrides={"scene": {"substeps": 20}},
            **kwargs,
        )

    @property
    def placement_groups(self) -> PlacementGroups:
        """Ruler independent, board and chopsticks grouped."""
        return (
            "ruler",
            (
                "first_chopstick",
                "second_chopstick",
                "third_chopstick",
                "fourth_chopstick",
                "fifth_chopstick",
                "sixth_chopstick",
                "board",
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
            ("first_chopstick", (0.535, -0.02, 0.85), (0.0, 0.0, 180.0)),
            ("second_chopstick", (0.535, -0.02, 0.82), (0, 0, 0)),
            ("third_chopstick", (0.535, -0.00, 0.85), (0, 0, 0)),
            ("fourth_chopstick", (0.535, 0.0, 0.82), (0, 0, 0)),
            ("fifth_chopstick", (0.535, 0.02, 0.82), (0, 0, 0)),
            ("sixth_chopstick", (0.535, 0.02, 0.85), (0, 0, 0)),
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

    @staticmethod
    def _heading_xy(quat) -> np.ndarray:
        """Unit XY heading of a chopstick's thin end, or a zero vector if undefined.

        The heading is the stick's local +x axis rotated into the world, i.e. the
        first column of R(quat), projected onto the table plane. At euler (0,0,0) a
        settled stick measures 0.27 m along world x against 0.02 m across, so local
        +x is the long axis.

        Reading ``euler[2]`` instead agrees for a stick lying flat -- a pitch about y
        only scales the projection by cos(pitch) -- but an XYZ decomposition is
        degenerate at pitch = ±90° (a stick stood on end) and flips the heading's sign
        once the stick tips past vertical. Rotating the axis has neither failure.

        A near-vertical stick has no planar heading at all, so it contributes a zero
        vector: that lowers the resultant without injecting a direction the stick does
        not have, and it can never pass the success tolerance against another stick.
        """
        w, x, y, z = (float(v) for v in quat)
        v = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)], dtype=float)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-6:
            return np.zeros(2, dtype=float)
        return v / n

    def _check_success(self) -> torch.Tensor:
        """Check if task is successful: all chopsticks aligned."""
        objs_info = self.collect_objs_info()

        x_min, x_max = 0.21, 1.00
        y_min, y_max = -0.69, 0.69
        margin = 1e-3

        chopstick_names = self.CHOPSTICK_NAMES

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
            quat = objs_info[name].get("quat")
            if quat is None or len(quat) != 4:
                return torch.tensor([False], dtype=torch.bool, device=self._device)
            headings.append(self._heading_xy(quat))

        cos_tol = float(np.cos(np.deg2rad(self.TOL_DEG)))

        ref = headings[0]
        for h in headings[1:]:
            dot = float(np.dot(ref, h))
            if dot < cos_tol:
                return torch.tensor([False], dtype=torch.bool, device=self._device)

        return torch.tensor([True], dtype=torch.bool, device=self._device)

    def _raw_alignment(self) -> tuple[int, float] | None:
        """Return (chopsticks on the table, absolute alignment in [0.5, 1]), or None.

        None means the state could not be read (an entity or its quaternion missing),
        which callers treat as no progress rather than guessing.
        """
        objs_info = self.collect_objs_info()

        x_min, x_max = 0.21, 1.00
        y_min, y_max = -0.69, 0.69
        margin = 1e-3

        chopstick_names = self.CHOPSTICK_NAMES

        on_table = 0
        headings = []
        for name in chopstick_names:
            if name not in objs_info:
                return None
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

            quat = obj.get("quat")
            if quat is None or len(quat) != 4:
                return None
            headings.append(self._heading_xy(quat))

        # mean over h of (dot(mean_dir, h) + 1) / 2 reduces to 0.5 + |sum h| / (2n),
        # because sum_h dot(mean_dir, h) is just |sum h| -- the direction cancels out
        # and only the resultant's length ever mattered. Evaluating that closed form
        # removes the division by |sum h| along with the discontinuity it needed
        # guarding: an exactly balanced set returned 0.0 where the formula it was
        # standing in for gives 0.5. Three sticks each way is the mutation variants'
        # designed start, so the old guard fired at reset by construction.
        n = len(chopstick_names)
        resultant = float(np.linalg.norm(np.sum(headings, axis=0)))
        return on_table, 0.5 + resultant / (2.0 * n)

    def reset_env(self) -> None:
        """Reset, then remember the alignment this episode starts from."""
        super().reset_env()
        raw = self._raw_alignment()
        self._initial_alignment = None if raw is None else raw[1]

    def _alignment_score(self) -> float:
        """Progress in [0, 1): how far alignment has come from its reset value.

        Measured against the episode's own starting alignment, because the absolute
        value is not a measure of the policy. The seed scenes start with five of six
        sticks already pointing the same way -- alignment 0.8333 before the robot has
        moved -- and reporting that directly credited the initial state with 0.83 of
        the range, leaving 0.17 for the whole of the actual task. Success needs only
        that one odd stick flipped, so nearly all of the work lived in that 0.17.

        Still scaled by the on-table fraction: knocking sticks off the table is a
        failure, not progress, and must not be rewarded for raising alignment.
        """
        raw = self._raw_alignment()
        if raw is None:
            return 0.0
        on_table, alignment = raw

        baseline = getattr(self, "_initial_alignment", None)
        if baseline is None:
            # Scored before any reset: take the current state as the baseline rather
            # than crediting whatever the scene happens to open with.
            baseline = alignment
            self._initial_alignment = baseline

        span = self.PERFECT_ALIGNMENT - baseline
        if span <= 1e-9:
            # Already perfectly aligned at reset; _check_success short-circuits get_reward
            # in that case, and validate_eval_scenes.py rejects such scenes.
            return 0.0
        frac = max(0.0, min(1.0, (alignment - baseline) / span))
        return (on_table / len(self.CHOPSTICK_NAMES)) * frac

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward: 1.0 on success, else alignment progress score in [0, 1)."""
        success = self._check_success()
        if success.item():
            reward = torch.ones(1, dtype=torch.float32, device=self._device)
        else:
            reward = torch.tensor([self._alignment_score()], dtype=torch.float32, device=self._device)
        return reward, {"success": success.float()}
