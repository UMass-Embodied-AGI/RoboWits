"""Base RoboWits environment for Genesis simulator.

Provides common functionality for RoboWits manipulation tasks.
"""

from __future__ import annotations

import json
import logging
import os
from abc import abstractmethod
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

from gs_gym.envs.base_gym_env import GenesisGymEnv
from gs_gym.envs.robowits import utils as _utils
from gs_gym.envs.robowits.placement import _is_fixed_entity
from gs_gym.robots.bimanual_marvin import BimanualMarvinRobot
from gs_gym.scenes.flat_scene import FlatScene

logger = logging.getLogger(__name__)

# Type alias for placement groups: each element is either a single object name (str)
# or a tuple of object names that should maintain relative positions during placement.
# Example: ("cube_a", ("cube_b", "cube_c"), "cube_d") means cube_b and cube_c are grouped.
PlacementGroup = str | tuple[str, ...]
PlacementGroups = tuple[PlacementGroup, ...] | None


class RoboWitsEnv(GenesisGymEnv):
    """Base class for RoboWits manipulation environments.

    Provides:
    - BimanualMarvin robot with 18-DOF control
    - Random object placement with collision avoidance
    - Custom entity support via _add_custom_entities()
    - Reachable area bounds for placement
    """

    # Robot dimensions
    ROBOT_DOF = 16  # 7 arm joints + 1 gripper width per arm

    # Bimanual Marvin arm 3D occupied volume at rest pose (robot base at (0,0,1.08)).
    # Measured from all link-center positions after reset(); values are link centers
    # only — add ~0.04 m for link radius when using as collision bounds.
    # Forearm+wrist+gripper (the region near table height, z ≈ 0.84–0.94 m):
    #   Right arm: x=[0.018, 0.622], y=[-0.334, -0.244], z=[0.857, 0.935]
    #   Left arm:  x=[0.019, 0.619], y=[ 0.243,  0.332], z=[0.842, 0.933]
    # Full arm (shoulder through gripper):
    #   Right arm: x=[0.000, 0.622], y=[-0.334, -0.037], z=[0.857, 1.200]
    #   Left arm:  x=[0.000, 0.619], y=[ 0.037,  0.332], z=[0.842, 1.200]
    ARM_FOREARM_BBOX_RIGHT = {"x": (0.018, 0.622), "y": (-0.334, -0.244), "z": (0.857, 0.935)}
    ARM_FOREARM_BBOX_LEFT = {"x": (0.019, 0.619), "y": (0.243, 0.332), "z": (0.842, 0.933)}
    ARM_FULL_BBOX_RIGHT = {"x": (0.000, 0.622), "y": (-0.334, -0.037), "z": (0.857, 1.200)}
    ARM_FULL_BBOX_LEFT = {"x": (0.000, 0.619), "y": (0.037, 0.332), "z": (0.842, 1.200)}

    # Default reachable bounds (x, y) for object placement
    DEFAULT_REACHABLE_BOUNDS = (
        (0.2, 0.75),  # x range
        (-0.45, 0.45),  # y range
    )

    # Maximum number of times to retry the whole random-placement pass when an
    # individual ``place_*`` call fails to find a collision-free arrangement.
    MAX_PLACEMENT_RETRIES: int = 50

    # Table configuration
    TABLE_HEIGHT = 0.76  # Objects spawn at this height + half their height
    TABLE_SIZE = (0.85, 1.5, 0.76)  # (width_x, depth_y, height_z)
    TABLE_POS = (0.597, 0.0, 0.38)  # Center of table (x, y, z=height/2)

    # Number of physics steps to run after placement to let objects settle
    _preprocess_steps: int = 10

    def placement_entity_extra_hulls(self, name: str) -> list:
        """Return arm exclusion hulls for entities tall enough to collide with the arm at rest.

        The forearm/wrist/gripper links sit at z ≈ 0.84–0.93 m. Any entity whose
        top (TABLE_HEIGHT + height) exceeds the arm's lowest link gets arm exclusion
        hulls injected into its placement collision check. Short entities are unconstrained.

        Fixed entities are skipped: they are structural task elements (beams, platforms,
        target markers) that the arm is designed to work around, not freely-placed objects.
        """
        entity_info = self._entities.get(name)
        if entity_info and _is_fixed_entity(entity_info["entity"]):
            return []
        arm_z_min = min(self.ARM_FOREARM_BBOX_RIGHT["z"][0], self.ARM_FOREARM_BBOX_LEFT["z"][0])
        size = (entity_info or {}).get("size") or (0, 0, 0)
        if self.TABLE_HEIGHT + size[2] <= arm_z_min:
            return []
        pad = 0.05

        def _hull(bbox):
            x0, x1 = bbox["x"][0] - pad, bbox["x"][1] + pad
            y0, y1 = bbox["y"][0] - pad, bbox["y"][1] + pad
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

        return [_hull(self.ARM_FOREARM_BBOX_RIGHT), _hull(self.ARM_FOREARM_BBOX_LEFT)]

    def __init__(
        self,
        config_name: str = "robowits_default",
        n_envs: int = 1,
        show_viewer: bool = False,
        max_episode_steps: int | None = None,
        control_freq: float | None = None,
        control_mode: str = "EE_ABS",
        observation_mode: str = "EE",
        config_overrides: dict[str, Any] | None = None,
        eval_dataset_json: str | None = None,
        **kwargs,
    ):
        """Initialize the RoboWits environment.

        Args:
            config_name: Configuration file name
            n_envs: Number of parallel environments (DEPRECATED - ignored, use gs_gym.make(task, n_envs=N) for vectorization)
            show_viewer: Whether to show the viewer
            max_episode_steps: Maximum steps per episode
            control_mode: Robot control mode. Options:
                - "JOINT_ABS": 16D absolute joint position control (radians)
                - "JOINT_DELTA": 16D delta joint position control (radians, added to current)
                - "EE_DELTA": 14D delta EE control
                - "EE_ABS": 14D absolute EE control (IK handled internally)
            observation_mode: Observation state format. Options:
                - "EE": 14D (ee_pos + ee_axis_angle + gripper)
                - "JOINT": 16D (joints + gripper)
            config_overrides: Optional dictionary of config values to override.
                              Supports nested keys via dict merging.
            eval_dataset_json: Optional path to evaluation dataset JSON file.
                               If provided, overrides the config-based evaluation settings.
            **kwargs: Additional arguments
        """
        self._n_envs = 1  # Always single env now (vectorization handled externally)
        self._control_mode = control_mode
        self._observation_mode = observation_mode
        self._config_overrides = config_overrides
        self._eval_dataset_json = eval_dataset_json

        super().__init__(config_name=config_name, **kwargs)

        # Apply config overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)

        # Override max_episode_steps if provided
        if max_episode_steps is not None:
            if "termination" not in self._config:
                self._config["termination"] = {}
            self._config["termination"]["max_episode_steps"] = max_episode_steps

        # Load and convert config
        self._args = self._config_to_env_args(self._config)

        # Create scene with collision enabled
        self._scene = FlatScene(
            n_envs=self._n_envs,
            args=self._args.scene_args,
            show_viewer=show_viewer,
            enable_collision=True,
            add_ground_plane=True,
        )
        self.dt = self._args.scene_args.dt

        # Create robot
        # Validate control_mode
        valid_modes = ("JOINT_ABS", "JOINT_DELTA", "EE_ABS", "EE_DELTA")
        if control_mode not in valid_modes:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Available: {valid_modes}")

        # Pass control mode directly to robot
        self._robot = BimanualMarvinRobot(
            n_envs=self._n_envs,
            scene=self._scene.scene,
            args=self._args.robot_args,
            device=self.device,
            control_mode=control_mode,
        )

        # Create all cameras (ego + wrist) before scene.build()
        self._cameras: dict[str, Any] = {}
        self._create_cameras()

        # Storage for custom entities (cubes, tools, etc.)
        self._entities: dict[str, Any] = {}

        # Add the table entity (required for RoboWits coordinate system)
        self._add_table()

        # Let subclasses add custom entities
        self._add_custom_entities()

        # Build scene after all entities and cameras are added
        self._scene.build()

        # Capture initial entity poses from morph definitions (must be after scene.build())
        self._capture_initial_entity_poses()

        # Initialize robot DOF indices from joint names (after scene.build())
        self._robot.init_dof_indices()

        self._eval_dataset: dict[str, Any] | None = None
        self._eval_dataset_keys: list[str] = []
        self._eval_dataset_idx: int = 0
        self._eval_particles: dict[str, np.ndarray] = {}
        self._load_eval_dataset()

        # Initialize auxiliary variables
        self._init()

    def _create_cameras(self):
        """Create ego + wrist cameras before scene.build(). Wrist cameras
        use entity_idx / link_idx_local / offset_T to attach to robot links.
        """
        from genesis.options.sensors import RasterizerCameraOptions

        scene = self._scene.scene

        # ── Ego camera (static world-view) ──
        # entity_idx=None: not attached to any entity. Must be explicit because
        # RasterizerCameraOptions defaults entity_idx to -1 (Python last-element
        # indexing), which fails validation when the last scene entity is an MPM entity.
        self._cameras["ego"] = scene.add_sensor(
            RasterizerCameraOptions(
                pos=(0.1468, -0.0175, 1.4),
                lookat=(0.4208, -0.0175, 1.0735),
                fov=65,
                res=(848, 480),
                entity_idx=None,
            )
        )

        # ── Wrist cameras (attached to gripper base links) ───────────────
        # Offset transform (pika gripper + D405 realsense)
        wrist_offset_T = np.array(
            [
                [-1, 0, 0, 0.009],
                [0, 1, 0, 0.04931],
                [0, 0, -1, 0.06159 - 0.0037 + 0.002],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        robot_entity = self._robot.robot_entity
        entity_idx = robot_entity.idx

        right_link = robot_entity.get_link(self._robot.GRIPPER_BASE_LINK_RIGHT)
        left_link = robot_entity.get_link(self._robot.GRIPPER_BASE_LINK_LEFT)

        self._cameras["wrist_right"] = scene.add_sensor(
            RasterizerCameraOptions(
                res=(848, 480),
                fov=58,
                up=(0.0, 0.0, 1.0),
                entity_idx=entity_idx,
                link_idx_local=right_link.idx_local,
                offset_T=wrist_offset_T,
            )
        )
        self._cameras["wrist_left"] = scene.add_sensor(
            RasterizerCameraOptions(
                res=(848, 480),
                fov=58,
                up=(0.0, 0.0, 1.0),
                entity_idx=entity_idx,
                link_idx_local=left_link.idx_local,
                offset_T=wrist_offset_T,
            )
        )

    def _add_table(self) -> None:
        """Add table entity to the scene."""
        from gs_gym.common.utils.asset_utils import (
            get_table_path,
            get_worktable_texture_normal_path,
            get_worktable_texture_roughness_path,
        )

        surface_kwargs = {}
        texture_from_path = getattr(gs.textures, "Image", None) or getattr(gs.textures, "Texture", None)
        if texture_from_path is not None and callable(texture_from_path):
            try:
                surface_kwargs["normal_texture"] = texture_from_path(path=get_worktable_texture_normal_path())
                surface_kwargs["roughness_texture"] = texture_from_path(path=get_worktable_texture_roughness_path())
            except (TypeError, Exception):
                pass

        surface = gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
            opacity=1.0,
            vis_mode="visual",
            **surface_kwargs,
        )

        table = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                file=get_table_path(),
                pos=(0.597, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                scale=(1.14, 1.0, 1.4377),  # height ≈ 0.76 m
                fixed=True,
                collision=True,
                coacd_options=gs.options.CoacdOptions(preprocess_resolution=150),
                file_meshes_are_zup=True,  # No Y-up to Z-up conversion (restore old behavior)
            ),
            material=gs.materials.Rigid(friction=0.8),
            surface=surface,
        )
        self._entities["table"] = {
            "entity": table,
            "size": self.TABLE_SIZE,
        }

    @abstractmethod
    def _add_custom_entities(self) -> None:
        """Add task-specific entities to the scene.

        Subclasses should override this to add cubes, tools, etc.
        Store entities in self._entities dict for later access.
        """
        pass

    @property
    @abstractmethod
    def placement_groups(self) -> PlacementGroups:
        """Names of entities to randomize on reset.

        Supports nested tuples for grouped placement where objects maintain
        relative positions during randomization.

        Examples:
            ("a", "b", "c")  # Place a, b, c independently
            (("a", "b"), "c")  # Place a, b together (relative positions), c independently
            ("a", ("b", "c", "d"))  # a independent, b/c/d grouped together

        Returns:
            Tuple of entity names or nested tuples, or None for no randomization.
        """
        pass

    def placement_mentioned_entities(self) -> frozenset[str]:
        """All entity names that appear in ``placement_groups`` (flattened).

        Rigid / MPM / SPH bodies not in this set are never passed to random
        placement helpers; subclasses should gate any extra motion on this too.
        """
        groups = self.placement_groups
        if groups is None:
            return frozenset()
        names: set[str] = set()
        for item in groups:
            if isinstance(item, tuple):
                names.update(item)
            else:
                names.add(item)
        return frozenset(names)

    def _capture_initial_entity_poses(self) -> None:
        """Capture and store initial poses from entity morph definitions.

        Must be called after scene.build() to ensure entities have their morph positions.
        Stores initial_pos and initial_quat in each entity dict for use during placement.
        Particle entities (MPM) do not support get_pos/get_quat and are skipped.
        """
        for _name, entity_info in self._entities.items():
            entity = entity_info["entity"]
            if _utils.is_particle_entity(entity):
                # Particle entities don't support get_pos/get_quat; their state is
                # restored by scene.reset() and they are not repositioned via set_pos.
                continue
            # Get the position/quat that was set in the morph definition
            pos = entity.get_pos()[0].cpu().numpy()
            quat = entity.get_quat()[0].cpu().numpy()
            entity_info["initial_pos"] = pos.copy()
            entity_info["initial_quat"] = quat.copy()
            # Always compute size from AABB so hardcoded values are never needed
            aabb = _utils.get_AABB(entity, self._device).cpu().numpy()[0]
            entity_info["size"] = tuple(float(aabb[1, i] - aabb[0, i]) for i in range(3))

    def _capture_spawn_poses(self) -> None:
        """Capture entity positions after placement and settling.

        Updates spawn_pos in each entity dict to reflect the actual randomized
        position at the start of the episode. Use spawn_pos (not initial_pos)
        in reward functions that need the per-episode starting position.
        """
        for _name, entity_info in self._entities.items():
            entity = entity_info["entity"]
            if _utils.is_particle_entity(entity):
                continue
            entity_info["spawn_pos"] = entity.get_pos()[0].cpu().numpy().copy()

    def _load_eval_dataset(self) -> None:
        """Load evaluation dataset from eval_dataset_json path if provided."""
        if self._eval_dataset_json is None:
            return

        json_path = self._eval_dataset_json
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"eval_dataset_json not found: {json_path}")
        with open(json_path) as f:
            raw = json.load(f)
        # Normalise list format: [{obj: info}, ...] -> {"0": {"objs_info": {...}}, ...}
        if isinstance(raw, list):
            raw = {str(i): {"objs_info": entry} for i, entry in enumerate(raw)}
        self._eval_dataset = raw
        self._eval_dataset_keys = list(self._eval_dataset.keys())
        self._eval_dataset_idx = 0
        logger.info(f"Loaded eval dataset with {len(self._eval_dataset_keys)} entries from {json_path}")

        # Load companion particle positions file if present ({stem}_particles.npz)
        from pathlib import Path as _Path

        particles_path = _Path(json_path).with_name(_Path(json_path).stem + "_particles.npz")
        if particles_path.exists():
            loaded = np.load(str(particles_path))
            self._eval_particles = {k: loaded[k] for k in loaded.files}
            logger.info(f"Loaded particle data for {list(self._eval_particles.keys())} from {particles_path}")

    def set_eval_dataset_idx(self, idx: int) -> None:
        """Set the eval dataset index (for resuming evaluation from a specific episode).

        Args:
            idx: The index to set. Will be wrapped to dataset size.
        """
        if self._eval_dataset is not None and len(self._eval_dataset_keys) > 0:
            self._eval_dataset_idx = idx % len(self._eval_dataset_keys)
            logger.info(f"Set eval dataset index to {self._eval_dataset_idx} (requested: {idx})")

    def _get_next_eval_data(self) -> dict[str, Any] | None:
        """Get the next evaluation data entry (cycles through the dataset).

        Returns:
            The objs_info dict for the current entry, or None if no dataset loaded.
        """
        if self._eval_dataset is None or len(self._eval_dataset_keys) == 0:
            return None

        key = self._eval_dataset_keys[self._eval_dataset_idx]
        entry = self._eval_dataset[key]

        # Advance to next entry (cycle back to beginning)
        self._eval_dataset_idx = (self._eval_dataset_idx + 1) % len(self._eval_dataset_keys)

        return entry.get("objs_info", None)

    def _init(self) -> None:
        """Initialize buffers and tracking variables (for single environment)."""
        self._random = np.random.default_rng()

        # Episode tracking (scalar values now)
        termination_cfg = self._config.get("termination", {})
        self._max_episode_steps = termination_cfg.get("max_episode_steps", 200)
        self._step_count = 0  # Scalar int instead of tensor
        self._success = torch.tensor(False, dtype=torch.bool, device=self._device)  # Scalar tensor

        # Action buffer - size depends on control mode (still batched for Genesis internal use)
        action_dim = 14 if self._control_mode in ("EE_DELTA", "EE_ABS") else self.ROBOT_DOF  # JOINT_ABS, JOINT_DELTA
        self.action_buf = torch.zeros((1, action_dim), device=self._device)  # (1, action_dim) for Genesis

    def _restore_eval_particles(self) -> None:
        """Restore canonical particle positions from the companion .npz file (no-op if none loaded)."""
        for entity_name, positions in self._eval_particles.items():
            if entity_name not in self._entities:
                continue
            poss = torch.tensor(positions, dtype=torch.float32, device=self._device)
            self._entities[entity_name]["entity"].set_particles_pos(poss)

    def _place_objects(self) -> None:
        """Place objects: uses eval dataset positions if available, otherwise random placement.

        Each entity in ``placement_groups`` gets a random pose (yaw randomized ±0.53 rad,
        position uniform in its reachable area, oriented-hull collision-free
        against already-placed
        objects AND against fixed entities not in any placement group). The
        whole pass is wrapped in a bounded retry loop; on persistent failure a
        :class:`PlacementError` is raised.
        """
        from gs_gym.common.utils.math_utils import euler_to_quat
        from gs_gym.envs.robowits.placement import (
            PlacementError,
            get_object_reachable_area,
            get_oriented_hull_2d,
            parse_reachable_bounds,
            place_group_together,
            place_single_object,
        )

        # Eval dataset placement
        objs_info = self._get_next_eval_data()
        if objs_info is not None:
            for obj_name, obj_info in objs_info.items():
                if obj_name not in self._entities:
                    continue
                entity = self._entities[obj_name]["entity"]
                if "pos" in obj_info:
                    entity.set_pos(torch.tensor(obj_info["pos"], dtype=torch.float32, device=self._device).unsqueeze(0))
                if "quat" in obj_info:
                    entity.set_quat(
                        torch.tensor(obj_info["quat"], dtype=torch.float32, device=self._device).unsqueeze(0)
                    )
                elif "euler" in obj_info:
                    quat = euler_to_quat(torch.tensor(obj_info["euler"], dtype=torch.float32, device=self._device))
                    entity.set_quat(quat.unsqueeze(0))
            self._restore_eval_particles()
            return

        # Random placement
        groups = self.placement_groups
        if groups is None:
            self._restore_eval_particles()
            return

        default_bounds = parse_reachable_bounds(self._config.get("reachable_bounds", self.DEFAULT_REACHABLE_BOUNDS))
        rotation_range = (0.0, 0.0, 0.53)

        # Names of entities that will be (re-)placed by this pass, so we can
        # collect collision hulls for everything else (fixed targets, decor)
        mentioned: set[str] = set()
        for group in groups:
            if isinstance(group, tuple):
                mentioned.update(group)
            else:
                mentioned.add(group)

        # Entities that should be excluded from the fixed-hull collision set
        # even though they are fixed and not in any placement group. Typical
        # use: support surfaces (``small table``) or landmarks that overlap a
        # randomized object's reachable area and would otherwise block every
        # candidate position. The big ``table`` is always excluded for this
        # reason (every object rests on it).
        ignored_hulls = set(getattr(self, "placement_ignored_hulls", ()) or ())
        ignored_hulls.add("table")

        def _build_fixed_hulls() -> list:
            hulls: list = []
            for name, info in self._entities.items():
                if name in ignored_hulls or name in mentioned:
                    continue
                entity = info.get("entity")
                if entity is None or _utils.is_particle_entity(entity):
                    continue
                pos_init = info.get("initial_pos")
                quat_init = info.get("initial_quat")
                size = info.get("size")
                if pos_init is None or quat_init is None or size is None:
                    continue
                hulls.append(get_oriented_hull_2d(pos_init, size, quat_init))
            extra = getattr(self, "placement_extra_fixed_hulls", None)
            if callable(extra):
                hulls.extend(extra())
            return hulls

        # Optional debug capture: when ``GS_GYM_PLACEMENT_DEBUG_DIR`` is set,
        # record per-attempt placement snapshots on failure and persist them
        # (together with env metadata) as JSON for offline inspection via
        # ``scripts/robowits/helpers/visualize_placement_debug.py``.
        #
        # Additional env vars to aid debugging:
        # - ``GS_GYM_PLACEMENT_DEBUG_EAGER=1``: dump a snapshot per failed
        #   attempt instead of waiting for all retries to fail.
        # - ``GS_GYM_PLACEMENT_DEBUG_RENDER=1``: also render a PNG inline using
        #   the visualizer module (requires matplotlib).
        # - ``GS_GYM_PLACEMENT_VERBOSE=1``: log the bounds / footprint summary
        #   on each retry, on top of the PlacementError message.
        import os

        debug_dir = os.environ.get("GS_GYM_PLACEMENT_DEBUG_DIR")
        debug_eager = bool(int(os.environ.get("GS_GYM_PLACEMENT_DEBUG_EAGER", "0") or "0"))
        debug_render = bool(int(os.environ.get("GS_GYM_PLACEMENT_DEBUG_RENDER", "0") or "0"))
        verbose = bool(int(os.environ.get("GS_GYM_PLACEMENT_VERBOSE", "0") or "0"))
        debug_attempts: list[dict] = [] if debug_dir else []  # type: ignore[assignment]

        last_error: PlacementError | None = None
        for attempt in range(self.MAX_PLACEMENT_RETRIES):
            try:
                placed_hulls = _build_fixed_hulls()
                attempt_debug: dict | None = {} if debug_dir else None
                if verbose:
                    logger.info(
                        f"Placement attempt {attempt + 1}/{self.MAX_PLACEMENT_RETRIES}: "
                        f"default_bounds=x[{default_bounds[0][0]:.3f},{default_bounds[0][1]:.3f}] "
                        f"y[{default_bounds[1][0]:.3f},{default_bounds[1][1]:.3f}], "
                        f"groups={[list(g) if isinstance(g, tuple) else g for g in groups]}, "
                        f"fixed_hulls={len(placed_hulls)}"
                    )
                _entity_extra_hulls_fn = getattr(self, "placement_entity_extra_hulls", None)
                for group in groups:
                    # Per-entity extra hulls (e.g. arm exclusion for tall objects).
                    # Injected only for the current group so short entities are unaffected.
                    entity_names = list(group) if isinstance(group, tuple) else [group]
                    entity_extra: list = []
                    if callable(_entity_extra_hulls_fn):
                        for _n in entity_names:
                            entity_extra.extend(_entity_extra_hulls_fn(_n))

                    if entity_extra:
                        # Build a temporary hull list: entity_extra first, then placed_hulls.
                        # After place_*, any newly placed hull is appended at the end of
                        # effective_hulls. Sync it back into placed_hulls so later groups
                        # still see it, without permanently polluting placed_hulls with the
                        # arm exclusion hulls.
                        n_placed_before = len(placed_hulls)
                        effective_hulls: list = entity_extra + placed_hulls
                        hulls_arg = effective_hulls
                    else:
                        hulls_arg = placed_hulls

                    if isinstance(group, tuple):
                        place_group_together(
                            group,
                            self._entities,
                            default_bounds,
                            hulls_arg,
                            self._random,
                            self._device,
                            _utils.is_particle_entity,
                            rotation_range=rotation_range,
                            debug=attempt_debug,
                            ignored_hulls=ignored_hulls,
                            placement_target_names=mentioned,
                        )
                    else:
                        bounds = get_object_reachable_area(
                            group, default_bounds, getattr(self, "_object_reachable_areas", {})
                        )
                        if verbose:
                            logger.info(
                                f"  place_single_object('{group}'): "
                                f"bounds=x[{bounds[0][0]:.3f},{bounds[0][1]:.3f}] "
                                f"y[{bounds[1][0]:.3f},{bounds[1][1]:.3f}], "
                                f"size={self._entities.get(group, {}).get('size')}, "
                                f"placed_hulls={len(placed_hulls)}"
                            )
                        place_single_object(
                            group,
                            self._entities,
                            bounds,
                            hulls_arg,
                            self._random,
                            self._device,
                            _utils.is_particle_entity,
                            rotation_range=rotation_range,
                            debug=attempt_debug,
                            placement_target_names=mentioned,
                        )

                    if entity_extra:
                        # effective_hulls = entity_extra + old_placed_hulls + [new_hulls...]
                        # new_hulls start at index len(entity_extra) + n_placed_before
                        placed_hulls.extend(effective_hulls[len(entity_extra) + n_placed_before :])
                self._restore_eval_particles()
                return
            except PlacementError as e:
                last_error = e
                if debug_dir and attempt_debug is not None:
                    attempt_debug["attempt"] = attempt + 1
                    attempt_debug["error"] = str(e)
                    debug_attempts.append(attempt_debug)
                    if debug_eager:
                        # Dump a per-attempt snapshot immediately for fast iteration.
                        self._dump_placement_debug(
                            debug_dir,
                            [attempt_debug],
                            default_bounds,
                            groups,
                            suffix=f"_attempt{attempt + 1:03d}",
                            render=debug_render,
                        )
                logger.warning(
                    f"Random placement failed (attempt {attempt + 1}/{self.MAX_PLACEMENT_RETRIES}): {e}. Retrying..."
                )
                continue

        if debug_dir:
            self._dump_placement_debug(debug_dir, debug_attempts, default_bounds, groups, render=debug_render)

        logger.warning(
            f"Failed to find a valid random placement after {self.MAX_PLACEMENT_RETRIES} attempts "
            f"({last_error}). Falling back to initial poses."
        )
        for name in mentioned:
            info = self._entities.get(name)
            if info is None:
                continue
            entity = info.get("entity")
            initial_pos = info.get("initial_pos")
            initial_quat = info.get("initial_quat")
            if entity is None or initial_pos is None or initial_quat is None:
                continue
            entity.set_pos(torch.tensor(initial_pos, dtype=torch.float32, device=self._device).unsqueeze(0))
            entity.set_quat(torch.tensor(initial_quat, dtype=torch.float32, device=self._device).unsqueeze(0))
        self._restore_eval_particles()

    def _dump_placement_debug(
        self,
        debug_dir: str,
        attempts: list[dict],
        default_bounds: tuple[tuple[float, float], tuple[float, float]],
        groups,
        suffix: str = "",
        render: bool = False,
    ) -> None:
        """Save a placement failure snapshot JSON to ``debug_dir``.

        The snapshot includes per-attempt debug info (bounds, placed hulls,
        sampled grid of free space, etc.) plus enough env metadata to make
        the output self-contained for
        ``scripts/robowits/helpers/visualize_placement_debug.py``.

        Args:
            suffix: Optional filename suffix (e.g. ``_attempt001``) — useful
                when called per-attempt with ``GS_GYM_PLACEMENT_DEBUG_EAGER``.
            render: If True, also render a PNG alongside the JSON using the
                visualizer module (requires matplotlib).
        """
        import os
        import time

        os.makedirs(debug_dir, exist_ok=True)
        task_id = getattr(self, "_task_id", None) or self.__class__.__name__
        task_id = str(task_id).replace("/", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(debug_dir, f"{task_id}_{ts}{suffix}.json")

        entities_meta: dict[str, dict] = {}
        for name, info in self._entities.items():
            if name == "table":
                continue
            pos = info.get("initial_pos")
            quat = info.get("initial_quat")
            size = info.get("size")
            entities_meta[name] = {
                "size": list(size) if size is not None else None,
                "initial_pos": [float(v) for v in pos] if pos is not None else None,
                "initial_quat": [float(v) for v in quat] if quat is not None else None,
            }

        reachable_areas = getattr(self, "_object_reachable_areas", {})

        payload = {
            "task": task_id,
            "timestamp": ts,
            "default_bounds": [list(default_bounds[0]), list(default_bounds[1])],
            "placement_groups": [list(g) if isinstance(g, tuple) else g for g in groups],
            "object_reachable_areas": reachable_areas,
            "entities": entities_meta,
            "attempts": attempts,
        }

        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.error(f"Saved placement-debug snapshot to {path}")
        except OSError as e:
            logger.error(f"Failed to save placement-debug snapshot: {e}")
            return

        if render:
            png_path = path.replace(".json", ".png")
            try:
                from scripts.robowits.helpers.visualize_placement_debug import render_snapshot
            except Exception:
                # Fall back to the path-based import (script dir not on sys.path).
                try:
                    import importlib.util
                    import pathlib

                    repo_root = pathlib.Path(__file__).resolve().parents[3]
                    vis_path = repo_root / "scripts" / "robowits" / "helpers" / "visualize_placement_debug.py"
                    spec = importlib.util.spec_from_file_location("_placement_debug_vis", vis_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"cannot load {vis_path}")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    render_snapshot = mod.render_snapshot  # type: ignore[attr-defined]
                except Exception as e:
                    logger.error(f"Failed to import placement visualizer: {e}")
                    return
            try:
                from pathlib import Path as _Path

                render_snapshot(payload, _Path(png_path))
                logger.error(f"Rendered placement-debug PNG to {png_path}")
            except Exception as e:
                logger.error(f"Failed to render placement-debug PNG: {e}")

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        self._random = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def reset_env(self) -> None:
        """Reset the environment (single environment with Genesis n_envs=1)."""
        envs_idx = torch.tensor([0], device=self._device, dtype=torch.int32)

        self._scene.reset(envs_idx)
        self._robot.reset(envs_idx)
        self._place_objects()

        for _ in range(self._preprocess_steps):
            self._robot.hold_reset_position()
            self._scene.scene.step()

        self._step_count = 0
        self._success = torch.tensor(False, dtype=torch.bool, device=self._device)
        self._capture_spawn_poses()

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to the environment.

        Args:
            action: Single action tensor (action_dim,) - will be expanded to (1, action_dim) for Genesis

        Action format depends on control_mode:
            - JOINT_ABS: 16D absolute joint positions [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
            - JOINT_DELTA: 16D delta joint positions (added to current)
            - EE_ABS: 14D absolute EE poses [R_pos(3), L_pos(3), R_aa(3), L_aa(3), R_grip(1), L_grip(1)]
            - EE_DELTA: 14D delta EE control (robot handles scaling)
        """
        # Expand action to (1, action_dim) for Genesis internal use
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (action_dim,) -> (1, action_dim)

        # Delegate all action processing to robot
        self._robot.apply_action(action)
        self.action_buf = action

        # Step the simulation
        self._scene.scene.step()
        self._step_count += 1

    def get_observations(self) -> dict:
        """Get current observations in format compatible with LeRobot's preprocess_observation.

        Returns:
            dict: Observations with:
                - "agent_pos": robot state (dimension depends on observation_mode)
                    - EE: 14D, JOINT: 16D
                - "pixels": dict of (H, W, 3) camera images
        """
        robot = self._robot
        dofs_raw = robot.robot_entity.get_dofs_position()
        dofs = robot._dof_to_sequential(dofs_raw)

        right_gripper = dofs[:, 7:8] + dofs[:, 8:9]
        left_gripper = dofs[:, 16:17] + dofs[:, 17:18]

        # EE positions from Genesis are in world frame; training data uses robot-base-relative
        # frame. Subtract robot base position so observations match the training distribution.
        robot_base_pos = torch.tensor(robot.args.position, dtype=torch.float32, device=self._device).unsqueeze(0)

        right_joints = dofs[:, :7]
        left_joints = dofs[:, 9:16]
        right_ee_pos = robot.ee_link_right.get_pos() - robot_base_pos
        left_ee_pos = robot.ee_link_left.get_pos() - robot_base_pos
        right_ee_aa = _utils.quat_to_axis_angle(robot.ee_link_right.get_quat())
        left_ee_aa = _utils.quat_to_axis_angle(robot.ee_link_left.get_quat())

        state_joint = torch.cat([right_joints, right_gripper, left_joints, left_gripper], dim=-1)
        state_ee = torch.cat([right_ee_pos, left_ee_pos, right_ee_aa, left_ee_aa, right_gripper, left_gripper], dim=-1)

        state = state_joint if self._observation_mode == "JOINT" else state_ee

        obs = {
            "agent_pos": state[0].cpu().numpy(),
            "agent_pos_joint": state_joint[0].cpu().numpy(),
            "agent_pos_ee": state_ee[0].cpu().numpy(),
        }

        pixels = {}
        for cam_name in ["ego", "wrist_right", "wrist_left"]:
            camera = self._cameras.get(cam_name)
            if camera is not None:
                data = camera.read()
                frame = data.rgb
                if frame.dtype != torch.uint8:
                    frame = (frame * 255).to(torch.uint8)
                pixels[cam_name] = frame[0].cpu().numpy()
            else:
                raise RuntimeError(f"Camera '{cam_name}' not available. Available: {list(self._cameras.keys())}")

        obs["pixels"] = pixels
        return obs

    def _get_entity_state(self, entity) -> tuple[torch.Tensor, torch.Tensor]:
        """Get position and quaternion for an entity, handling both rigid and particle.

        For rigid entities, returns (pos, quat) with shape (n_envs, 3) and (n_envs, 4).
        For particle entities (MPMEntity), returns centroid position and identity quaternion.

        Args:
            entity: Genesis RigidEntity or MPMEntity

        Returns:
            Tuple of (pos, quat) tensors
        """
        # Check if this is a particle entity
        if _utils.is_particle_entity(entity):
            # For particle entities, return centroid and identity quaternion
            if hasattr(entity, "get_particles_pos"):
                particles_pos = entity.get_particles_pos()
                # Compute centroid - handle (n_envs, n_particles, 3) or (n_particles, 3)
                if particles_pos.ndim == 3:
                    centroid = particles_pos.mean(dim=1)  # (n_envs, 3)
                else:
                    centroid = particles_pos.mean(dim=0, keepdim=True)  # (1, 3)
                    centroid = centroid  # Already (1, 3)
            else:
                # Fallback: return zeros
                centroid = torch.zeros(1, 3, device=self._device)

            # Identity quaternion (wxyz)
            identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self._device)

            return centroid, identity_quat
        else:
            # Rigid entity - use standard get_pos/get_quat
            pos = entity.get_pos()
            quat = entity.get_quat()
            return pos, quat

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get reward. Override in subclass for task-specific rewards.

        Returns:
            reward: Scalar tensor (shape: ())
            reward_dict: Dictionary of reward components
        """
        # Default: zero reward (scalar)
        reward = torch.tensor(0.0, device=self._device)
        return reward, {}

    def get_terminated(self) -> torch.Tensor:
        """Check if episode is terminated based on task success.

        Returns:
            terminated: Scalar boolean tensor
        """
        self._success = self._check_success().squeeze()
        return self._success

    def get_truncated(self) -> torch.Tensor:
        """Check if episode is truncated (max steps).

        Returns:
            truncated: Scalar boolean tensor
        """
        # _step_count is now a scalar int, comparison returns scalar bool
        return torch.tensor(self._step_count >= self._max_episode_steps, dtype=torch.bool, device=self._device)

    def get_extra_infos(self) -> dict[str, Any]:
        """Get extra information.

        Returns:
            info: Dictionary with scalar values (no batch dimension)
        """
        return {
            "is_success": bool(self._success.item()),
            "step_count": int(self._step_count),
        }

    def get_entity(self, name: str):
        """Get entity by name."""
        return self._entities.get(name, {}).get("entity")

    # Properties
    @property
    def n_envs(self) -> int:
        """Number of environments."""
        return self._n_envs

    @property
    def control_mode(self) -> str:
        """Current control mode."""
        return self._control_mode

    @property
    def observation_mode(self) -> str:
        """Current observation mode (EE or JOINT)."""
        return self._observation_mode

    @property
    def robot(self) -> BimanualMarvinRobot:
        """Robot instance."""
        return self._robot

    @property
    def cameras(self) -> dict[str, Any]:
        """All available cameras (ego, wrist_right, wrist_left)."""
        return self._cameras

    def render(self) -> np.ndarray | None:
        """Render the environment for video recording.

        Returns:
            RGB image as numpy array with shape (H, W, 3), dtype uint8,
            or None if no camera is available.
        """
        return self.render_frame(env_idx=0, camera_name="ego")

    def render_frame(self, env_idx: int = 0, camera_name: str = "ego") -> np.ndarray:
        """Render a frame from a camera sensor.

        Args:
            env_idx: Environment index to render (default 0)
            camera_name: Camera to render from ("ego", "wrist_right", "wrist_left")

        Returns:
            RGB image as numpy array with shape (H, W, 3), dtype uint8
        """
        camera = self._cameras.get(camera_name, self._cameras.get("ego"))
        if camera is None:
            raise RuntimeError(f"Camera '{camera_name}' not available. Available: {list(self._cameras.keys())}")

        data = camera.read()
        # data.rgb has shape (n_envs, H, W, 3)
        frame = data.rgb[env_idx].cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        return frame

    def collect_objs_info(self, names: list[str] | None = None, env_idx: int = 0) -> dict[str, dict[str, Any]]:
        """Collect object info squeezed to a single env for use in _check_success().

        Returned shapes:
        - rigid:    pos (3,), vel (3,), euler (3,), bounds (2, 3), convex_hull_2d ndarray or None
        - particle: pos (n_particles, 3), vel (n_particles, 3), bounds (2, 3)
        """
        from genesis.engine.entities.particle_entity import ParticleEntity
        from genesis.engine.entities.rigid_entity import RigidEntity

        if names is None:
            names = list(self._entities.keys())

        objs_info: dict[str, dict[str, Any]] = {}
        for name in names:
            entity = self.get_entity(name)
            if entity is None:
                continue

            try:
                if isinstance(entity, RigidEntity):
                    bounds = _utils.get_AABB(entity, self._device).cpu().numpy()[env_idx]
                    quat = entity.get_quat()[env_idx]
                    euler = _utils.quat_to_euler(quat).cpu().numpy()
                    convex_hull_2d = _utils.compute_2d_convex_hull(entity, env_idx=env_idx)
                    objs_info[name] = {
                        "material": "rigid",
                        "pos": entity.get_pos().cpu().numpy()[env_idx],
                        "vel": entity.get_vel().cpu().numpy()[env_idx],
                        "euler": euler,
                        "bounds": bounds,
                        "convex_hull_2d": convex_hull_2d,
                    }
                elif isinstance(entity, ParticleEntity):
                    pos = entity.get_particles_pos().detach().cpu().numpy()[env_idx]
                    vel = entity.get_particles_vel().detach().cpu().numpy()[env_idx]
                    bounds = np.array([pos.min(axis=0), pos.max(axis=0)]) if len(pos) > 0 else np.zeros((2, 3))
                    objs_info[name] = {
                        "material": "particle",
                        "pos": pos,
                        "vel": vel,
                        "bounds": bounds,
                    }
                else:
                    # Fallback: try particle-style access (MPM entities not matching ParticleEntity)
                    pos = entity.get_particles_pos().detach().cpu().numpy()[env_idx]
                    vel = entity.get_particles_vel().detach().cpu().numpy()[env_idx]
                    bounds = np.array([pos.min(axis=0), pos.max(axis=0)]) if len(pos) > 0 else np.zeros((2, 3))
                    objs_info[name] = {
                        "material": "particle",
                        "pos": pos,
                        "vel": vel,
                        "bounds": bounds,
                    }
            except Exception:
                pass  # Skip entities that fail (e.g. fixed markers with no dynamics)

        return objs_info

    def _apply_config_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply config overrides via deep merge.

        Args:
            overrides: Dictionary of config values to override
        """

        def deep_merge(base: dict, override: dict) -> dict:
            """Recursively merge override into base."""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        deep_merge(self._config, overrides)

    def _config_to_env_args(self, config: dict):
        """Convert config dict to EnvArgs object."""
        from gs_gym.envs.schema import EnvArgs
        from gs_gym.robots.schema import ManipulatorRobotArgs
        from gs_gym.scenes.schema import SceneArgs

        return EnvArgs(
            scene_args=SceneArgs(**config.get("scene", {})),
            robot_args=ManipulatorRobotArgs(**config.get("robot", {})),
        )

    def _get_spaces_from_config(self, config: dict):
        """Get space information from config."""
        # Action space depends on control mode:
        # EE_DELTA / EE_ABS: 14-DOF [right: 3 pos + 3 rot + 1 gripper, left: same]
        # JOINT_ABS / JOINT_DELTA: 16-DOF [right: 7 arm + 1 gripper, left: same]
        if self._control_mode in ("EE_DELTA", "EE_ABS"):
            action_dim = 14
            if self._control_mode == "EE_DELTA":
                # lo=-1.0, hi=1.0
                action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
            else:
                # EE_ABS layout: [right_pos(3), left_pos(3), right_aa(3), left_aa(3), right_grip(1), left_grip(1)]
                # pos: lo=[-0.5, -0.8, 0.0], hi=[0.8, 0.8, 0.8]; aa: lo=-pi, hi=pi; grip: lo=0.0, hi=0.1
                action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        elif self._control_mode in ("JOINT_ABS", "JOINT_DELTA"):
            # JOINT_ABS/JOINT_DELTA layout: [right_joints(7), right_grip(1), left_joints(7), left_grip(1)]
            # arm: lo=[-3.1067, -1.8, -3.1067, -2.5307, -3.1067, -1.047, -1.047], hi=[3.1067, 2.0944, 3.1067, -0.08727, 3.1067, 1.047, 1.047]; grip: lo=0.0, hi=0.1
            action_dim = self.ROBOT_DOF
            action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        else:
            raise ValueError(
                f"Unknown control_mode '{self._control_mode}'. Available: JOINT_ABS, JOINT_DELTA, EE_ABS, EE_DELTA"
            )

        # State dimension depends on observation_mode
        state_dim = 16 if self._observation_mode == "JOINT" else 14

        observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
                "agent_pos_joint": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
                "agent_pos_ee": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
                "pixels": gym.spaces.Dict(
                    {
                        "ego": gym.spaces.Box(low=0, high=255, shape=(480, 848, 3), dtype=np.uint8),
                        "wrist_right": gym.spaces.Box(low=0, high=255, shape=(480, 848, 3), dtype=np.uint8),
                        "wrist_left": gym.spaces.Box(low=0, high=255, shape=(480, 848, 3), dtype=np.uint8),
                    }
                ),
            }
        )

        return action_space, observation_space
