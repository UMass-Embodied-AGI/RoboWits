#!/usr/bin/env python
"""Evaluate the VLM oracle-primitive planner baseline on RoboWits tasks.

A VLM plans a sequence of primitives (pick, place, push, ...) from camera images
and object state, and an oracle executor applies each primitive directly to the
Genesis entities. `--mode closed` replans after every primitive; `--mode open`
plans the whole sequence up front, isolating the value of closed-loop feedback.

Unlike scripts/robowits/eval/vlm/controller.py, the primitives here teleport
objects rather than driving the arms through IK, so this measures planning in
isolation from control.

Usage:
    OPENAI_API_KEY=... python scripts/robowits/eval/vlm/planner.py --task 02
    python scripts/robowits/eval/vlm/planner.py --suite robowits_10 --num-episodes 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import trimesh
from numpy.typing import NDArray
from PIL import Image

from gs_gym.common.utils import np_math_utils as npm
from gs_gym.envs.registry import TASK_REGISTRY

# summarize_results.py lives in the parent eval/ directory and is not an
# installed module: it owns the suite definitions and the eval_info.json writers,
# shared with controller.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# isort: off
from summarize_results import (  # noqa: E402
    SUITES,
    dataset_file,
    resolve_suite,
    resolve_task,
    variant_stem,
    variants_with_dataset,
    write_eval_info,
    write_grouped_eval_info,
)
# isort: on


# ==============================================================================
# Oracle primitive evaluator
# ==============================================================================


# The VLM baselines render with Nyx unless GS_GYM_CAMERA_TYPE says otherwise (RoboWitsEnv reads it); Nyx runs on
# the CUDA backend, which "auto" picks for it.
os.environ.setdefault("GS_GYM_CAMERA_TYPE", "nyx")


class OracleControlEvaluator:
    """Evaluator for oracle primitive controls on robowits environments."""

    # End effector link names
    RIGHT_HAND_LINK = "Gripper_Tip_R"
    LEFT_HAND_LINK = "Gripper_Tip_L"

    # Simulation stepping parameters
    N_MAX_STEP = 100
    STABLE_THRESHOLD_PER_STEP = 1e-4  # Max position change (meters) to consider stable

    # Max position error (meters) for an IK solution to count as reaching the target
    IK_POSITION_TOLERANCE = 0.01

    # DOFs control parameters
    DOFS_KP = 10000.0
    DOFS_KV = 10000.0
    DOFS_FORCE_RANGE = 10000.0

    def __init__(self) -> None:
        """Initialize the oracle control evaluator."""
        super().__init__()
        self.scene: Any = None
        self.entities: dict[str, Any] = {}
        self.robot: Any = None

        # Cameras
        self.ego_cam: Any = None
        self.wrist_left_cam: Any = None
        self.wrist_right_cam: Any = None
        self.gallery_cam: Any = None

        # Track which objects are grasped by which arm
        self._grasped_objects: dict[str, str] = {}  # obj_name -> arm ("left" or "right")

        # Track which objects are being held at raised position
        self._held_objects: set[str] = set()

    def set_env_from_task(
        self,
        task_name: str,
        show_viewer: bool = False,
        eval_dataset_json: str | None = None,
        genesis_backend: str = "cpu",
    ) -> None:
        """Set up the environment by instantiating a registered gs-gym task.

        Args:
            task_name: Registered gs-gym task id (e.g. "robowits/02-retrieve-cube-v0").
            show_viewer: Whether to show the Genesis viewer.
            eval_dataset_json: Optional path to the eval dataset JSON holding the
                per-episode initial object placements, as used by the policy evals.
            genesis_backend: Genesis physics backend ("cpu" or "cuda").
        """
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown gs-gym task: '{task_name}'")

        # Instantiate the task class directly rather than through gs_gym.make():
        # the primitives drive a single scene through raw Genesis calls, so the
        # SyncVectorEnv wrapper would only get in the way. Genesis itself is
        # initialised inside GenesisGymEnv.__init__.
        env_class = TASK_REGISTRY[task_name]
        self._env = env_class(
            show_viewer=show_viewer,
            control_mode="EE_ABS",
            observation_mode="EE",
            eval_dataset_json=eval_dataset_json,
            genesis_backend=genesis_backend,
        )
        self._env.reset_env()

        # Genesis scene + robot handles
        self.scene = self._env._scene.scene
        self._robot = self._env.robot
        self.robot = self._robot.robot_entity

        # Scene entities (the robot is included so collision pairs can name it)
        self.entities = {"robot": self.robot}
        for name in self._env._entities:
            entity = self._env.get_entity(name)
            if entity is not None:
                self.entities[name] = entity

        # Cameras are Genesis sensors, read via read().
        self.ego_cam = self._env.cameras.get("ego")
        self.wrist_left_cam = self._env.cameras.get("wrist_left")
        self.wrist_right_cam = self._env.cameras.get("wrist_right")

        # Clear grasped objects tracking
        self._grasped_objects.clear()
        self._held_objects.clear()

        # Build entity index to name mapping
        self.entity_idx2name: dict[int, str] = {}
        for name, entity in self.entities.items():
            self.entity_idx2name[entity.idx] = name

        print(f"Loaded gs-gym task: {task_name}")
        print("entities:", self.entities.keys())

        # Setup dofs control for all rigid objects
        self._setup_dofs_control()

        # Keep all objects at their current position by default
        self._control_all_objects_position()

        self.STABLE_THRESHOLD_PER_STEP = 0.3 * 9.8 * self.scene.dt**2
        self.N_MAX_STEP = int(5 / self.scene.dt)
        self.N_STABLE_STEPS = int(0.5 / self.scene.dt)
        print("STABLE_THRESHOLD_PER_STEP:", self.STABLE_THRESHOLD_PER_STEP)
        print("N_MAX_STEP:", self.N_MAX_STEP)

    def is_task_completed(self) -> bool:
        """Check whether the task's success condition currently holds."""
        return bool(self._env.get_terminated().item())

    def reset_episode(self, episode_idx: int = 0) -> None:
        """Reset the environment for a new episode.

        Args:
            episode_idx: Episode index, used to select the eval dataset entry so
                episode N starts from the same scene as every other policy eval.
        """
        self._env.set_eval_dataset_idx(episode_idx)
        self._env.reset_env()

        # Clear grasped objects tracking
        self._grasped_objects.clear()
        self._held_objects.clear()

        # Re-setup dofs control
        self._setup_dofs_control()
        self._control_all_objects_position()

    def collect_objs_info(self) -> dict[str, dict[str, Any]]:
        """Gather information about all entities in the scene.

        Delegates to RoboWitsEnv.collect_objs_info, which returns the same
        per-object fields (pos, vel, bounds, euler, convex_hull_2d) the
        primitives and the VLM prompt expect.

        Returns:
            Dictionary mapping object names to their info.
        """
        return self._env.collect_objs_info()

    def _get_AABB(self, entity: Any) -> NDArray[np.floating[Any]]:
        """Get axis-aligned bounding box for an entity.

        Args:
            entity: The RigidEntity to get AABB for.

        Returns:
            Array of shape [2, 3] with [min_corner, max_corner].
        """
        from genesis.options.morphs import Box, Cylinder, Sphere

        if entity.n_geoms > 0:
            aabb = entity.get_AABB()
            aabb_np = aabb.cpu().numpy()
            # Squeeze batch dimension if present: [1, 2, 3] -> [2, 3]
            if aabb_np.ndim == 3:
                aabb_np = aabb_np[0]
            return aabb_np  # [2, 3]

        # Fallback for entities without collision geometry
        pos = entity.get_pos().cpu().numpy().flatten()
        cx, cy, cz = pos[0], pos[1], pos[2]

        if isinstance(entity.morph, Box):
            size = entity.morph.size
            if size is None:
                raise ValueError("Box morph has no size")
            sx, sy, sz = size
            return np.array(
                [
                    [cx - sx / 2, cy - sy / 2, cz - sz / 2],
                    [cx + sx / 2, cy + sy / 2, cz + sz / 2],
                ]
            )

        if isinstance(entity.morph, Cylinder):
            r = entity.morph.radius
            h = entity.morph.height / 2
            return np.array(
                [
                    [cx - r, cy - r, cz - h],
                    [cx + r, cy + r, cz + h],
                ]
            )

        if isinstance(entity.morph, Sphere):
            r = entity.morph.radius
            return np.array(
                [
                    [cx - r, cy - r, cz - r],
                    [cx + r, cy + r, cz + r],
                ]
            )

        raise ValueError(f"Cannot compute AABB for morph type: {type(entity.morph)}")

    def _get_mesh_vertices(self, entity: Any) -> NDArray[np.floating[Any]] | None:
        """Get mesh vertices of an entity in world coordinates.

        Args:
            entity: The RigidEntity to get vertices for.

        Returns:
            Array of vertices in world coordinates, or None if not available.
        """
        from genesis.options.morphs import Box, Cylinder, Sphere
        from trimesh.transformations import quaternion_matrix

        pos = entity.get_pos().cpu().numpy()[0]  # Shape: [3]
        quat = entity.get_quat().cpu().numpy()[0]  # Shape: [4]

        T = quaternion_matrix(quat)
        T[:3, 3] = pos

        vertices_3d = None

        # Try to get vertices from geoms
        if entity.n_geoms > 0 and hasattr(entity, "geoms"):
            try:
                meshes = []
                for geom in entity.geoms:
                    mesh = geom.get_trimesh()
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(T)
                    meshes.append(mesh_copy)
                if meshes:
                    combined_mesh = trimesh.util.concatenate(meshes)
                    vertices_3d = combined_mesh.vertices
            except Exception:
                pass

        # Fallback to morph-based vertices
        if vertices_3d is None:
            if isinstance(entity.morph, Box):
                morph_size = entity.morph.size
                if morph_size is None:
                    return None
                size = np.array(morph_size)
                corners_local = np.array(
                    [
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [size[0] / 2, size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, size[2] / 2],
                        [size[0] / 2, -size[1] / 2, size[2] / 2],
                        [size[0] / 2, size[1] / 2, size[2] / 2],
                        [-size[0] / 2, size[1] / 2, size[2] / 2],
                    ]
                )
                vertices_3d = trimesh.transformations.transform_points(corners_local, T)

            elif isinstance(entity.morph, Cylinder):
                radius = entity.morph.radius
                height = entity.morph.height
                n_points = 16
                angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

                bottom_circle = np.column_stack(
                    [
                        radius * np.cos(angles),
                        radius * np.sin(angles),
                        np.full(n_points, -height / 2),
                    ]
                )
                top_circle = np.column_stack(
                    [
                        radius * np.cos(angles),
                        radius * np.sin(angles),
                        np.full(n_points, height / 2),
                    ]
                )
                points_local = np.vstack([bottom_circle, top_circle])
                vertices_3d = trimesh.transformations.transform_points(points_local, T)

            elif isinstance(entity.morph, Sphere):
                radius = entity.morph.radius
                n_points = 32
                indices = np.arange(0, n_points, dtype=np.float32)
                theta = np.arccos(1 - 2 * indices / (n_points - 1)) if n_points > 1 else np.array([0])
                phi = np.pi * (1 + np.sqrt(5)) * indices

                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                points_local = radius * np.column_stack([x, y, z])
                vertices_3d = trimesh.transformations.transform_points(points_local, T)

        return vertices_3d

    def _get_ee_link(self, which_arm: Literal["left", "right"]) -> Any:
        """Get the end effector link for the specified arm."""
        link_name = self.LEFT_HAND_LINK if which_arm == "left" else self.RIGHT_HAND_LINK
        return self.robot.get_link(link_name)

    def _setup_dofs_control(self) -> None:
        """Set up DOFs control parameters (kp, kv, force_range) for all rigid objects."""
        from genesis.engine.entities.rigid_entity import RigidEntity

        for name, entity in self.entities.items():
            if name in ("robot", "table", "floor", "wall_front"):
                continue
            if isinstance(entity, RigidEntity):
                try:
                    n_dofs = entity.n_dofs
                    if n_dofs > 0:
                        entity.set_dofs_kp([self.DOFS_KP] * n_dofs)
                        entity.set_dofs_kv([self.DOFS_KV] * n_dofs)
                        entity.set_dofs_force_range(
                            lower=[-self.DOFS_FORCE_RANGE] * n_dofs,
                            upper=[self.DOFS_FORCE_RANGE] * n_dofs,
                        )
                except Exception:
                    pass

    def _control_all_objects_position(self, exclude: set[str] | None = None) -> None:
        """Keep all objects at their current position using control_dofs_position.

        Args:
            exclude: Set of object names to exclude from control.
        """
        from genesis.engine.entities.rigid_entity import RigidEntity

        if exclude is None:
            exclude = set()

        for name, entity in self.entities.items():
            if name in ("robot", "table", "floor", "wall_front"):
                continue
            if name in exclude:
                continue
            if isinstance(entity, RigidEntity):
                try:
                    if entity.n_dofs > 0:
                        current_dofs_pos = entity.get_dofs_position()
                        entity.control_dofs_position(current_dofs_pos)
                except Exception:
                    pass

    def _release_object_control(self, obj_name: str) -> None:
        """Release position control for a specific object.

        Args:
            obj_name: Name of the object to release control.
        """
        if obj_name in self._held_objects:
            self._held_objects.discard(obj_name)

    def _get_all_positions(self) -> dict[str, NDArray[np.floating[Any]]]:
        """Get current positions of all trackable entities.

        Returns:
            Dictionary mapping entity names to their positions.
        """
        from genesis.engine.entities.particle_entity import ParticleEntity
        from genesis.engine.entities.rigid_entity import RigidEntity

        positions: dict[str, NDArray[np.floating[Any]]] = {}

        for name, entity in self.entities.items():
            if name in ("robot", "table", "floor", "wall_front"):
                continue
            if isinstance(entity, RigidEntity):
                positions[name] = entity.get_pos().cpu().numpy().flatten()
            elif isinstance(entity, ParticleEntity):
                # Use mean position for particle entities
                positions[name] = entity.get_particles_pos().cpu().numpy().mean(axis=0)

        return positions

    def _step_until_stable(self) -> int:
        """Step the simulation until objects stabilize or max steps reached.

        Runs simulation steps until the last N_STABLE_STEPS steps all have
        movement within STABLE_THRESHOLD_PER_STEP, or N_MAX_STEP is reached.

        Returns:
            Number of steps taken.
        """
        from collections import deque

        N_STABLE_STEPS = self.N_STABLE_STEPS  # Last N steps must all be stable

        prev_positions = self._get_all_positions()

        # Keep track of max movements for the last N_STABLE_STEPS
        recent_movements: deque[float] = deque(maxlen=N_STABLE_STEPS)

        for step in range(self.N_MAX_STEP):
            self.scene.step()
            curr_positions = self._get_all_positions()

            # Check max movement this step
            max_movement = 0.0
            for name in prev_positions:
                if name in curr_positions:
                    movement = float(np.linalg.norm(curr_positions[name] - prev_positions[name]))
                    max_movement = max(max_movement, movement)

            recent_movements.append(max_movement)

            # Check if all of the last N_STABLE_STEPS are stable
            if len(recent_movements) >= N_STABLE_STEPS and all(
                m < self.STABLE_THRESHOLD_PER_STEP for m in recent_movements
            ):
                return step + 1

            prev_positions = curr_positions

        print("Unstable after", self.N_MAX_STEP, "steps")
        return self.N_MAX_STEP

    def _check_ik_collision(
        self,
        which_arm: Literal["left", "right"],
        target_pos: NDArray[np.floating[Any]],
        target_quat: NDArray[np.floating[Any]] | None = None,
        exclude_obj: str | None = None,
    ) -> tuple[bool, str, list[str]]:
        """Check if an IK solution exists for the target and is collision-free.

        Uses the Pink solver. Pink clamps its result to the joint limits and
        reports no residual, so an unreachable target comes back as a plausible
        looking solution; reachability is therefore judged by forward kinematics
        on the resulting configuration against an error threshold.

        Args:
            which_arm: Which arm to check ("left" or "right").
            target_pos: Target end effector position [x, y, z].
            target_quat: Target end effector orientation [w, x, y, z].
                         If None, uses the arm's current orientation.
            exclude_obj: Object name to exclude from the collision check.

        Returns:
            Tuple of (success, failure_reason, collision_objects) where:
            - success: True if IK succeeds with no collision
            - failure_reason: "ik_failed", "collision", "exception", or "" if success
            - collision_objects: List of object names that caused collision (empty if not collision)
        """
        ee_link = self._get_ee_link(which_arm)
        is_right = which_arm == "right"

        # Get current orientation if not specified
        if target_quat is None:
            target_quat = ee_link.get_quat().cpu().numpy()[0]

        # Ensure target_pos has shape [3]
        target_pos = np.asarray(target_pos).flatten()[:3]

        # Save current state
        saved_state = self.save_all_states()

        try:
            robot = self._robot
            device = robot.device

            # Pink solves in the robot's URDF frame, which is offset from the
            # Genesis world frame by the robot base position (a pure translation
            # -- base rotation is identity, so quaternions carry over unchanged).
            # Object positions from collect_objs_info are world frame, so the
            # target has to be shifted before it reaches the solver. gs-gym's own
            # apply_ee_delta sidesteps this by seeding targets from Pink's FK.
            base_pos = np.asarray(robot.args.position, dtype=np.float64)
            target_pos_pink = np.asarray(target_pos, dtype=np.float64) - base_pos

            target_pos_t = torch.as_tensor(target_pos_pink, dtype=torch.float32, device=device).unsqueeze(0)
            target_quat_t = torch.as_tensor(
                np.asarray(target_quat).flatten()[:4], dtype=torch.float32, device=device
            ).unsqueeze(0)

            arm_dofs = robot.right_arm_dofs if is_right else robot.left_arm_dofs
            gripper_dofs = robot.right_gripper_dofs if is_right else robot.left_gripper_dofs
            joint_lower = robot.right_arm_lower if is_right else robot.left_arm_lower
            joint_upper = robot.right_arm_upper if is_right else robot.left_arm_upper

            # Solve for this arm only; the other arm stays where it is.
            arm_joints = robot._simple_ik(target_pos_t, target_quat_t, ee_link, arm_dofs, joint_lower, joint_upper)

            # Apply the candidate configuration, opening the gripper so the
            # fingers do not register as the colliding geometry.
            self.robot.set_dofs_position(arm_joints, dofs_idx_local=arm_dofs)
            self.robot.set_dofs_position(
                torch.zeros((1, len(gripper_dofs)), dtype=torch.float32, device=device),
                dofs_idx_local=gripper_dofs,
            )

            # Reachability: did the solved configuration actually land on the target?
            achieved_pos = ee_link.get_pos().cpu().numpy().flatten()[:3]
            ik_err_val = float(np.abs(achieved_pos - target_pos).max())
            if ik_err_val > self.IK_POSITION_TOLERANCE:
                print("IK fail")
                self.restore_all_states(saved_state)
                return False, "ik_failed", []

            # Check collision
            collision_pairs = self.robot.detect_collision()

            entity = self.robot
            collision_with_entity_idx = set()
            for collision_pair in collision_pairs:
                if self.scene.rigid_solver.geoms[collision_pair[0]].entity.idx != entity.idx:
                    collision_with_entity_idx.add(self.scene.rigid_solver.geoms[collision_pair[0]].entity.idx)
                if self.scene.rigid_solver.geoms[collision_pair[1]].entity.idx != entity.idx:
                    collision_with_entity_idx.add(self.scene.rigid_solver.geoms[collision_pair[1]].entity.idx)
            collision_obj_names = [
                self.entity_idx2name[idx] for idx in collision_with_entity_idx if idx in self.entity_idx2name
            ]
            # Exclude target object and table from collision check
            exclude_names = {"table"}
            if exclude_obj is not None:
                exclude_names.add(exclude_obj)
            collision_obj_names = [name for name in collision_obj_names if name not in exclude_names]

            if len(collision_obj_names) > 0:
                print("collision_obj_names:", collision_obj_names)
                self.restore_all_states(saved_state)
                return False, "collision", collision_obj_names

            self.restore_all_states(saved_state)
            return True, "", []

        except Exception as e:
            print(f"Exception when checking IK collision: {e}, traceback: {traceback.format_exc()}")
            self.restore_all_states(saved_state)
            return False, "exception", []

    def check_availability(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
    ) -> tuple[bool, str]:
        """Check if the arm can reach the object without collision.

        Samples up to 10 points on the object's vertices and checks if
        any can be reached by the end effector without collision.

        Args:
            which_arm: Which arm to check ("left" or "right").
            obj_name: Name of the object to check.

        Returns:
            Tuple of (success, failure_reason) where failure_reason is:
            - "": success
            - "out_of_reach": IK failed for all sampled points
            - "collision_with_{obj}": grasp trajectory collides with object(s)
        """
        if obj_name not in self.entities:
            raise ValueError(f"Object '{obj_name}' not found in entities")

        entity = self.entities[obj_name]
        vertices = self._get_mesh_vertices(entity)

        if vertices is None or len(vertices) == 0:
            # Fallback to center position
            pos = entity.get_pos().cpu().numpy()[0]  # Shape: [3]
            success, reason, collision_objs = self._check_ik_collision(which_arm, pos, exclude_obj=obj_name)
            if success:
                return True, ""
            if reason == "ik_failed":
                return False, "out_of_reach"
            if reason == "collision":
                return False, f"collision_with_{','.join(collision_objs)}"
            return False, "unknown_failure"

        # Sample up to 10 vertices
        n_samples = min(10, len(vertices))
        if len(vertices) > n_samples:
            indices = np.random.choice(len(vertices), n_samples, replace=False)
            sampled_vertices = vertices[indices]
        else:
            sampled_vertices = vertices

        # Track failure reasons across all samples
        all_ik_failed = True
        collision_objects: set[str] = set()

        # Check if any sampled vertex is reachable without collision
        for vertex in sampled_vertices:
            success, reason, collision_objs = self._check_ik_collision(which_arm, vertex, exclude_obj=obj_name)
            if success:
                return True, ""
            if reason != "ik_failed":
                all_ik_failed = False
            if reason == "collision":
                collision_objects.update(collision_objs)

        # Determine the most informative failure reason
        if all_ik_failed:
            return False, "out_of_reach"
        if collision_objects:
            return False, f"collision_with_{','.join(sorted(collision_objects))}"
        return False, "unknown_failure"

    def primitive_grasp(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
    ) -> tuple[bool, str]:
        """Grasp an object using the specified arm.

        Holds the object at its current position.

        Args:
            which_arm: Which arm to use ("left" or "right").
            obj_name: Name of the object to grasp.

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        if obj_name not in self.entities:
            return False, f"Object '{obj_name}' not found in the scene"

        entity = self.entities[obj_name]

        # Check if object is fixed (immovable)
        if hasattr(entity, "morph") and getattr(entity.morph, "fixed", False):
            return False, f"Object '{obj_name}' is fixed and cannot be grasped"

        # Check availability first
        available, failure_reason = self.check_availability(which_arm, obj_name)
        if not available:
            print(f"checking availability fail during grasp: {failure_reason}")
            if failure_reason == "out_of_reach":
                return False, f"Object '{obj_name}' is out of reach for {which_arm} arm"
            if failure_reason.startswith("collision_with_"):
                collision_objs = failure_reason[len("collision_with_") :]
                return False, f"Grasp trajectory for '{obj_name}' is in collision with {collision_objs}"
            return False, f"Object '{obj_name}' is not reachable by {which_arm} arm ({failure_reason})"

        # Hold the object at current position using control_dofs_position
        try:
            if entity.n_dofs > 0:
                current_dofs_pos = entity.get_dofs_position()
                entity.control_dofs_position(current_dofs_pos)
                self._held_objects.add(obj_name)
        except Exception:
            pass

        # Mark object as grasped
        self._grasped_objects[obj_name] = which_arm
        return True, f"Successfully grasped '{obj_name}' with {which_arm} arm"

    def primitive_moveto(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
        obj_name2: str,
        offset: tuple[float, float, float] | None = None,
    ) -> tuple[bool, str]:
        """Move obj_name to obj_name2's top surface while keeping gripper closed.

        The object remains held after the move (stays in _grasped_objects).

        Args:
            which_arm: Which arm is holding the object.
            obj_name: Name of the object being moved.
            obj_name2: Name of the target object (support surface).
            offset: Optional (dx, dy, dz) offset from the center of the support surface.

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        # Check if obj_name was grasped by this arm
        if obj_name not in self._grasped_objects:
            return False, f"Cannot move '{obj_name}': it has not been grasped yet. Grasp it first."
        if self._grasped_objects[obj_name] != which_arm:
            other_arm = "right" if which_arm == "left" else "left"
            return False, f"Cannot move '{obj_name}' with {which_arm} arm: it was grasped by {other_arm} arm"

        if obj_name not in self.entities:
            return False, f"Object '{obj_name}' not found in the scene"
        if obj_name2 not in self.entities:
            return False, f"Target surface '{obj_name2}' not found in the scene"

        entity1 = self.entities[obj_name]
        entity2 = self.entities[obj_name2]

        # Get objects inside the AABB before moving (e.g., objects inside a container)
        bounds1 = self._get_AABB(entity1)
        inside_objects = self._get_objects_inside_aabb(bounds1, obj_name)

        # Release control on inside objects so they can be moved
        for inside_obj_name in inside_objects:
            self._release_object_control(inside_obj_name)

        # Get current position and AABB of both objects
        pos1 = entity1.get_pos().cpu().numpy().flatten()
        bounds2 = self._get_AABB(entity2)
        # print("obj1:", obj_name, "pos1:", pos1, "bounds1:", bounds1)
        # print("obj2:", obj_name2, "bounds2:", bounds2)

        # Compute target position: obj1 centered on top of obj2
        # Account for offset between object position and its bounding box center
        center1_x = (bounds1[0, 0] + bounds1[1, 0]) / 2
        center1_y = (bounds1[0, 1] + bounds1[1, 1]) / 2
        center2_x = (bounds2[0, 0] + bounds2[1, 0]) / 2
        center2_y = (bounds2[0, 1] + bounds2[1, 1]) / 2

        # target_x/y: center obj1 on obj2, accounting for pos-to-center offset
        target_x = center2_x + (pos1[0] - center1_x)
        target_y = center2_y + (pos1[1] - center1_y)
        # target_z: place bottom of obj1 on top of obj2
        # offset from pos to bottom = pos1[2] - bounds1[0, 2]
        target_z = bounds2[1, 2] + (pos1[2] - bounds1[0, 2])

        # Apply offset if provided (dx, dy, dz)
        if offset is not None:
            target_x += offset[0]
            target_y += offset[1]
            target_z += offset[2]

        target_pos = np.array([target_x, target_y, target_z])

        # Compute displacement to apply to inside objects
        displacement = target_pos - pos1

        # Set main object position (Genesis requires batch dim)
        entity1.set_pos(torch.tensor(target_pos[np.newaxis, :]))

        # Move inside objects by the same displacement
        for inside_obj_name in inside_objects:
            inside_entity = self.entities[inside_obj_name]
            inside_pos = inside_entity.get_pos().cpu().numpy().flatten()
            new_inside_pos = inside_pos + displacement
            inside_entity.set_pos(torch.tensor(new_inside_pos[np.newaxis, :]))

        # Step once to update DOFs state
        self.scene.step()

        # Hold the main object at new position using control_dofs_position
        try:
            if entity1.n_dofs > 0:
                current_dofs_pos = entity1.get_dofs_position()
                entity1.control_dofs_position(current_dofs_pos)
                self._held_objects.add(obj_name)
        except Exception:
            pass

        # Update control for all other objects (excluding main object and inside objects)
        self._control_all_objects_position(exclude={obj_name} | set(inside_objects))

        # Check availability at new position
        available, failure_reason = self.check_availability(which_arm, obj_name)
        if not available:
            if failure_reason == "out_of_reach":
                return False, f"Target position for moving '{obj_name}' to '{obj_name2}' is out of reach"
            if failure_reason.startswith("collision_with_"):
                collision_objs = failure_reason[len("collision_with_") :]
                return False, f"Moving '{obj_name}' to '{obj_name2}' would cause collision with {collision_objs}"
            return (
                False,
                f"Target position for moving '{obj_name}' to '{obj_name2}' is not reachable ({failure_reason})",
            )

        # Object remains grasped (gripper stays closed) - do NOT remove from _grasped_objects

        self._step_until_stable()

        # Check for intersections with other rigid entities
        has_intersection, intersecting_objs = self._check_intersection_with_others(obj_name, exclude_names={obj_name2})
        if has_intersection:
            return False, f"Object '{obj_name}' intersects with {', '.join(intersecting_objs)} after move"

        if offset is not None:
            return True, f"Successfully moved '{obj_name}' to '{obj_name2}' with offset {offset}"
        return True, f"Successfully moved '{obj_name}' to '{obj_name2}'"

    def primitive_drop(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
    ) -> tuple[bool, str]:
        """Drop/release the held object from the specified arm.

        Opens the gripper and lets the object fall due to gravity.

        Args:
            which_arm: Which arm is holding the object.
            obj_name: Name of the object to drop.

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        # Check if obj_name was grasped by this arm
        if obj_name not in self._grasped_objects:
            return False, f"Cannot drop '{obj_name}': it has not been grasped yet."
        if self._grasped_objects[obj_name] != which_arm:
            other_arm = "right" if which_arm == "left" else "left"
            return False, f"Cannot drop '{obj_name}' with {which_arm} arm: it was grasped by {other_arm} arm"

        if obj_name not in self.entities:
            return False, f"Object '{obj_name}' not found in the scene"

        entity = self.entities[obj_name]

        # Release position control on the object
        self._release_object_control(obj_name)

        # Set zero force to let gravity take over
        try:
            if entity.n_dofs > 0:
                n_dofs = entity.n_dofs
                entity.control_dofs_force(torch.zeros(1, n_dofs))
        except Exception as e:
            print(f"Warning: Failed to release force control on {obj_name}: {e}")

        # Remove from grasped objects (gripper opens)
        del self._grasped_objects[obj_name]

        # Update control for all other objects
        self._control_all_objects_position(exclude={obj_name})

        # Let physics simulate the drop
        self._step_until_stable()

        return True, f"Successfully dropped '{obj_name}' from {which_arm} arm"

    def _check_aabb_intersection(
        self,
        bounds1: NDArray[np.floating[Any]],
        bounds2: NDArray[np.floating[Any]],
    ) -> bool:
        """Check if two AABBs intersect (overlap in all three axes).

        Args:
            bounds1: First AABB [min_corner, max_corner] of shape [2, 3].
            bounds2: Second AABB [min_corner, max_corner] of shape [2, 3].

        Returns:
            True if the AABBs intersect, False otherwise.
        """
        # Check overlap on each axis: max1 >= min2 and max2 >= min1
        return (
            bounds1[1, 0] >= bounds2[0, 0]
            and bounds2[1, 0] >= bounds1[0, 0]
            and bounds1[1, 1] >= bounds2[0, 1]
            and bounds2[1, 1] >= bounds1[0, 1]
            and bounds1[1, 2] >= bounds2[0, 2]
            and bounds2[1, 2] >= bounds1[0, 2]
        )

    def _get_entity_mesh(self, entity: Any) -> trimesh.Trimesh | None:
        """Get the combined mesh of an entity in world coordinates.

        Args:
            entity: The RigidEntity to get mesh for.

        Returns:
            Combined trimesh in world coordinates, or None if not available.
        """
        from genesis.options.morphs import Box, Cylinder, Sphere
        from trimesh.transformations import quaternion_matrix

        pos = entity.get_pos().cpu().numpy()[0]  # Shape: [3]
        quat = entity.get_quat().cpu().numpy()[0]  # Shape: [4]

        T = quaternion_matrix(quat)
        T[:3, 3] = pos

        # Try to get mesh from geoms
        if entity.n_geoms > 0 and hasattr(entity, "geoms"):
            try:
                meshes = []
                for geom in entity.geoms:
                    mesh = geom.get_trimesh()
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(T)
                    meshes.append(mesh_copy)
                if meshes:
                    return trimesh.util.concatenate(meshes)
            except Exception:
                pass

        # Fallback to primitive shapes
        try:
            if isinstance(entity.morph, Box):
                morph_size = entity.morph.size
                if morph_size is not None:
                    mesh = trimesh.creation.box(extents=morph_size)
                    mesh.apply_transform(T)
                    return mesh
            elif isinstance(entity.morph, Cylinder):
                mesh = trimesh.creation.cylinder(
                    radius=entity.morph.radius,
                    height=entity.morph.height,
                )
                mesh.apply_transform(T)
                return mesh
            elif isinstance(entity.morph, Sphere):
                mesh = trimesh.creation.icosphere(radius=entity.morph.radius)
                mesh.apply_transform(T)
                return mesh
        except Exception:
            pass

        return None

    def _check_mesh_intersection_raycast(
        self,
        mesh1: trimesh.Trimesh,
        mesh2: trimesh.Trimesh,
    ) -> bool:
        """Check mesh intersection using ray casting (no FCL required).

        Checks if any vertices of mesh1 are inside mesh2 or vice versa.

        Args:
            mesh1: First mesh.
            mesh2: Second mesh.

        Returns:
            True if meshes intersect, False otherwise.
        """
        try:
            # Check if any vertices of mesh1 are inside mesh2
            if mesh2.is_watertight:
                inside1 = mesh2.contains(mesh1.vertices)
                if np.any(inside1):
                    return True

            # Check if any vertices of mesh2 are inside mesh1
            if mesh1.is_watertight:
                inside2 = mesh1.contains(mesh2.vertices)
                if np.any(inside2):
                    return True

            # For non-watertight meshes or edge cases, sample and check
            # Check if mesh centers are inside each other
            center1 = mesh1.centroid
            center2 = mesh2.centroid

            if mesh2.is_watertight and mesh2.contains([center1])[0]:
                return True
            if mesh1.is_watertight and mesh1.contains([center2])[0]:
                return True

            # Check face centers for more thorough coverage
            if len(mesh1.triangles) > 0 and len(mesh2.triangles) > 0:
                # Sample some face centers from mesh1 and check against mesh2
                face_centers1 = mesh1.triangles.mean(axis=1)
                n_samples = min(50, len(face_centers1))
                if n_samples > 0:
                    indices = np.linspace(0, len(face_centers1) - 1, n_samples, dtype=int)
                    sample_points1 = face_centers1[indices]
                    if mesh2.is_watertight and np.any(mesh2.contains(sample_points1)):
                        return True

                # Sample some face centers from mesh2 and check against mesh1
                face_centers2 = mesh2.triangles.mean(axis=1)
                n_samples = min(50, len(face_centers2))
                if n_samples > 0:
                    indices = np.linspace(0, len(face_centers2) - 1, n_samples, dtype=int)
                    sample_points2 = face_centers2[indices]
                    if mesh1.is_watertight and np.any(mesh1.contains(sample_points2)):
                        return True

            return False

        except Exception:
            # If ray casting fails, return False (no confirmed intersection)
            return False

    def _check_mesh_intersection(
        self,
        entity1: Any,
        entity2: Any,
    ) -> bool | None:
        """Check if two entities' meshes intersect.

        Uses ray casting to check if vertices/points of one mesh are inside the other.

        Args:
            entity1: First entity.
            entity2: Second entity.

        Returns:
            True if meshes intersect, False if not, None if check unavailable.
        """
        mesh1 = self._get_entity_mesh(entity1)
        mesh2 = self._get_entity_mesh(entity2)

        if mesh1 is None or mesh2 is None:
            # If we can't get meshes, return None to indicate check unavailable
            return None

        # Try FCL-based collision first (faster if available)
        try:
            manager = trimesh.collision.CollisionManager()
            manager.add_object("mesh1", mesh1)
            is_collision = manager.in_collision_single(mesh2)
            return bool(is_collision)
        except Exception:
            # FCL not available, fall back to ray casting method
            pass

        # Use ray casting method (no FCL required)
        return self._check_mesh_intersection_raycast(mesh1, mesh2)

    def _check_intersection_with_others(
        self,
        obj_name: str,
        exclude_names: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check if an object intersects with other rigid entities.

        Uses AABB as a fast primary filter, then checks mesh intersection
        for candidates with overlapping AABBs.

        Args:
            obj_name: Name of the object to check.
            exclude_names: Set of object names to exclude from the check.

        Returns:
            Tuple of (has_intersection, intersecting_object_names).
        """
        from genesis.engine.entities.rigid_entity import RigidEntity

        if exclude_names is None:
            exclude_names = set()

        # Always exclude these
        exclude_names = exclude_names | {"robot", "table", "floor", "wall_front", obj_name}

        entity = self.entities[obj_name]
        bounds = self._get_AABB(entity)

        intersecting_objects: list[str] = []

        for other_name, other_entity in self.entities.items():
            if other_name in exclude_names:
                continue
            if not isinstance(other_entity, RigidEntity):
                continue

            try:
                other_bounds = self._get_AABB(other_entity)
                # Fast AABB check first
                if not self._check_aabb_intersection(bounds, other_bounds):
                    continue

                # AABBs overlap - do detailed mesh intersection check
                mesh_result = self._check_mesh_intersection(entity, other_entity)
                # Only report intersection if mesh check confirms it
                # If mesh check is unavailable (None), skip this object
                if mesh_result is True:
                    intersecting_objects.append(other_name)
            except Exception:
                continue

        return len(intersecting_objects) > 0, intersecting_objects

    def _get_objects_inside_aabb(self, bounds: NDArray[np.floating[Any]], exclude_obj: str) -> list[str]:
        """Get names of objects whose center is inside the given AABB.

        Args:
            bounds: AABB bounds [min_corner, max_corner].
            exclude_obj: Object name to exclude from results.

        Returns:
            List of object names inside the AABB.
        """
        inside_objects = []

        for name, entity in self.entities.items():
            if name in ("robot", "table", "floor", "wall_front", exclude_obj):
                continue

            try:
                pos = entity.get_pos().cpu().numpy()[0]  # Shape: [3]
                if (
                    bounds[0, 0] <= pos[0] <= bounds[1, 0]
                    and bounds[0, 1] <= pos[1] <= bounds[1, 1]
                    and bounds[0, 2] <= pos[2] <= bounds[1, 2]
                ):
                    inside_objects.append(name)
            except Exception:
                continue

        return inside_objects

    def primitive_rotate180(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
    ) -> tuple[bool, str]:
        """Rotate object upside down (180 degrees around x-axis).

        The object is held at its current position using control_dofs_position.

        Args:
            which_arm: Which arm is holding the object.
            obj_name: Name of the object to rotate.

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        # Check if obj_name was grasped by this arm
        if obj_name not in self._grasped_objects:
            return False, f"Cannot rotate '{obj_name}': it has not been grasped yet. Grasp it first."
        if self._grasped_objects[obj_name] != which_arm:
            other_arm = "right" if which_arm == "left" else "left"
            return False, f"Cannot rotate '{obj_name}' with {which_arm} arm: it was grasped by {other_arm} arm"

        if obj_name not in self.entities:
            return False, f"Object '{obj_name}' not found in the scene"

        entity = self.entities[obj_name]

        # Get objects inside the AABB before rotation
        bounds = self._get_AABB(entity)
        inside_objects = self._get_objects_inside_aabb(bounds, obj_name)

        # Get current quaternion and apply 180 degree rotation around x-axis
        current_quat = entity.get_quat().cpu().numpy().flatten()
        rotation_quat = npm.euler_to_quat(np.array([np.pi, 0, 0]))  # 180 deg around x
        new_quat = npm.quat_mul(rotation_quat, current_quat)

        # Apply rotation to main object (Genesis requires batch dim)
        entity.set_quat(torch.tensor(new_quat[np.newaxis, :]))

        # Rotate inside objects as well
        print("inside_objects:", inside_objects)
        for inside_obj_name in inside_objects:
            inside_entity = self.entities[inside_obj_name]
            inside_quat = inside_entity.get_quat().cpu().numpy().flatten()
            new_inside_quat = npm.quat_mul(rotation_quat, inside_quat)
            inside_entity.set_quat(torch.tensor(new_inside_quat[np.newaxis, :]))

        # Step once to update DOFs state
        self.scene.step()

        # Hold the object at current position using control_dofs_position
        try:
            if entity.n_dofs > 0:
                current_dofs_pos = entity.get_dofs_position()
                entity.control_dofs_position(current_dofs_pos)
                self._held_objects.add(obj_name)
        except Exception:
            pass

        self._step_until_stable()
        return True, f"Successfully rotated '{obj_name}' 180 degrees (flipped upside down)"

    def primitive_rotate90(
        self,
        which_arm: Literal["left", "right"],
        obj_name: str,
        direction: Literal["front", "back", "left", "right"],
    ) -> tuple[bool, str]:
        """Rotate object 90 degrees in the specified direction.

        Args:
            which_arm: Which arm is holding the object.
            obj_name: Name of the object to rotate.
            direction: Direction to push the object down:
                - "front" (+x): rotate around -y axis
                - "back" (-x): rotate around +y axis
                - "left" (+y): rotate around +x axis
                - "right" (-y): rotate around -x axis

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        # Check if obj_name was grasped by this arm
        if obj_name not in self._grasped_objects:
            return False, f"Cannot rotate '{obj_name}': it has not been grasped yet. Grasp it first."
        if self._grasped_objects[obj_name] != which_arm:
            other_arm = "right" if which_arm == "left" else "left"
            return False, f"Cannot rotate '{obj_name}' with {which_arm} arm: it was grasped by {other_arm} arm"

        if obj_name not in self.entities:
            return False, f"Object '{obj_name}' not found in the scene"

        # Release held object
        self._release_object_control(obj_name)

        entity = self.entities[obj_name]

        # Check availability before rotation
        available, failure_reason = self.check_availability(which_arm, obj_name)
        if not available:
            if failure_reason == "out_of_reach":
                return False, f"Cannot rotate '{obj_name}': position is out of reach for {which_arm} arm"
            if failure_reason.startswith("collision_with_"):
                collision_objs = failure_reason[len("collision_with_") :]
                return False, f"Cannot rotate '{obj_name}': trajectory would collide with {collision_objs}"
            return False, f"Cannot rotate '{obj_name}': position is not reachable by {which_arm} arm ({failure_reason})"

        # Get objects inside the AABB before rotation
        bounds = self._get_AABB(entity)
        inside_objects = self._get_objects_inside_aabb(bounds, obj_name)

        # Release inside objects too
        for inside_obj_name in inside_objects:
            self._release_object_control(inside_obj_name)

        # Determine rotation axis based on direction
        rotation_map = {
            "front": np.array([0, -np.pi / 2, 0]),  # +x: rotate around -y
            "back": np.array([0, np.pi / 2, 0]),  # -x: rotate around +y
            "left": np.array([np.pi / 2, 0, 0]),  # +y: rotate around +x
            "right": np.array([-np.pi / 2, 0, 0]),  # -y: rotate around -x
        }

        rotation_euler = rotation_map[direction]
        rotation_quat = npm.euler_to_quat(rotation_euler)

        # Get current quaternion and apply rotation
        current_quat = entity.get_quat().cpu().numpy().flatten()
        new_quat = npm.quat_mul(rotation_quat, current_quat)

        # Apply rotation to main object (Genesis requires batch dim)
        entity.set_quat(torch.tensor(new_quat[np.newaxis, :]))

        # Rotate inside objects as well
        for inside_obj_name in inside_objects:
            inside_entity = self.entities[inside_obj_name]
            inside_quat = inside_entity.get_quat().cpu().numpy().flatten()
            new_inside_quat = npm.quat_mul(rotation_quat, inside_quat)
            inside_entity.set_quat(torch.tensor(new_inside_quat[np.newaxis, :]))

        # Update control to new positions
        self._control_all_objects_position()

        # print("Calling _step_until_stable() in rotate90")
        self._step_until_stable()
        return True, f"Successfully rotated '{obj_name}' 90 degrees toward {direction}"

    def primitive_hit(
        self,
        which_arm: Literal["left", "right"],
        obj_A: str,
        obj_B: str,
    ) -> tuple[bool, str]:
        """Apply a force to obj_A in the direction from obj_A to obj_B.

        Args:
            which_arm: Which arm is holding obj_A.
            obj_A: Name of the object to apply force to.
            obj_B: Name of the target object (defines force direction).

        Returns:
            Tuple of (success, feedback) where feedback explains the result.
        """
        # Check if obj_A was grasped by this arm
        if obj_A not in self._grasped_objects:
            return False, f"Cannot hit with '{obj_A}': it has not been grasped yet. Grasp it first."
        if self._grasped_objects[obj_A] != which_arm:
            other_arm = "right" if which_arm == "left" else "left"
            return False, f"Cannot hit with '{obj_A}' using {which_arm} arm: it was grasped by {other_arm} arm"

        if obj_A not in self.entities:
            return False, f"Object '{obj_A}' not found in the scene"
        if obj_B not in self.entities:
            return False, f"Target object '{obj_B}' not found in the scene"

        # Release held objects that are involved
        self._release_object_control(obj_A)
        self._release_object_control(obj_B)

        entity_A = self.entities[obj_A]
        entity_B = self.entities[obj_B]

        # Get positions (index [0] for first batch element)
        pos_A = entity_A.get_pos().cpu().numpy()[0]  # Shape: [3]
        pos_B = entity_B.get_pos().cpu().numpy()[0]  # Shape: [3]

        # Compute direction vector
        direction = pos_B - pos_A
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return False, f"Objects '{obj_A}' and '{obj_B}' are at the same position, cannot determine hit direction"

        direction = direction / distance

        # Release position control on obj_A by setting zero force
        # This allows the object to move freely when we set its velocity
        try:
            if entity_A.n_dofs > 0:
                n_dofs = entity_A.n_dofs
                entity_A.control_dofs_force(torch.zeros(1, n_dofs))
        except Exception as e:
            print(f"Warning: Failed to release force control on {obj_A}: {e}")

        # Apply velocity in the direction (simulating a hit/push)
        velocity_magnitude = 3.0  # m/s (kept moderate to avoid simulation instability)
        linear_velocity = direction * velocity_magnitude

        # Set DOFs velocity - first 3 DOFs are linear velocity [vx, vy, vz]
        # Create velocity array with 6 DOFs: [vx, vy, vz, wx, wy, wz]
        dofs_velocity = np.zeros(6)
        dofs_velocity[:3] = linear_velocity
        entity_A.set_dofs_velocity(torch.tensor(dofs_velocity[np.newaxis, :]))

        # Update control (excluding the objects being hit)
        self._control_all_objects_position(exclude={obj_A, obj_B})

        # Let physics simulate the hit
        try:
            self._step_until_stable()
        except Exception as e:
            print(f"Warning: Simulation instability during hit: {e}")
            return False, "Hit action caused simulation instability"
        return True, f"Successfully hit '{obj_B}' with '{obj_A}'"

    def get_three_images(self) -> NDArray[np.uint8]:
        """Get images from the three observation cameras (ego, wrist_left, wrist_right).

        Returns:
            Array of shape [3, 3, H, W] where:
            - Dimension 0: camera index (ego, wrist_left, wrist_right)
            - Dimension 1: RGB channels
            - Dimension 2: height
            - Dimension 3: width
        """
        images = []

        # Render from each camera using .read() which returns camera data with .rgb
        for cam in [self.ego_cam, self.wrist_left_cam, self.wrist_right_cam]:
            if cam is None:
                raise ValueError("Camera not initialized. Call set_env() first.")
            camera_data = cam.read()
            # camera_data.rgb tensor, squeeze to get [H, W, 3].
            # Genesis may hand back float rgb; match RoboWitsEnv.get_observations.
            frame = camera_data.rgb
            if frame.dtype != torch.uint8:
                frame = (frame * 255).to(torch.uint8)
            img = frame.cpu().numpy().squeeze()
            # img is [H, W, 3] in uint8, convert to [3, H, W]
            img_chw = np.transpose(img, (2, 0, 1))
            images.append(img_chw)

        # Stack to [3, 3, H, W]
        return np.stack(images, axis=0)

    def save_image(self, save_path: str = "images.png") -> None:
        """Save the current images to a file.

        Args:
            save_path: Path to save the image file.
        """
        img1 = self.ego_cam.read().rgb.cpu().numpy().squeeze()
        img2 = self.wrist_left_cam.read().rgb.cpu().numpy().squeeze()
        img3 = self.wrist_right_cam.read().rgb.cpu().numpy().squeeze()
        # Concatenate ego + wrist_left + wrist_right horizontally
        img = np.concatenate([img1, img2, img3], axis=1)
        Image.fromarray(img).save(save_path)

    def save_all_states(self) -> dict[str, dict[str, Any]]:
        """Save the state of all entities in the scene.

        For rigid entities: saves pos, quat, dofs_position, dofs_velocity.
        For particle entities: saves particles_pos, particles_vel.

        Returns:
            Dictionary mapping entity names to their saved state.
        """
        from genesis.engine.entities.particle_entity import ParticleEntity
        from genesis.engine.entities.rigid_entity import RigidEntity

        states: dict[str, dict[str, Any]] = {}

        for name, entity in self.entities.items():
            if isinstance(entity, RigidEntity):
                state: dict[str, Any] = {
                    "type": "rigid",
                    "pos": entity.get_pos().cpu().clone(),
                    "quat": entity.get_quat().cpu().clone(),
                }
                # DOFs may not exist for all rigid entities
                try:
                    state["dofs_position"] = entity.get_dofs_position().cpu().clone()
                    state["dofs_velocity"] = entity.get_dofs_velocity().cpu().clone()
                except Exception:
                    pass
                states[name] = state
            elif isinstance(entity, ParticleEntity):
                states[name] = {
                    "type": "particle",
                    "particles_pos": entity.get_particles_pos().cpu().clone(),
                    "particles_vel": entity.get_particles_vel().cpu().clone(),
                }

        return states

    def restore_all_states(self, states: dict[str, dict[str, Any]]) -> None:
        """Restore the state of all entities from a saved state dict.

        Args:
            states: Dictionary mapping entity names to their saved state
                    (as returned by save_all_states).
        """
        from genesis.engine.entities.particle_entity import ParticleEntity
        from genesis.engine.entities.rigid_entity import RigidEntity

        for name, state in states.items():
            if name not in self.entities:
                continue

            entity = self.entities[name]

            if state["type"] == "rigid" and isinstance(entity, RigidEntity):
                entity.set_pos(state["pos"])
                entity.set_quat(state["quat"])
                if "dofs_position" in state:
                    entity.set_dofs_position(state["dofs_position"])
                if "dofs_velocity" in state:
                    entity.set_dofs_velocity(state["dofs_velocity"])
            elif state["type"] == "particle" and isinstance(entity, ParticleEntity):
                entity.set_particles_pos(state["particles_pos"])
                entity.set_particles_vel(state["particles_vel"])

        self.scene.step()


# ==============================================================================
# VLM planners
# ==============================================================================


@dataclass
class Action:
    """Represents a primitive action to execute."""

    name: str  # "grasp", "moveto", "drop", "rotate180", "rotate90", "hit", "done"
    which_arm: Literal["left", "right"] | None = None
    obj_name: str | None = None
    obj_name2: str | None = None  # For place and hit
    direction: Literal["front", "back", "left", "right"] | None = None  # For rotate90
    offset: tuple[float, float, float] | None = None  # For moveto: (dx, dy, dz) offset on support surface


@dataclass
class VLMCallRecord:
    """Record of a single VLM call with prompt and response.

    Shared by both planners: the closed-loop planner records one call per action
    (`action_index`, `parsed_action`), the open-loop planner records a single
    call holding the whole plan (`parsed_plan`).
    """

    system_prompt: str
    user_prompt: str
    response: str
    action_index: int = 0
    parsed_action: dict[str, Any] | None = None
    parsed_plan: list[dict[str, Any]] | None = None
    images: NDArray[np.uint8] | None = None  # [3, 3, H, W] images


def _write_vlm_call(
    vlm_dir: Path,
    record: VLMCallRecord,
    *,
    prefix: str = "",
    prompt_stem: str = "prompt",
    parsed: dict[str, Any] | list[dict[str, Any]] | None = None,
    parsed_stem: str = "parsed",
    write_system_prompt: bool = True,
) -> None:
    """Write one VLM call's prompts, response, parsed output and images into `vlm_dir`.

    Both planners save the same artifacts and differ only in how they name them:
    the closed-loop planner writes one set per action under an `action_NNN_`
    prefix, the open-loop planner writes a single unprefixed set.
    """
    if write_system_prompt:
        (vlm_dir / "system_prompt.txt").write_text(record.system_prompt)

    (vlm_dir / f"{prefix}{prompt_stem}.txt").write_text(record.user_prompt)
    (vlm_dir / f"{prefix}response.txt").write_text(record.response)

    if parsed is not None:
        with open(vlm_dir / f"{prefix}{parsed_stem}.json", "w") as f:
            json.dump(parsed, f, indent=2)

    if record.images is not None:
        for i, cam_name in enumerate(["ego", "wrist_left", "wrist_right"]):
            img_hwc = np.transpose(record.images[i], (1, 2, 0))
            Image.fromarray(img_hwc).save(vlm_dir / f"{prefix}{cam_name}.png")


class ClosedLoopVLMPlanner:
    """Vision-Language Model planner for robowits tasks."""

    SYSTEM_PROMPT = """You are a robotic manipulation planner. You have a dual-arm robot that can perform the following primitive actions:

AVAILABLE PRIMITIVES:
1. grasp(which_arm, obj_name) - Grasp an object with the specified arm ("left" or "right")
2. moveto(which_arm, obj_name, target_obj_name, offset) - Move obj_name to target_obj_name's top surface (gripper stays closed)
   - offset is optional [dx, dy, dz] in meters to offset from center of target surface
   - Use offset to avoid overlapping objects when placing multiple items on the same surface
   - Object remains held after moveto; use drop to release
3. drop(which_arm, obj_name) - Drop/release the held object from the specified arm (opens gripper)
4. rotate180(which_arm, obj_name) - Rotate the held object upside down (180 degrees)
5. rotate90(which_arm, obj_name, direction) - Rotate object 90 degrees in direction ("front", "back", "left", "right")
6. hit(which_arm, obj_A, obj_B) - Hit obj_B with obj_A (obj_A must be held)
7. done() - Call this when the task is completed

RULES:
- You must grasp an object before you can moveto, drop, rotate, or hit with it
- Each arm can only hold one object at a time
- Output exactly ONE action per turn
- If the last action failed, try a different approach

OUTPUT FORMAT:
You must respond with a JSON object containing the action. Examples:
{"action": "grasp", "which_arm": "left", "obj_name": "cube"}
{"action": "moveto", "which_arm": "left", "obj_name": "cube", "obj_name2": "table"}
{"action": "moveto", "which_arm": "left", "obj_name": "cube", "obj_name2": "table", "offset": [0.1, 0.0, 0.05]}
{"action": "drop", "which_arm": "left", "obj_name": "cube"}
{"action": "rotate180", "which_arm": "left", "obj_name": "container"}
{"action": "rotate90", "which_arm": "right", "obj_name": "box", "direction": "front"}
{"action": "hit", "which_arm": "left", "obj_A": "hammer", "obj_B": "nail"}
{"action": "done"}

Think step by step about what needs to be done, then output the JSON action."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_steps: int = 20,
    ) -> None:
        """Initialize the VLM planner.

        Args:
            model: The model to use (default: gpt-4o).
            max_steps: Maximum number of planning steps.
        """
        self.model = model
        self.max_steps = max_steps
        self.action_history: list[tuple[Action, bool, str]] = []  # (action, success, feedback)
        self.vlm_call_records: list[VLMCallRecord] = []  # Records of all VLM calls

        # Initialize OpenAI client (uses OPENAI_API_KEY env var)
        try:
            from openai import OpenAI

            client = OpenAI()

            # Wrap with LangSmith tracing if available
            try:
                from langsmith import wrappers as langsmith_wrappers

                client = langsmith_wrappers.wrap_openai(client)
                print("LangSmith tracing enabled for OpenAI client")
            except ImportError:
                print("LangSmith not installed, skipping tracing")

            self.client = client
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: uv pip install 'openai>=1.0.0' "
                "(or install gs-gym with the 'baselines' extra)"
            ) from e

    def _format_obj_state(self, objs_info: dict[str, dict[str, Any]]) -> str:
        """Format object state information as readable text.

        Args:
            objs_info: Object information from collect_objs_info().

        Returns:
            Formatted string describing all objects.
        """
        lines = ["CURRENT OBJECT STATES:"]

        for name, info in objs_info.items():
            pos = info.get("pos", [0, 0, 0])
            material = info.get("material", "unknown")

            # Format position nicely
            if hasattr(pos, "flatten"):
                pos = pos.flatten()
            pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

            lines.append(f"- {name}: position={pos_str}, material={material}")

            # Add bounds if available
            if "bounds" in info:
                bounds = info["bounds"]
                size = bounds[1] - bounds[0]
                lines.append(f"    size=[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

        return "\n".join(lines)

    def _format_action_history(self) -> str:
        """Format the action history as readable text.

        Returns:
            Formatted string describing past actions and results.
        """
        if not self.action_history:
            return "ACTION HISTORY: No actions taken yet."

        lines = ["ACTION HISTORY:"]
        for i, (action, success, feedback) in enumerate(self.action_history):
            result = "SUCCESS" if success else "FAILED"
            action_str = self._action_to_string(action)
            if feedback:
                lines.append(f"  Step {i + 1}: {action_str} -> {result} ({feedback})")
            else:
                lines.append(f"  Step {i + 1}: {action_str} -> {result}")

        return "\n".join(lines)

    def _action_to_string(self, action: Action) -> str:
        """Convert an Action to a readable string."""
        if action.name == "done":
            return "done()"
        elif action.name == "grasp":
            return f"grasp({action.which_arm}, {action.obj_name})"
        elif action.name == "moveto":
            if action.offset:
                return f"moveto({action.which_arm}, {action.obj_name}, {action.obj_name2}, offset={action.offset})"
            return f"moveto({action.which_arm}, {action.obj_name}, {action.obj_name2})"
        elif action.name == "drop":
            return f"drop({action.which_arm}, {action.obj_name})"
        elif action.name == "rotate180":
            return f"rotate180({action.which_arm}, {action.obj_name})"
        elif action.name == "rotate90":
            return f"rotate90({action.which_arm}, {action.obj_name}, {action.direction})"
        elif action.name == "hit":
            return f"hit({action.which_arm}, {action.obj_name}, {action.obj_name2})"
        return f"unknown({action.name})"

    def _encode_images(self, images: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Encode images as base64 for the API.

        Args:
            images: Array of shape [3, 3, H, W] from get_three_images().

        Returns:
            List of image content dicts for the API.
        """
        image_contents = []
        camera_names = ["ego (front view)", "wrist_left", "wrist_right"]

        for i, cam_name in enumerate(camera_names):
            # images[i] is [3, H, W], convert to [H, W, 3]
            img_hwc = np.transpose(images[i], (1, 2, 0))
            pil_img = Image.fromarray(img_hwc)

            # Encode as base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            image_contents.append(
                {
                    "type": "text",
                    "text": f"Camera: {cam_name}",
                }
            )
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        return image_contents

    def _parse_action(self, response_text: str) -> Action:
        """Parse the VLM response to extract an action.

        Args:
            response_text: Raw text response from the VLM.

        Returns:
            Parsed Action object.
        """
        # Try to find JSON in the response (including nested arrays for offset)
        json_match = re.search(r"\{[^{}]*(?:\[[^\]]*\][^{}]*)?\}", response_text, re.DOTALL)
        if not json_match:
            print(f"Warning: Could not find JSON in response: {response_text[:200]}")
            return Action(name="done")

        try:
            action_dict = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            return Action(name="done")

        action_name = action_dict.get("action", "done")

        # Parse offset if present (dx, dy, dz)
        offset = None
        if "offset" in action_dict and action_dict["offset"] is not None:
            offset_list = action_dict["offset"]
            if isinstance(offset_list, list) and len(offset_list) >= 3:
                offset = (float(offset_list[0]), float(offset_list[1]), float(offset_list[2]))
            elif isinstance(offset_list, list) and len(offset_list) >= 2:
                # Backwards compatibility: if only 2 values provided, assume dz=0
                offset = (float(offset_list[0]), float(offset_list[1]), 0.0)

        return Action(
            name=action_name,
            which_arm=action_dict.get("which_arm"),
            obj_name=action_dict.get("obj_name") or action_dict.get("obj_A"),
            obj_name2=action_dict.get("obj_name2") or action_dict.get("obj_B"),
            direction=action_dict.get("direction"),
            offset=offset,
        )

    def plan_next_action(
        self,
        task_description: str,
        objs_info: dict[str, dict[str, Any]],
        images: NDArray[np.uint8],
        last_action_result: bool | None = None,
    ) -> Action:
        """Plan the next action based on current state.

        Args:
            task_description: Description of the task to accomplish.
            objs_info: Object information from collect_objs_info().
            images: Images from get_three_images().
            last_action_result: Result of the last action (None if first action).

        Returns:
            The next Action to execute.
        """
        # Update history with last result if applicable
        if last_action_result is not None and self.action_history:
            # The last action's result is updated
            pass  # Already added in execute loop

        # Build the user message
        obj_state_text = self._format_obj_state(objs_info)
        history_text = self._format_action_history()

        user_text = f"""TASK: {task_description}

{obj_state_text}

{history_text}

Based on the current state and images, what is the next action to take?
Remember to output your action as a JSON object."""

        # Encode images
        image_contents = self._encode_images(images)

        # Build message content
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(image_contents)

        action_index = len(self.vlm_call_records) + 1

        print(f"=== VLM PROMPT (Action {action_index}) ===")
        print(f"System: {self.SYSTEM_PROMPT[:200]}...")
        print(f"User: {user_text}")
        print("=== END PROMPT ===")

        # Call the VLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )

            response_text = response.choices[0].message.content or ""
            print(f"=== VLM RESPONSE (Action {action_index}) ===")
            print(response_text)
            print("=== END RESPONSE ===")

            # Record the VLM call
            parsed = self._parse_action(response_text)
            record = VLMCallRecord(
                action_index=action_index,
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_text,
                response=response_text,
                parsed_action={
                    "name": parsed.name,
                    "which_arm": parsed.which_arm,
                    "obj_name": parsed.obj_name,
                    "obj_name2": parsed.obj_name2,
                    "direction": parsed.direction,
                    "offset": list(parsed.offset) if parsed.offset else None,
                },
                images=images.copy(),
            )
            self.vlm_call_records.append(record)

            return parsed

        except Exception as e:
            print(f"Error calling VLM: {e}, traceback: {traceback.format_exc()}")
            # Record the failed call
            record = VLMCallRecord(
                action_index=action_index,
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_text,
                response=f"[Error: {e}]",
                parsed_action=None,
                images=images.copy(),
            )
            self.vlm_call_records.append(record)
            return Action(name="done")

    def add_action_result(self, action: Action, success: bool, feedback: str = "") -> None:
        """Record an action and its result.

        Args:
            action: The action that was executed.
            success: Whether the action succeeded.
            feedback: Optional feedback/reason for the result (especially useful for failures).
        """
        self.action_history.append((action, success, feedback))

    def reset(self) -> None:
        """Reset the planner state for a new task."""
        self.action_history.clear()
        self.vlm_call_records.clear()

    def save_vlm_records(self, output_dir: Path) -> None:
        """Save all VLM call records to the output directory.

        Args:
            output_dir: Directory to save records to.
        """
        vlm_dir = output_dir / "vlm_calls"
        vlm_dir.mkdir(parents=True, exist_ok=True)

        for record in self.vlm_call_records:
            _write_vlm_call(
                vlm_dir,
                record,
                prefix=f"action_{record.action_index:03d}_",
                parsed=record.parsed_action,
                # The system prompt is identical across calls; write it once.
                write_system_prompt=record.action_index == 1,
            )

        # Save summary JSON
        summary = {
            "total_vlm_calls": len(self.vlm_call_records),
            "calls": [
                {
                    "action_index": r.action_index,
                    "parsed_action": r.parsed_action,
                }
                for r in self.vlm_call_records
            ],
        }
        summary_file = vlm_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved {len(self.vlm_call_records)} VLM call records to {vlm_dir}")


class OpenLoopVLMPlanner:
    """Vision-Language Model planner for robowits tasks.

    This planner generates a FULL plan upfront without taking feedback from the environment.
    It queries the VLM once to get all actions, then executes them in sequence.
    """

    SYSTEM_PROMPT = """You are a robotic manipulation planner. You have a dual-arm robot that can perform the following primitive actions:

AVAILABLE PRIMITIVES:
1. grasp(which_arm, obj_name) - Grasp an object with the specified arm ("left" or "right")
2. moveto(which_arm, obj_name, target_obj_name, offset) - Move obj_name to target_obj_name's top surface (gripper stays closed)
   - offset is optional [dx, dy, dz] in meters to offset from center of target surface
   - Use offset to avoid overlapping objects when placing multiple items on the same surface
   - Object remains held after moveto; use drop to release
3. drop(which_arm, obj_name) - Drop/release the held object from the specified arm (opens gripper)
4. rotate180(which_arm, obj_name) - Rotate the held object upside down (180 degrees)
5. rotate90(which_arm, obj_name, direction) - Rotate object 90 degrees in direction ("front", "back", "left", "right")
6. hit(which_arm, obj_A, obj_B) - Hit obj_B with obj_A (obj_A must be held)
7. done() - Call this when the task is completed

RULES:
- You must grasp an object before you can moveto, drop, rotate, or hit with it
- Each arm can only hold one object at a time
- Plan ALL actions needed to complete the task in one response
- Output a JSON array containing all actions in order

OUTPUT FORMAT:
You must respond with a JSON array of actions. Examples:
```json
[
    {"action": "grasp", "which_arm": "left", "obj_name": "cube"},
    {"action": "moveto", "which_arm": "left", "obj_name": "cube", "obj_name2": "table", "offset": [0.1, 0.0, 0.0]},
    {"action": "drop", "which_arm": "left", "obj_name": "cube"},
    {"action": "done"}
]
```

Another example with multiple steps:
```json
[
    {"action": "grasp", "which_arm": "right", "obj_name": "container"},
    {"action": "rotate180", "which_arm": "right", "obj_name": "container"},
    {"action": "moveto", "which_arm": "right", "obj_name": "container", "obj_name2": "bowl"},
    {"action": "drop", "which_arm": "right", "obj_name": "container"},
    {"action": "done"}
]
```

Think step by step about what needs to be done, then output the complete plan as a JSON array."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_steps: int = 20,
    ) -> None:
        """Initialize the VLM planner.

        Args:
            model: The model to use (default: gpt-4o).
            max_steps: Maximum number of planning steps.
        """
        self.model = model
        self.max_steps = max_steps
        self.action_plan: list[Action] = []  # Full plan of actions
        self.action_index: int = 0  # Current action index in the plan
        self.action_history: list[tuple[Action, bool, str]] = []  # (action, success, feedback) for logging
        self.vlm_call_record: VLMCallRecord | None = None  # Record of the VLM call

        # Initialize OpenAI client (uses OPENAI_API_KEY env var)
        try:
            from openai import OpenAI

            client = OpenAI()

            # Wrap with LangSmith tracing if available
            try:
                from langsmith import wrappers as langsmith_wrappers

                client = langsmith_wrappers.wrap_openai(client)
                print("LangSmith tracing enabled for OpenAI client")
            except ImportError:
                print("LangSmith not installed, skipping tracing")

            self.client = client
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: uv pip install 'openai>=1.0.0' "
                "(or install gs-gym with the 'baselines' extra)"
            ) from e

    def _format_obj_state(self, objs_info: dict[str, dict[str, Any]]) -> str:
        """Format object state information as readable text.

        Args:
            objs_info: Object information from collect_objs_info().

        Returns:
            Formatted string describing all objects.
        """
        lines = ["CURRENT OBJECT STATES:"]

        for name, info in objs_info.items():
            pos = info.get("pos", [0, 0, 0])
            material = info.get("material", "unknown")

            # Format position nicely
            if hasattr(pos, "flatten"):
                pos = pos.flatten()
            pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

            lines.append(f"- {name}: position={pos_str}, material={material}")

            # Add bounds if available
            if "bounds" in info:
                bounds = info["bounds"]
                size = bounds[1] - bounds[0]
                lines.append(f"    size=[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

        return "\n".join(lines)

    def _action_to_string(self, action: Action) -> str:
        """Convert an Action to a readable string."""
        if action.name == "done":
            return "done()"
        elif action.name == "grasp":
            return f"grasp({action.which_arm}, {action.obj_name})"
        elif action.name == "moveto":
            if action.offset:
                return f"moveto({action.which_arm}, {action.obj_name}, {action.obj_name2}, offset={action.offset})"
            return f"moveto({action.which_arm}, {action.obj_name}, {action.obj_name2})"
        elif action.name == "drop":
            return f"drop({action.which_arm}, {action.obj_name})"
        elif action.name == "rotate180":
            return f"rotate180({action.which_arm}, {action.obj_name})"
        elif action.name == "rotate90":
            return f"rotate90({action.which_arm}, {action.obj_name}, {action.direction})"
        elif action.name == "hit":
            return f"hit({action.which_arm}, {action.obj_name}, {action.obj_name2})"
        return f"unknown({action.name})"

    def _encode_images(self, images: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Encode images as base64 for the API.

        Args:
            images: Array of shape [3, 3, H, W] from get_three_images().

        Returns:
            List of image content dicts for the API.
        """
        image_contents = []
        camera_names = ["ego (front view)", "wrist_left", "wrist_right"]

        for i, cam_name in enumerate(camera_names):
            # images[i] is [3, H, W], convert to [H, W, 3]
            img_hwc = np.transpose(images[i], (1, 2, 0))
            pil_img = Image.fromarray(img_hwc)

            # Encode as base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            image_contents.append(
                {
                    "type": "text",
                    "text": f"Camera: {cam_name}",
                }
            )
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        return image_contents

    def _parse_action(self, action_dict: dict[str, Any]) -> Action:
        """Parse a single action dictionary to an Action object.

        Args:
            action_dict: Dictionary containing action parameters.

        Returns:
            Parsed Action object.
        """
        action_name = action_dict.get("action", "done")

        # Parse offset if present (dx, dy, dz)
        offset = None
        if "offset" in action_dict and action_dict["offset"] is not None:
            offset_list = action_dict["offset"]
            if isinstance(offset_list, list) and len(offset_list) >= 3:
                offset = (float(offset_list[0]), float(offset_list[1]), float(offset_list[2]))
            elif isinstance(offset_list, list) and len(offset_list) >= 2:
                # Backwards compatibility: if only 2 values provided, assume dz=0
                offset = (float(offset_list[0]), float(offset_list[1]), 0.0)

        return Action(
            name=action_name,
            which_arm=action_dict.get("which_arm"),
            obj_name=action_dict.get("obj_name") or action_dict.get("obj_A"),
            obj_name2=action_dict.get("obj_name2") or action_dict.get("obj_B"),
            direction=action_dict.get("direction"),
            offset=offset,
        )

    def _parse_plan(self, response_text: str) -> list[Action]:
        """Parse the VLM response to extract a list of actions.

        Args:
            response_text: Raw text response from the VLM.

        Returns:
            List of parsed Action objects.
        """
        # Try to find JSON array in the response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            print(f"Warning: Could not find JSON array in response: {response_text[:200]}")
            return [Action(name="done")]

        try:
            action_list = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            return [Action(name="done")]

        if not isinstance(action_list, list):
            print(f"Warning: Expected JSON array, got {type(action_list)}")
            return [Action(name="done")]

        actions = []
        for action_dict in action_list:
            if isinstance(action_dict, dict):
                actions.append(self._parse_action(action_dict))

        # Ensure plan ends with done if not already
        if not actions or actions[-1].name != "done":
            actions.append(Action(name="done"))

        return actions

    def make_full_plan(
        self,
        task_description: str,
        objs_info: dict[str, dict[str, Any]],
        images: NDArray[np.uint8],
    ) -> list[Action]:
        """Generate a complete plan for the task upfront.

        This method queries the VLM once to get ALL actions needed to complete
        the task, without any feedback from the environment.

        Args:
            task_description: Description of the task to accomplish.
            objs_info: Object information from collect_objs_info().
            images: Images from get_three_images().

        Returns:
            List of all Actions to execute.
        """
        # Build the user message
        obj_state_text = self._format_obj_state(objs_info)

        user_text = f"""TASK: {task_description}

{obj_state_text}

Based on the current state and images, create a complete plan to accomplish the task.
Output ALL actions needed as a JSON array. The plan will be executed without any modifications."""

        # Encode images
        image_contents = self._encode_images(images)

        # Build message content
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(image_contents)

        print("=== VLM PROMPT ===")
        print(f"System: {self.SYSTEM_PROMPT[:200]}...")
        print(f"User: {user_text}")
        print("=== END PROMPT ===")

        # Call the VLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )

            response_text = response.choices[0].message.content or ""
            print("=== VLM RESPONSE ===")
            print(response_text)
            print("=== END RESPONSE ===")

            self.action_plan = self._parse_plan(response_text)
            self.action_index = 0

            # Record the VLM call
            parsed_plan = [
                {
                    "name": a.name,
                    "which_arm": a.which_arm,
                    "obj_name": a.obj_name,
                    "obj_name2": a.obj_name2,
                    "direction": a.direction,
                    "offset": list(a.offset) if a.offset else None,
                }
                for a in self.action_plan
            ]
            self.vlm_call_record = VLMCallRecord(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_text,
                response=response_text,
                parsed_plan=parsed_plan,
                images=images.copy(),
            )

            print(f"Generated plan with {len(self.action_plan)} actions:")
            for i, action in enumerate(self.action_plan):
                print(f"  {i + 1}. {self._action_to_string(action)}")

            return self.action_plan

        except Exception as e:
            print(f"Error calling VLM: {e}, traceback: {traceback.format_exc()}")
            # Record the failed call
            self.vlm_call_record = VLMCallRecord(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_text,
                response=f"[Error: {e}]",
                parsed_plan=None,
                images=images.copy(),
            )
            self.action_plan = [Action(name="done")]
            self.action_index = 0
            return self.action_plan

    def get_next_action(self) -> Action | None:
        """Get the next action from the pre-generated plan.

        Returns:
            The next Action to execute, or None if plan is complete.
        """
        if self.action_index >= len(self.action_plan):
            return None

        action = self.action_plan[self.action_index]
        self.action_index += 1
        return action

    @property
    def has_plan(self) -> bool:
        """Check if a plan has been generated."""
        return len(self.action_plan) > 0

    @property
    def plan_complete(self) -> bool:
        """Check if the current plan is complete."""
        return self.action_index >= len(self.action_plan)

    def add_action_result(self, action: Action, success: bool, feedback: str = "") -> None:
        """Record an action and its result (for logging purposes only).

        Note: In open-loop mode, this feedback is NOT used for re-planning.

        Args:
            action: The action that was executed.
            success: Whether the action succeeded.
            feedback: Optional feedback/reason for the result.
        """
        self.action_history.append((action, success, feedback))

    def reset(self) -> None:
        """Reset the planner state for a new task."""
        self.action_plan.clear()
        self.action_index = 0
        self.action_history.clear()
        self.vlm_call_record = None

    def save_vlm_records(self, output_dir: Path) -> None:
        """Save VLM call record to the output directory.

        Args:
            output_dir: Directory to save records to.
        """
        if self.vlm_call_record is None:
            print("No VLM call record to save")
            return

        vlm_dir = output_dir / "vlm_calls"
        vlm_dir.mkdir(parents=True, exist_ok=True)

        _write_vlm_call(
            vlm_dir,
            self.vlm_call_record,
            prompt_stem="user_prompt",
            parsed=self.vlm_call_record.parsed_plan,
            parsed_stem="parsed_plan",
        )

        print(f"Saved VLM call record to {vlm_dir}")


# ==============================================================================
# Per-task evaluation loop
# ==============================================================================


def run_task_eval(
    task_name: str,
    output_dir: Path,
    mode: Literal["closed", "open"] = "closed",
    model: str = "gpt-4o",
    max_steps: int = 20,
    num_episodes: int = 1,
    eval_dataset_json: str | None = None,
    show_viewer: bool = False,
    overwrite: bool = False,
    genesis_backend: str = "cpu",
) -> dict[str, Any]:
    """Run the VLM oracle-primitive baseline on one gs-gym task.

    Args:
        task_name: Registered gs-gym task id (e.g. "robowits/02-retrieve-cube-v0").
        output_dir: Directory to write per-episode artifacts and summaries into.
        mode: "closed" replans after every primitive; "open" plans once up front.
        model: VLM model name.
        max_steps: Maximum primitives per episode.
        num_episodes: Number of episodes to evaluate.
        eval_dataset_json: Eval dataset JSON with per-episode initial placements,
            as used by the policy evals.
        show_viewer: Whether to show the Genesis viewer.
        overwrite: Re-evaluate episodes that already have results.
        genesis_backend: Genesis physics backend ("cpu" or "cuda"). The
            SPH/MPM tasks are impractically slow on cpu.

    Returns:
        The overall summary dict (also written to overall_summary.json).
    """
    planner_cls = OpenLoopVLMPlanner if mode == "open" else ClosedLoopVLMPlanner

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task: {task_name}  mode: {mode}")
    print(f"Output directory: {output_dir}")

    task_id = task_name
    all_episode_results: list[dict[str, Any]] = []
    all_episode_object_states: list[dict[str, Any]] = []

    # Check for existing completed episodes (for continue support)
    completed_episodes: set[int] = set()
    if not overwrite:
        for episode_idx in range(num_episodes):
            episode_dir = output_dir / f"episode_{episode_idx:03d}"
            summary_file = episode_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        episode_summary = json.load(f)
                    all_episode_results.append(
                        {
                            "episode_idx": episode_idx,
                            "success": episode_summary.get("success", False),
                            "total_steps": episode_summary.get("total_steps", 0),
                            "max_reward": episode_summary.get("max_reward", 0.0),
                        }
                    )
                    completed_episodes.add(episode_idx)
                except Exception as e:
                    print(f"Warning: Failed to load summary for episode {episode_idx}: {e}")

        if completed_episodes:
            print(f"Found {len(completed_episodes)} completed episodes, continuing from remaining...")
    else:
        print("Overwrite mode enabled, will re-evaluate all episodes")
        for episode_idx in range(num_episodes):
            episode_dir = output_dir / f"episode_{episode_idx:03d}"
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
                print(f"  Removed existing episode directory: {episode_dir}")
        overall_summary_file = output_dir / "overall_summary.json"
        if overall_summary_file.exists():
            overall_summary_file.unlink()
            print(f"  Removed existing overall summary: {overall_summary_file}")

    # Nothing left to do: report the existing results without building a scene.
    # Each Genesis scene build is expensive and, across a long sequential sweep,
    # eventually segfaults the process -- so a resume must not rebuild scenes for
    # variants it is only going to skip.
    if not overwrite and len(completed_episodes) >= num_episodes:
        print(f"All {num_episodes} episodes already complete for {task_name}; skipping env build")
        all_episode_results.sort(key=lambda x: x["episode_idx"])
        successes = [bool(r.get("success", False)) and not r.get("crashed", False) for r in all_episode_results]
        rewards = [float(r.get("max_reward", 0.0)) for r in all_episode_results]
        summary = {
            "task_id": task_id,
            "task_name": task_name,
            "model": model,
            "num_episodes": num_episodes,
            "completed_episodes": len(all_episode_results),
            "crashed_episodes": 0,
            "success_count": sum(successes),
            "success_rate": (sum(successes) / len(successes)) if successes else 0,
            "avg_max_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
            "episodes": all_episode_results,
        }
        with open(output_dir / "overall_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        write_eval_info(output_dir, task_name, successes, rewards)
        return summary

    evaluator = OracleControlEvaluator()
    evaluator.set_env_from_task(
        task_name,
        show_viewer=show_viewer,
        eval_dataset_json=eval_dataset_json,
        genesis_backend=genesis_backend,
    )

    # The natural-language instruction lives on each task env as `task_description`.
    task_instruction = evaluator._env.task_description
    print(f"Task Instruction: {task_instruction}")
    task_description = task_instruction

    crashed_episodes = []

    for episode_idx in range(num_episodes):
        # Skip already completed episodes
        if episode_idx in completed_episodes:
            print(f"\nSkipping episode {episode_idx + 1}/{num_episodes} (already completed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"EPISODE {episode_idx + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        try:
            # Reset environment for new episode
            evaluator.reset_episode(episode_idx)

            # Create episode output directory
            episode_dir = output_dir / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Print collected object info
            objs_info = evaluator.collect_objs_info()
            print("\nInitial object states:")
            for name, info in objs_info.items():
                print(f"  {name}: pos={info['pos']}")

            # Initialize VLM planner for this episode
            planner = planner_cls(model=model, max_steps=max_steps)

            last_result: bool | None = None
            step = 0
            episode_success = False
            # Progress score: max milestone reward reached, sampled after each
            # primitive (the granularity at which this baseline changes state).
            episode_max_reward = 0.0

            if mode == "open":
                # Open-loop: plan the whole sequence from the initial
                # observation, then execute it without any replanning.
                objs_info = evaluator.collect_objs_info()
                images = evaluator.get_three_images()
                print("\n=== Generating Full Plan ===")
                planner.make_full_plan(
                    task_description=task_description,
                    objs_info=objs_info,
                    images=images,
                )

            while step < max_steps:
                if mode == "open" and planner.plan_complete:
                    break
                step += 1
                print(f"\n=== Step {step} ===")

                if mode == "open":
                    # Get next action from the pre-generated plan
                    action = planner.get_next_action()
                    if action is None:
                        print("Plan complete, no more actions.")
                        break
                    print(f"Executing action: {planner._action_to_string(action)}")
                else:
                    # Closed-loop: re-observe and replan every step
                    objs_info = evaluator.collect_objs_info()
                    images = evaluator.get_three_images()
                    action = planner.plan_next_action(
                        task_description=task_description,
                        objs_info=objs_info,
                        images=images,
                        last_action_result=last_result,
                    )

                # Execute the action
                result = False
                feedback = ""
                try:
                    if action.name == "grasp":
                        if action.which_arm and action.obj_name:
                            result, feedback = evaluator.primitive_grasp(action.which_arm, action.obj_name)
                        else:
                            feedback = "Missing required parameters: which_arm or obj_name"
                    elif action.name == "moveto":
                        if action.which_arm and action.obj_name and action.obj_name2:
                            result, feedback = evaluator.primitive_moveto(
                                action.which_arm,
                                action.obj_name,
                                action.obj_name2,
                                offset=action.offset,
                            )
                        else:
                            feedback = "Missing required parameters: which_arm, obj_name, or obj_name2"
                    elif action.name == "drop":
                        if action.which_arm and action.obj_name:
                            result, feedback = evaluator.primitive_drop(action.which_arm, action.obj_name)
                        else:
                            feedback = "Missing required parameters: which_arm or obj_name"
                    elif action.name == "rotate180":
                        if action.which_arm and action.obj_name:
                            result, feedback = evaluator.primitive_rotate180(action.which_arm, action.obj_name)
                        else:
                            feedback = "Missing required parameters: which_arm or obj_name"
                    elif action.name == "rotate90":
                        if action.which_arm and action.obj_name and action.direction:
                            result, feedback = evaluator.primitive_rotate90(
                                action.which_arm, action.obj_name, action.direction
                            )
                        else:
                            feedback = "Missing required parameters: which_arm, obj_name, or direction"
                    elif action.name == "hit":
                        if action.which_arm and action.obj_name and action.obj_name2:
                            result, feedback = evaluator.primitive_hit(
                                action.which_arm, action.obj_name, action.obj_name2
                            )
                        else:
                            feedback = "Missing required parameters: which_arm, obj_name, or obj_name2"
                    # If VLM says done, check actual task completion
                    elif action.name == "done":
                        feedback = "Task is not actually completed"
                        result = False
                    else:
                        feedback = f"Unknown action: {action.name}"
                except Exception as e:
                    print(f"Error executing action: {e}, traceback: {traceback.format_exc()}")
                    result = False
                    feedback = f"Exception during execution: {str(e)}"

                reward, _ = evaluator._env.get_reward()
                episode_max_reward = max(episode_max_reward, float(reward.item()))

                print(f"Action result: {result}, feedback: {feedback}")
                last_result = result

                # Record the result with feedback
                planner.add_action_result(action, result, feedback)

                # Check if task is completed after this action
                if evaluator.is_task_completed():
                    print("Task completed after this action!")
                    episode_success = True
                    # Save final image and state before breaking
                    evaluator.save_image(str(episode_dir / f"{step:04d}.png"))
                    objs_info = evaluator.collect_objs_info()
                    step_state = {
                        "step": step,
                        "action": {
                            "name": action.name,
                            "which_arm": action.which_arm,
                            "obj_name": action.obj_name,
                            "obj_name2": action.obj_name2,
                            "direction": action.direction,
                            "offset": list(action.offset) if action.offset else None,
                        },
                        "result": result,
                        "feedback": feedback,
                        "task_completed": True,
                        "objs_info": {
                            name: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in info.items()}
                            for name, info in objs_info.items()
                        },
                    }
                    with open(episode_dir / f"{step:04d}_state.json", "w") as f:
                        json.dump(step_state, f, indent=2)
                    break

                # Save image for this step
                evaluator.save_image(str(episode_dir / f"{step:04d}.png"))

                # Save state for this step
                step_state = {
                    "step": step,
                    "action": {
                        "name": action.name,
                        "which_arm": action.which_arm,
                        "obj_name": action.obj_name,
                        "obj_name2": action.obj_name2,
                        "direction": action.direction,
                        "offset": list(action.offset) if action.offset else None,
                    },
                    "result": result,
                    "feedback": feedback,
                    "objs_info": {
                        name: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in info.items()}
                        for name, info in objs_info.items()
                    },
                }
                with open(episode_dir / f"{step:04d}_state.json", "w") as f:
                    json.dump(step_state, f, indent=2)

            print(f"\nEpisode {episode_idx + 1} completed in {step} steps, success={episode_success}")

            # Collect final object states for this episode
            final_objs_info = evaluator.collect_objs_info()
            episode_object_state = {
                "episode_index": episode_idx,
                "episode_success": episode_success,
                "task_name": task_instruction,
                "objects_info": {
                    name: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in info.items()}
                    for name, info in final_objs_info.items()
                },
            }
            all_episode_object_states.append(episode_object_state)

            # Save episode summary
            episode_summary = {
                "episode_idx": episode_idx,
                "task_id": task_id,
                "task_name": task_name,
                "task_instruction": task_instruction,
                "model": model,
                "total_steps": step,
                "success": episode_success,
                "max_reward": episode_max_reward,
                "action_history": [
                    {
                        "action": planner._action_to_string(a),
                        "result": r,
                        "feedback": f,
                    }
                    for a, r, f in planner.action_history
                ],
            }
            with open(episode_dir / "summary.json", "w") as f:
                json.dump(episode_summary, f, indent=2)

            # Save VLM call records
            planner.save_vlm_records(episode_dir)

            all_episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "success": episode_success,
                    "total_steps": step,
                    "max_reward": episode_max_reward,
                }
            )

        except Exception as e:
            print(f"\n{'!' * 60}")
            print(f"EPISODE {episode_idx + 1}/{num_episodes} CRASHED")
            print(f"Error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"{'!' * 60}")

            crashed_episodes.append(episode_idx)

            # Record crashed episode
            all_episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "success": False,
                    "total_steps": 0,
                    "crashed": True,
                    "error": str(e),
                }
            )

            # Try to save crash info to episode directory
            try:
                episode_dir = output_dir / f"episode_{episode_idx:03d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                crash_info = {
                    "episode_idx": episode_idx,
                    "crashed": True,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                with open(episode_dir / "crash_info.json", "w") as f:
                    json.dump(crash_info, f, indent=2)
            except Exception:
                pass  # Best effort to save crash info

            print("Continuing to next episode...\n")
            continue

    # Save overall evaluation summary
    # Sort results by episode index
    all_episode_results.sort(key=lambda x: x["episode_idx"])
    completed_count = len(all_episode_results)
    crashed_count = len(crashed_episodes)
    success_count = sum(1 for r in all_episode_results if r.get("success", False) and not r.get("crashed", False))
    overall_summary = {
        "task_id": task_id,
        "task_name": task_name,
        "task_instruction": task_instruction,
        "model": model,
        "num_episodes": num_episodes,
        "completed_episodes": completed_count,
        "crashed_episodes": crashed_count,
        "success_count": success_count,
        "success_rate": success_count / (completed_count - crashed_count)
        if (completed_count - crashed_count) > 0
        else 0,
        "episodes": all_episode_results,
    }
    with open(output_dir / "overall_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    # Save episode object states (similar format to vla eval)
    with open(output_dir / "episode_object_states.json", "w") as f:
        json.dump(all_episode_object_states, f, indent=2)

    # Also emit eval_info.json so scripts/robowits/eval/summarize_results.py
    # picks these runs up alongside the policy evals. Crashed episodes count as
    # failures rather than being dropped.
    write_eval_info(
        output_dir,
        task_name,
        [bool(r.get("success", False)) and not r.get("crashed", False) for r in all_episode_results],
        [float(r.get("max_reward", 0.0)) for r in all_episode_results],
    )

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Completed: {completed_count}/{num_episodes} episodes")
    if crashed_count > 0:
        print(f"Crashed: {crashed_count} episodes")
    valid_count = completed_count - crashed_count
    if valid_count > 0:
        print(f"Success rate: {success_count}/{valid_count} ({100 * success_count / valid_count:.1f}%)")
    else:
        print("No valid episodes completed")
    print(f"Results saved to {output_dir}")

    return overall_summary


# ==============================================================================
# Command line interface
# ==============================================================================


# Suite names come from summarize_results.SUITES so the seed and mutation paths
# cannot drift apart (they did once: --suite rejected "normal" while
# --mutations accepted it).
NAMED_SUITES = SUITES


def _successes_from_metrics(metrics: dict) -> list[bool]:
    """Per-episode success flags from a planner summary."""
    return [bool(e.get("success", False)) and not e.get("crashed", False) for e in metrics.get("episodes", [])]


def _rewards_from_metrics(metrics: dict) -> list[float]:
    """Per-episode progress scores (max milestone reward)."""
    return [float(e.get("max_reward", 0.0)) for e in metrics.get("episodes", [])]


def run_mutations(args, run_one, output_root):
    """Run every mutation variant of the requested base tasks.

    Each variant is evaluated into `<output_root>/mutation_<task>/<variant>/`, and
    the variants of a base task are then pooled into one
    `mutation_<task>/eval_info.json` -- the layout summarize_results.py expects
    for a mutation suite.

    Args:
        args: Parsed CLI args (needs .eval_dataset_json, .num_episodes).
        run_one: Callable(task_name, output_dir, eval_dataset_json) -> metrics dict.
        output_root: Base results directory.

    Returns:
        Process exit code (non-zero if any variant crashed).
    """
    eval_json_dir = args.eval_dataset_json or "dataset/robowits/eval_dataset_mutation"
    base_tasks = resolve_suite(args.mutations)

    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    any_succeeded = False

    for task_id in base_tasks:
        variants, missing = variants_with_dataset(task_id, eval_json_dir)
        skipped.extend(missing)
        if missing:
            print(f"task {task_id}: skipping {len(missing)} variant(s) with no eval dataset: {', '.join(missing)}")
        if not variants:
            print(f"task {task_id}: no mutation variants with eval data; skipping")
            continue

        group_dir = output_root / f"mutation_{task_id}"
        entries: list[tuple[str, list[bool]]] = []

        for task_name in variants:
            print(f"\n{'=' * 60}\n{task_name}\n{'=' * 60}")
            try:
                metrics = run_one(
                    task_name,
                    group_dir / variant_stem(task_name),
                    str(dataset_file(task_name, eval_json_dir)),
                )
            except Exception as e:  # one bad variant must not lose the sweep
                print(f"\n{'!' * 60}\nVARIANT CRASHED: {task_name}\nError: {e}\n{'!' * 60}")
                failed.append((task_name, str(e)))
                continue
            entries.append((task_name, _successes_from_metrics(metrics), _rewards_from_metrics(metrics)))

        if entries:
            any_succeeded = True
            path = write_grouped_eval_info(group_dir, f"robowits_mutation_{task_id}", entries)
            pooled = [s for _, ss, _ in entries for s in ss]
            rate = 100.0 * sum(pooled) / len(pooled) if pooled else 0.0
            print(f"\ntask {task_id}: {len(entries)} variants, pooled SR {rate:.1f}% -> {path}")

    if skipped:
        print(f"\nSkipped {len(skipped)} variant(s) lacking eval data: {', '.join(skipped)}")
    if failed:
        print(f"\nCrashed variants ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")

    # Exit non-zero only if nothing at all was evaluated. A sweep that completed
    # most variants is not a failed job: returning 1 for any individual crash
    # made slurm mark multi-hour sweeps FAILED and hid that they had finished.
    if failed and not any_succeeded:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="Task id ('02') or registered task name ('robowits/02_03-v0')")
    group.add_argument("--suite", choices=sorted(NAMED_SUITES), help="Named task suite to sweep")
    group.add_argument(
        "--mutations",
        help="Run mutation variants for a suite ('robowits_10', 'robowits_20', 'all') or a task id like '02'",
    )
    parser.add_argument("--mode", choices=["closed", "open"], default="closed", help="Planning mode")
    parser.add_argument("--model", default="gpt-4o", help="VLM model name")
    parser.add_argument("--num-episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=20, help="Max primitives per episode")
    parser.add_argument("--output-dir", type=Path, default=None, help="Results directory")
    parser.add_argument(
        "--eval-dataset-json",
        default=None,
        help="Eval dataset JSON with per-episode initial placements. For a suite sweep, "
        "pass a directory containing <task_id>.json.",
    )
    parser.add_argument(
        "--genesis-backend",
        default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="Genesis physics backend. 'auto' is cuda with Nyx cameras (the default), else cpu; use 'cuda' for the "
        "slow SPH/MPM tasks.",
    )
    parser.add_argument("--show-viewer", action="store_true", help="Show the Genesis viewer")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate existing episodes")
    args = parser.parse_args()
    if args.genesis_backend == "auto":
        args.genesis_backend = "cuda" if os.environ.get("GS_GYM_CAMERA_TYPE") == "nyx" else "cpu"

    output_root = args.output_dir or Path(f"tmp/eval/vlm_planner_{args.mode}")

    if args.mutations:

        def _run_one(task_name, out_dir, ds_json):
            return run_task_eval(
                task_name=task_name,
                output_dir=out_dir,
                mode=args.mode,
                model=args.model,
                max_steps=args.max_steps,
                num_episodes=args.num_episodes,
                eval_dataset_json=ds_json,
                show_viewer=args.show_viewer,
                overwrite=args.overwrite,
                genesis_backend=args.genesis_backend,
            )

        return run_mutations(args, _run_one, output_root)

    task_ids = resolve_suite(args.suite) if args.suite else [args.task]

    results: dict[str, dict] = {}
    failed: list[tuple[str, str]] = []

    for task_id in task_ids:
        task_name = resolve_task(task_id)

        # A suite sweep takes a directory of per-task eval datasets; a single
        # task takes the file directly.
        eval_dataset_json = args.eval_dataset_json
        if eval_dataset_json is not None and Path(eval_dataset_json).is_dir():
            candidate = Path(eval_dataset_json) / f"{task_id}.json"
            eval_dataset_json = str(candidate) if candidate.exists() else None
            if eval_dataset_json is None:
                print(f"No eval dataset for task {task_id} in {args.eval_dataset_json}; using random placement")

        print(f"\n{'=' * 60}\n{task_name}\n{'=' * 60}")
        try:
            summary = run_task_eval(
                task_name=task_name,
                output_dir=output_root / task_id.replace("/", "_"),
                mode=args.mode,
                model=args.model,
                max_steps=args.max_steps,
                num_episodes=args.num_episodes,
                eval_dataset_json=eval_dataset_json,
                show_viewer=args.show_viewer,
                overwrite=args.overwrite,
                genesis_backend=args.genesis_backend,
            )
            results[task_id] = summary
        except Exception as e:  # a crashing task must not abort the sweep
            print(f"\n{'!' * 60}\nTASK CRASHED: {task_name}\nError: {e}\n{'!' * 60}")
            failed.append((task_id, str(e)))

    if len(task_ids) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        sweep = {
            "mode": args.mode,
            "model": args.model,
            "num_episodes": args.num_episodes,
            "tasks": {tid: {"success_rate": s.get("success_rate")} for tid, s in results.items()},
            "failed_tasks": dict(failed),
        }
        with open(output_root / "sweep_summary.json", "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"\nSweep summary written to {output_root / 'sweep_summary.json'}")
        for tid, s in results.items():
            print(f"  {tid}: {s.get('success_rate', 0) * 100:.1f}%")
        for tid, err in failed:
            print(f"  {tid}: CRASHED ({err})")

    # Same convention as run_mutations: a sweep that evaluated most tasks is not
    # a failed job. Exit non-zero only when nothing at all produced results.
    if failed and not results:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
