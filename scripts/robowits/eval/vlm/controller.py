#!/usr/bin/env python
"""Evaluate the VLM closed-loop controller baseline on RoboWits tasks.

A VLM issues high-level actions (grasp, move-to, open/close gripper, hold,
return-home) from camera images and object state; a low-level controller turns
each into an end-effector waypoint trajectory executed as EE_ABS actions.

Unlike scripts/robowits/eval/vlm/planner.py, which uses oracle primitives
that teleport objects, this baseline actually drives the arms through IK.

Usage:
    OPENAI_API_KEY=... python scripts/robowits/eval/vlm/controller.py --task 02
    python scripts/robowits/eval/vlm/controller.py --suite robowits_10 --num-episodes 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from gs_gym.common.utils import np_math_utils as npm
from gs_gym.envs.registry import TASK_REGISTRY

# summarize_results.py lives in the parent eval/ directory and is not an
# installed module: it owns the suite definitions and the eval_info.json writers,
# shared with planner.py.
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

logger = logging.getLogger(__name__)

# The VLM baselines render with Nyx unless GS_GYM_CAMERA_TYPE says otherwise (RoboWitsEnv reads it); Nyx runs on
# the CUDA backend, which "auto" picks for it.
os.environ.setdefault("GS_GYM_CAMERA_TYPE", "nyx")


# ==============================================================================
# Pose types
# ==============================================================================


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RobotBaseRefFrame:
    """Marker for poses expressed relative to the robot base."""


@dataclass(frozen=True)
class EulerAngle:
    """Yaw-pitch-roll intrinsic Euler angles (z-y'-x" Tait-Bryan), radians."""

    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class Vector3D:
    """Vector in 3D space (meters)."""

    x: float
    y: float
    z: float

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Vector3D:
        return cls(x=float(array[0]), y=float(array[1]), z=float(array[2]))

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class UnitQuaternion:
    """Unit quaternion (w, x, y, z), normalised on construction."""

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        norm = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if norm == 0:
            object.__setattr__(self, "w", 1.0)
            object.__setattr__(self, "x", 0.0)
            object.__setattr__(self, "y", 0.0)
            object.__setattr__(self, "z", 0.0)
        elif not math.isclose(norm, 1.0):
            object.__setattr__(self, "w", self.w / norm)
            object.__setattr__(self, "x", self.x / norm)
            object.__setattr__(self, "y", self.y / norm)
            object.__setattr__(self, "z", self.z / norm)

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> UnitQuaternion:
        return cls(w=float(array[0]), x=float(array[1]), y=float(array[2]), z=float(array[3]))

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def to_euler_angle(self) -> EulerAngle:
        roll, pitch, yaw = npm.quat_to_euler(self.to_numpy())
        return EulerAngle(roll=float(roll), pitch=float(pitch), yaw=float(yaw))

    def to_axis_angle(self) -> npt.NDArray[np.float64]:
        """Axis-angle (rotation vector), the form the EE_ABS actions expect."""
        return npm.quat_to_axis_angle(self.to_numpy())


@dataclass(frozen=True)
class Pose6D:
    """6D pose: position plus rotation, in a named reference frame."""

    vector: Vector3D
    rotation: UnitQuaternion
    ref_frame: RobotBaseRefFrame

    def __class_getitem__(cls, _item: object) -> type[Pose6D]:
        # The controller annotates poses as Pose6D[UnitQuaternion, RobotBaseRefFrame];
        # accept and ignore the parameters so those annotations stay valid.
        return cls


# ==============================================================================
# Low-level controller and high-level VLM planner
# ==============================================================================


# gs-gym EE_ABS action layout:
#   [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
# Positions are in the robot base frame, which is what the controller already
# works in (its DEFAULT_ROBOT_BASE_POS matches the measured robot base offset),
# so no frame conversion is needed on the action path.
EE_ABS_ACTION_DIM = 14


# =============================================================================
# Direction Rotation Mapping
# =============================================================================

# Format: '{gripper_towards}_{gripper_open}_{camera_location}' -> (euler_x, euler_y, euler_z) in degrees
DIRECTIONS_TO_ROTATION: dict[str, tuple[float, float, float]] = {
    "front_left-right_up": (90, 0, 90),
    "front_left-right_down": (-90, 0, -90),
    "front_up-down_left": (0, 90, 0),
    "front_up-down_right": (90, -90, 90),
    "left_front-back_up": (90, 0, 180),
    "left_front-back_down": (-90, 0, 0),
    "left_up-down_back": (0, 90, 90),
    "left_up-down_front": (90, -90, 180),
    "back_left-right_up": (90, 0, -90),
    "back_left-right_down": (-90, 0, 90),
    "back_up-down_left": (90, -90, -90),
    "back_up-down_right": (0, 90, 180),
    "right_front-back_up": (90, 0, 0),
    "right_front-back_down": (-90, 0, 180),
    "right_up-down_back": (90, -90, 0),
    "right_up-down_front": (0, 90, -90),
    "down_front-back_left": (0, 180, 0),
    "down_front-back_right": (0, 180, 180),
    "down_left-right_back": (0, 180, 90),
    "down_left-right_front": (0, 180, -90),
    "up_front-back_left": (0, 0, 0),
    "up_front-back_right": (0, 0, 180),
    "up_left-right_back": (0, 0, 90),
    "up_left-right_front": (0, 0, -90),
}


def direction_to_approach_vector(grasping_direction: str) -> npt.NDArray[np.float64]:
    """Convert grasping direction string to approach unit vector."""
    gripper_towards = grasping_direction.split("_")[0].lower()

    direction_map = {
        "down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "front": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "back": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }
    if gripper_towards not in direction_map:
        raise ValueError(f"Unknown gripper_towards direction: '{gripper_towards}'")
    return direction_map[gripper_towards]


def direction_to_quaternion(grasping_direction: str) -> UnitQuaternion:
    """Convert grasping direction string to gripper orientation quaternion."""
    if grasping_direction not in DIRECTIONS_TO_ROTATION:
        raise ValueError(f"Unknown grasping_direction: '{grasping_direction}'")

    euler_deg = np.array(DIRECTIONS_TO_ROTATION[grasping_direction], dtype=np.float64)
    euler_rad = np.deg2rad(euler_deg)
    quat_np = npm.euler_to_quat(euler_rad)
    return UnitQuaternion.from_numpy(quat_np)


def axis_angle_to_quaternion(axis_angle: npt.NDArray[np.float64]) -> UnitQuaternion:
    """Convert an axis-angle vector [wx, wy, wz] (magnitude = angle) to a UnitQuaternion."""
    return UnitQuaternion.from_numpy(npm.axis_angle_to_quat(axis_angle))


# =============================================================================
# Type Aliases
# =============================================================================

# EEF waypoint: (left_pose, right_pose, left_gripper, right_gripper)
EEFWaypoint = tuple[
    Pose6D[UnitQuaternion, RobotBaseRefFrame],
    Pose6D[UnitQuaternion, RobotBaseRefFrame],
    float,
    float,
]


class Arm(Enum):
    """Which arm to use."""

    LEFT = auto()
    RIGHT = auto()


# =============================================================================
# High-Level Actions
# =============================================================================


@dataclass
class HighLevelAction(ABC):
    """Base class for high-level actions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the action."""
        ...


@dataclass
class GraspObjectAction(HighLevelAction):
    """Move to object and close gripper to grasp it."""

    arm: Arm
    object_name: str
    grasping_direction: str = "down_left-right_front"
    pre_grasp_steps: int = 40
    grasp_steps: int = 30
    offs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dlt_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Degrees
    gripper: str = ""  # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed

    @property
    def name(self) -> str:
        return f"grasp({self.arm.name.lower()}, {self.object_name}, {self.grasping_direction})"


@dataclass
class MoveToAction(HighLevelAction):
    """Move arm by delta position and/or delta euler angles."""

    arm: Arm
    dlt_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dlt_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Degrees
    steps: int = 50
    gripper: str = ""  # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed

    @property
    def name(self) -> str:
        return f"moveto({self.arm.name.lower()}, dlt_pos={self.dlt_pos}, dlt_euler={self.dlt_euler})"


@dataclass
class MoveToObjectAction(HighLevelAction):
    """Move arm to an object position with offset."""

    arm: Arm
    object_name: str
    dlt_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dlt_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Degrees
    steps: int = 50
    gripper: str = ""  # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed

    @property
    def name(self) -> str:
        return f"moveto_object({self.arm.name.lower()}, {self.object_name}, dlt_pos={self.dlt_pos}, dlt_euler={self.dlt_euler})"


@dataclass
class MoveToPositionAction(HighLevelAction):
    """Move arm to a target position (absolute)."""

    arm: Arm
    target_position: npt.NDArray[np.float64]
    target_rotation: UnitQuaternion | None = None
    dlt_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Degrees
    keep_gripper: bool = True  # If True, maintain current gripper state
    steps: int = 40
    gripper: str = ""  # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed
    is_world_frame: bool = False  # If True, target_position is in world frame and needs conversion

    @property
    def name(self) -> str:
        return f"move_to({self.arm.name.lower()}, {self.target_position})"


@dataclass
class OpenGripperAction(HighLevelAction):
    """Open the gripper."""

    arm: Arm
    steps: int = 15

    @property
    def name(self) -> str:
        return f"open_gripper({self.arm.name.lower()})"


@dataclass
class CloseGripperAction(HighLevelAction):
    """Close the gripper."""

    arm: Arm
    steps: int = 15

    @property
    def name(self) -> str:
        return f"close_gripper({self.arm.name.lower()})"


@dataclass
class HoldAction(HighLevelAction):
    """Hold current position for a number of steps."""

    steps: int = 10
    gripper: str = ""  # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed

    @property
    def name(self) -> str:
        return f"hold({self.steps})"


@dataclass
class ReturnToHomeAction(HighLevelAction):
    """Return arm to initial/home position."""

    arm: Arm
    steps: int = 40

    @property
    def name(self) -> str:
        return f"return_home({self.arm.name.lower()})"


# =============================================================================
# Low-Level Controller
# =============================================================================


@dataclass
class LowLevelControllerConfig:
    """Configuration for the low-level controller."""

    gripper_width_range: tuple[float, float] = (0.0, 0.08)
    default_approach_steps: int = 30
    default_grasp_steps: int = 20
    default_move_steps: int = 40
    default_gripper_steps: int = 15
    default_hold_steps: int = 10


class LowLevelController:
    """Converts high-level actions to EEF waypoints.

    This controller maintains the current arm state and generates
    interpolated trajectories for each high-level action.
    """

    # Default robot base position in world frame (from MORPH_MARVIN_BIMANUAL_DEFAULT)
    DEFAULT_ROBOT_BASE_POS: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1.08], dtype=np.float64)

    def __init__(self, config: LowLevelControllerConfig | None = None) -> None:
        self.config = config or LowLevelControllerConfig()

        # Current arm state
        self._pose_left: Pose6D[UnitQuaternion, RobotBaseRefFrame] | None = None
        self._pose_right: Pose6D[UnitQuaternion, RobotBaseRefFrame] | None = None
        self._gripper_left: float = self.config.gripper_width_range[1]
        self._gripper_right: float = self.config.gripper_width_range[1]

        # Initial (home) positions
        self._initial_pose_left: Pose6D[UnitQuaternion, RobotBaseRefFrame] | None = None
        self._initial_pose_right: Pose6D[UnitQuaternion, RobotBaseRefFrame] | None = None

        # Object info cache
        self._objs_info: dict[str, dict[str, Any]] = {}

        # Robot base position in world frame (for coordinate conversion)
        self._robot_base_pos: npt.NDArray[np.float64] = self.DEFAULT_ROBOT_BASE_POS.copy()

        # Current trajectory being executed
        self._trajectory: list[EEFWaypoint] = []
        self._trajectory_index: int = 0

    def reset(self) -> None:
        """Reset the controller state."""
        self._pose_left = None
        self._pose_right = None
        self._gripper_left = self.config.gripper_width_range[1]
        self._gripper_right = self.config.gripper_width_range[1]
        self._initial_pose_left = None
        self._initial_pose_right = None
        self._objs_info = {}
        self._robot_base_pos = self.DEFAULT_ROBOT_BASE_POS.copy()
        self._trajectory = []
        self._trajectory_index = 0

    def update_state(
        self,
        pose_left: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        pose_right: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        gripper_left: float,
        gripper_right: float,
        objs_info: dict[str, dict[str, Any]],
        robot_base_pos: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Update controller state from environment observation.

        Args:
            pose_left: Left arm pose in robot base frame
            pose_right: Right arm pose in robot base frame
            gripper_left: Left gripper width
            gripper_right: Right gripper width
            objs_info: Object info dictionary (positions are in world frame)
            robot_base_pos: Robot base position in world frame. If None, uses default.
        """
        self._pose_left = pose_left
        self._pose_right = pose_right
        self._gripper_left = gripper_left
        self._gripper_right = gripper_right
        self._objs_info = objs_info
        if robot_base_pos is not None:
            self._robot_base_pos = robot_base_pos

        # Save initial poses on first update
        if self._initial_pose_left is None:
            self._initial_pose_left = pose_left
            self._initial_pose_right = pose_right

    def set_trajectory(self, trajectory: list[EEFWaypoint]) -> None:
        """Set a new trajectory to execute."""
        self._trajectory = trajectory
        self._trajectory_index = 0

    @property
    def trajectory_done(self) -> bool:
        """Check if current trajectory is complete."""
        return self._trajectory_index >= len(self._trajectory)

    @property
    def trajectory_length(self) -> int:
        """Get the length of the current trajectory."""
        return len(self._trajectory)

    @property
    def trajectory_progress(self) -> int:
        """Get the current trajectory index."""
        return self._trajectory_index

    def get_next_waypoint(self) -> EEFWaypoint:
        """Get the next waypoint from the trajectory."""
        if self._trajectory_index < len(self._trajectory):
            waypoint = self._trajectory[self._trajectory_index]
            self._trajectory_index += 1
            # Update internal state to match waypoint
            self._pose_left = waypoint[0]
            self._pose_right = waypoint[1]
            self._gripper_left = waypoint[2]
            self._gripper_right = waypoint[3]
            return waypoint
        else:
            # Return current state as hold
            return (
                self._pose_left,  # type: ignore
                self._pose_right,  # type: ignore
                self._gripper_left,
                self._gripper_right,
            )

    # -------------------------------------------------------------------------
    # Helper methods for trajectory generation
    # -------------------------------------------------------------------------

    @staticmethod
    def _slerp_quaternion(q1: UnitQuaternion, q2: UnitQuaternion, t: float) -> UnitQuaternion:
        """Spherical linear interpolation between two quaternions."""
        q1_np = q1.to_numpy()
        q2_np = q2.to_numpy()

        dot = np.clip(np.dot(q1_np, q2_np), -1.0, 1.0)

        if abs(dot) > 0.9995:
            result = q1_np + t * (q2_np - q1_np)
            result = result / np.linalg.norm(result)
            return UnitQuaternion.from_numpy(result)

        if dot < 0:
            q2_np = -q2_np
            dot = -dot

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        result = w1 * q1_np + w2 * q2_np
        result = result / np.linalg.norm(result)
        return UnitQuaternion.from_numpy(result)

    def _interpolate_pose(
        self,
        start: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        end: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        t: float,
    ) -> Pose6D[UnitQuaternion, RobotBaseRefFrame]:
        """Interpolate between two poses."""
        pos_start = start.vector.to_numpy()
        pos_end = end.vector.to_numpy()
        pos_interp = pos_start * (1 - t) + pos_end * t

        rot_interp = self._slerp_quaternion(start.rotation, end.rotation, t)

        return Pose6D(
            vector=Vector3D.from_numpy(pos_interp),
            rotation=rot_interp,
            ref_frame=RobotBaseRefFrame(),
        )

    def _make_pose(
        self,
        position: npt.NDArray[np.float64],
        rotation: UnitQuaternion,
    ) -> Pose6D[UnitQuaternion, RobotBaseRefFrame]:
        """Create a Pose6D from position and rotation.

        Note: The position should already be in robot base frame.
        """
        return Pose6D(
            vector=Vector3D.from_numpy(position.astype(np.float64)),
            rotation=rotation,
            ref_frame=RobotBaseRefFrame(),
        )

    def _world_to_robot_base_pos(self, pos_world: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a position from world frame to robot base frame.

        Since the robot base has identity rotation (euler=(0,0,0)), the conversion
        is simply subtracting the robot base position from the world position.

        Args:
            pos_world: Position in world frame (x, y, z)

        Returns:
            Position in robot base frame (x, y, z)
        """
        return pos_world - self._robot_base_pos

    def _get_object_position(self, object_name: str) -> npt.NDArray[np.float64] | None:
        """Get the position of an object in robot base frame.

        Object positions from objs_info are in world frame, so we convert
        them to robot base frame for trajectory generation.
        """
        if object_name not in self._objs_info:
            return None

        obj_info = self._objs_info[object_name]
        if "pos" not in obj_info:
            return None

        pos = obj_info["pos"]
        if isinstance(pos, np.ndarray):
            # For particle objects, pos is (N, 3) - return center
            pos_world = pos.mean(axis=0).astype(np.float64) if pos.ndim == 2 else pos.astype(np.float64)
        elif isinstance(pos, list | tuple):
            pos_world = np.array(pos, dtype=np.float64)
        elif isinstance(pos, torch.Tensor):
            pos_np = pos.cpu().numpy()
            pos_world = pos_np.mean(axis=0).astype(np.float64) if pos_np.ndim == 2 else pos_np.astype(np.float64)
        else:
            return None

        # Convert from world frame to robot base frame
        return self._world_to_robot_base_pos(pos_world)

    def _get_object_bounds(self, object_name: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Get the AABB bounds of an object in robot base frame.

        Bounds from objs_info are in world frame, so we convert them
        to robot base frame for trajectory generation.
        """
        if object_name not in self._objs_info:
            return None

        obj_info = self._objs_info[object_name]

        # Check for bounds field (AABB)
        if "bounds" in obj_info and obj_info["bounds"] is not None:
            bounds = obj_info["bounds"]
            if isinstance(bounds, np.ndarray) and bounds.shape == (2, 3):
                aabb_min_world = bounds[0].astype(np.float64)
                aabb_max_world = bounds[1].astype(np.float64)
                # Convert from world frame to robot base frame
                return (
                    self._world_to_robot_base_pos(aabb_min_world),
                    self._world_to_robot_base_pos(aabb_max_world),
                )

        # For particle objects, compute bounds from positions
        if obj_info.get("material") == "particle" and "pos" in obj_info:
            pos = obj_info["pos"]
            if isinstance(pos, np.ndarray) and pos.ndim == 2:
                aabb_min_world = pos.min(axis=0).astype(np.float64)
                aabb_max_world = pos.max(axis=0).astype(np.float64)
            elif isinstance(pos, torch.Tensor) and pos.ndim == 2:
                pos_np = pos.cpu().numpy()
                aabb_min_world = pos_np.min(axis=0).astype(np.float64)
                aabb_max_world = pos_np.max(axis=0).astype(np.float64)
            else:
                return None
            # Convert from world frame to robot base frame
            return (
                self._world_to_robot_base_pos(aabb_min_world),
                self._world_to_robot_base_pos(aabb_max_world),
            )

        return None

    # -------------------------------------------------------------------------
    # High-level action execution
    # -------------------------------------------------------------------------

    def execute_action(self, action: HighLevelAction) -> list[EEFWaypoint]:
        """Execute a high-level action and return the trajectory."""
        match action:
            case GraspObjectAction():
                return self.grasp_object(action)
            case MoveToAction():
                return self.moveto(action)
            case MoveToObjectAction():
                return self.moveto_object(action)
            case MoveToPositionAction():
                return self.move_to_position(action)
            case OpenGripperAction():
                return self.open_gripper(action)
            case CloseGripperAction():
                return self.close_gripper(action)
            case HoldAction():
                return self.hold(action)
            case ReturnToHomeAction():
                return self.return_to_home(action)
            case _:
                logger.warning(f"Unknown action type: {type(action)}")
                return []

    def grasp_object(
        self,
        action: GraspObjectAction,
    ) -> list[EEFWaypoint]:
        """Generate a grasp trajectory for a specific object.

        This implementation matches the reference RobotController.grasp() behavior:
        1. Calculate pre-grasp offset based on object AABB and approach direction
        2. Open gripper to appropriate width based on object size
        3. Move to pre-grasp position
        4. Move to grasp position (object center)
        (Note: gripper closing is a separate CloseGripperAction)

        Args:
            action: The grasp action with parameters (grasping_direction, offs, gripper)

        Returns:
            List of EEF waypoints for the grasp motion
        """
        PRE_GRASP_OFFSET_M = 0.1  # Matches reference

        max_gripper = self.config.gripper_width_range[1]

        obj_pos = self._get_object_position(action.object_name)
        if obj_pos is None:
            logger.error(f"Object '{action.object_name}' not found in objs_info")
            return []

        # Get approach direction from grasping_direction string
        approach_direction = direction_to_approach_vector(action.grasping_direction)
        base_grasp_rotation = direction_to_quaternion(action.grasping_direction)

        # Apply dlt_euler to the grasp rotation
        if action.dlt_euler != (0.0, 0.0, 0.0):
            base_quat = base_grasp_rotation.to_numpy()
            base_euler = npm.quat_to_euler(base_quat)
            dlt_euler_rad = np.deg2rad(np.array(action.dlt_euler, dtype=np.float64))
            target_euler = base_euler + dlt_euler_rad
            grasp_rotation = UnitQuaternion.from_numpy(npm.euler_to_quat(target_euler))
        else:
            grasp_rotation = base_grasp_rotation

        # Calculate offset based on object bounds (like reference implementation)
        bounds = self._get_object_bounds(action.object_name)
        if bounds is not None:
            aabb_min, aabb_max = bounds
            # max_dimension along approach direction
            max_dimension = np.abs(np.dot(approach_direction, aabb_max - aabb_min))
            offset_distance = max_dimension * 0.7 + PRE_GRASP_OFFSET_M
            # Gripper open width based on object size
            gripper_open_width = max(max_dimension * 0.7, max_gripper)
        else:
            # Fallback to fixed offset if no bounds available
            offset_distance = PRE_GRASP_OFFSET_M
            gripper_open_width = max_gripper

        # Apply offs offset
        offs = np.array(action.offs, dtype=np.float64)

        # Calculate pre-grasp and grasp positions (like reference)
        # Pre-grasp: offset from object center opposite to approach direction
        pre_grasp_pos = obj_pos - approach_direction * offset_distance + offs
        # Grasp: at the object center
        grasp_pos = obj_pos + offs

        # print(f"pre_grasp_pos: {pre_grasp_pos}")
        # print(f"grasp_pos: {grasp_pos}")

        logger.debug(f"Grasp {action.object_name}: obj_pos={obj_pos}, pre_grasp={pre_grasp_pos}, grasp={grasp_pos}")
        logger.debug(f"  offset_distance={offset_distance:.3f}, gripper_open={gripper_open_width:.3f}")

        trajectory: list[EEFWaypoint] = []

        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left

        # Determine gripper states based on action.gripper string
        # "" = both open, "l" = left closed, "r" = right closed, "lr" = both closed
        left_gripper_target = 0.0 if "l" in action.gripper else max_gripper
        right_gripper_target = 0.0 if "r" in action.gripper else max_gripper

        # Step 1: Move to pre-grasp position while opening gripper
        target_pose = self._make_pose(pre_grasp_pos, grasp_rotation)
        for i in range(1, action.pre_grasp_steps + 1):
            t = i / action.pre_grasp_steps
            interp_pose = self._interpolate_pose(current_pose, target_pose, t)  # type: ignore
            interp_left_gripper = left_gripper_target
            # self._gripper_left * (1 - t) + (gripper_open_width if action.arm == Arm.LEFT else left_gripper_target) * t
            interp_right_gripper = right_gripper_target
            # self._gripper_right * (1 - t) + (gripper_open_width if action.arm == Arm.RIGHT else right_gripper_target) * t
            if action.arm == Arm.LEFT:
                trajectory.append((interp_pose, other_pose, interp_left_gripper, interp_right_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, interp_pose, interp_left_gripper, interp_right_gripper))  # type: ignore
        current_pose = target_pose
        if action.arm == Arm.LEFT:
            current_gripper = left_gripper_target  # gripper_open_width
            other_gripper = right_gripper_target
        else:
            other_gripper = left_gripper_target
            current_gripper = right_gripper_target  # gripper_open_width

        # Hold at pre-grasp
        for _ in range(5):
            if action.arm == Arm.LEFT:
                trajectory.append((current_pose, other_pose, current_gripper, other_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, current_pose, other_gripper, current_gripper))  # type: ignore

        # Step 2: Move to grasp position (approach the object)
        target_pose = self._make_pose(grasp_pos, grasp_rotation)
        for i in range(1, action.grasp_steps + 1):
            t = i / action.grasp_steps
            interp_pose = self._interpolate_pose(current_pose, target_pose, t)
            if action.arm == Arm.LEFT:
                trajectory.append((interp_pose, other_pose, current_gripper, other_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, interp_pose, other_gripper, current_gripper))  # type: ignore
        current_pose = target_pose

        # Hold at grasp position
        for _ in range(5):
            if action.arm == Arm.LEFT:
                trajectory.append((current_pose, other_pose, current_gripper, other_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, current_pose, other_gripper, current_gripper))  # type: ignore

        # print(f"trajectory: {trajectory}")
        # print(f"trajectory length: {len(trajectory)}")
        # (
        #   Pose6D [UnitQuaternion, RobotBaseRefFrame]
        #   (
        #    vector=Vector3D(x=0.592031779570301, y=0.319316608346254, z=-0.21464277503895784),
        #    rotation=UnitQuaternion(w=0.4472292617926033, x=0.4749329239784279, y=0.5615924970032647, z=0.5089583209137878),
        #    ref_frame=RobotBaseRefFrame(has_origin=True, is_robot_base=True)
        #   ),
        #   Pose6D [UnitQuaternion, RobotBaseRefFrame]
        #   (
        #    vector=Vector3D(x=0.4545600846701258, y=-0.16818742727178632, z=0.8382312256283521),
        #    rotation=UnitQuaternion(w=0.5000000000000001, x=0.5, y=0.4999999999999999, z=0.5),
        #    ref_frame=RobotBaseRefFrame(has_origin=True, is_robot_base=True)
        #   ),
        #   0.08,
        #   0.08
        # )

        # assert False

        return trajectory

    def moveto(self, action: MoveToAction) -> list[EEFWaypoint]:
        """Move arm by delta position and/or delta euler angles.

        Matches the reference RobotController.moveto() interface:
        moveto(arm, dlt_pos, dlt_euler, steps, gripper)
        """
        max_gripper = self.config.gripper_width_range[1]
        min_gripper = self.config.gripper_width_range[0]

        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left

        # Get current position and orientation
        current_pos = current_pose.vector.to_numpy()  # type: ignore
        current_quat = current_pose.rotation.to_numpy()  # type: ignore
        current_euler = npm.quat_to_euler(current_quat)

        # Apply deltas
        dlt_pos = np.array(action.dlt_pos, dtype=np.float64)
        dlt_euler_rad = np.deg2rad(np.array(action.dlt_euler, dtype=np.float64))

        target_pos = current_pos + dlt_pos
        target_euler = current_euler + dlt_euler_rad
        target_quat = npm.euler_to_quat(target_euler)
        target_rotation = UnitQuaternion.from_numpy(target_quat)
        target_pose = self._make_pose(target_pos, target_rotation)

        # Determine gripper states based on action.gripper string
        left_gripper_target = min_gripper if "l" in action.gripper else max_gripper
        right_gripper_target = min_gripper if "r" in action.gripper else max_gripper

        trajectory: list[EEFWaypoint] = []
        for i in range(1, action.steps + 1):
            t = i / action.steps
            interp_pose = self._interpolate_pose(current_pose, target_pose, t)  # type: ignore
            interp_left = left_gripper_target
            # self._gripper_left * (1 - t) + left_gripper_target * t
            interp_right = right_gripper_target
            # self._gripper_right * (1 - t) + right_gripper_target * t
            if action.arm == Arm.LEFT:
                trajectory.append((interp_pose, other_pose, interp_left, interp_right))  # type: ignore
            else:
                trajectory.append((other_pose, interp_pose, interp_left, interp_right))  # type: ignore

        return trajectory

    def moveto_object(self, action: MoveToObjectAction) -> list[EEFWaypoint]:
        """Move arm to an object position with offset.

        Matches the reference RobotController.moveto_object() interface:
        moveto_object(arm, object_name, dlt_pos, dlt_euler, steps, gripper)
        """
        max_gripper = self.config.gripper_width_range[1]
        min_gripper = self.config.gripper_width_range[0]

        obj_pos = self._get_object_position(action.object_name)
        if obj_pos is None:
            logger.error(f"Object '{action.object_name}' not found in objs_info")
            return []

        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left

        # Get current gripper orientation
        current_quat = current_pose.rotation.to_numpy()  # type: ignore
        current_euler_rad = npm.quat_to_euler(current_quat)

        # Apply deltas to gripper euler (not object euler)
        dlt_pos = np.array(action.dlt_pos, dtype=np.float64)
        dlt_euler_rad = np.deg2rad(np.array(action.dlt_euler, dtype=np.float64))

        target_pos = obj_pos + dlt_pos
        target_euler = current_euler_rad + dlt_euler_rad
        target_quat = npm.euler_to_quat(target_euler)
        target_rotation = UnitQuaternion.from_numpy(target_quat)
        target_pose = self._make_pose(target_pos, target_rotation)

        # Determine gripper states based on action.gripper string
        left_gripper_target = min_gripper if "l" in action.gripper else max_gripper
        right_gripper_target = min_gripper if "r" in action.gripper else max_gripper

        trajectory: list[EEFWaypoint] = []
        for i in range(1, action.steps + 1):
            t = i / action.steps
            interp_pose = self._interpolate_pose(current_pose, target_pose, t)  # type: ignore
            interp_left = left_gripper_target
            # self._gripper_left * (1 - t) + left_gripper_target * t
            interp_right = right_gripper_target
            # self._gripper_right * (1 - t) + right_gripper_target * t
            if action.arm == Arm.LEFT:
                trajectory.append((interp_pose, other_pose, interp_left, interp_right))  # type: ignore
            else:
                trajectory.append((other_pose, interp_pose, interp_left, interp_right))  # type: ignore

        return trajectory

    def move_to_position(self, action: MoveToPositionAction) -> list[EEFWaypoint]:
        """Generate trajectory to move arm to target position."""
        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left

        target_rotation = action.target_rotation
        if target_rotation is None:
            target_rotation = current_pose.rotation  # type: ignore

        # Convert target position from world frame to robot base frame if needed
        target_position = action.target_position
        if action.is_world_frame:
            target_position = self._world_to_robot_base_pos(target_position)

        target_pose = self._make_pose(target_position, target_rotation)

        left_gripper_target = 0 if "l" in action.gripper else self.config.gripper_width_range[1]
        right_gripper_target = 0 if "r" in action.gripper else self.config.gripper_width_range[1]

        trajectory: list[EEFWaypoint] = []
        for i in range(1, action.steps + 1):
            t = i / action.steps
            interp_pose = self._interpolate_pose(current_pose, target_pose, t)  # type: ignore
            if action.arm == Arm.LEFT:
                trajectory.append((interp_pose, other_pose, left_gripper_target, right_gripper_target))  # type: ignore
            else:
                trajectory.append((other_pose, interp_pose, left_gripper_target, right_gripper_target))  # type: ignore

        return trajectory

    def open_gripper(self, action: OpenGripperAction) -> list[EEFWaypoint]:
        """Generate trajectory to open gripper."""
        max_gripper = self.config.gripper_width_range[1]

        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
            current_gripper = self._gripper_left
            other_gripper = self._gripper_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left
            current_gripper = self._gripper_right
            other_gripper = self._gripper_left

        trajectory: list[EEFWaypoint] = []
        for i in range(1, action.steps + 1):
            t = i / action.steps
            interp_gripper = current_gripper * (1 - t) + max_gripper * t
            if action.arm == Arm.LEFT:
                trajectory.append((current_pose, other_pose, interp_gripper, other_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, current_pose, other_gripper, interp_gripper))  # type: ignore

        return trajectory

    def close_gripper(self, action: CloseGripperAction) -> list[EEFWaypoint]:
        """Generate trajectory to close gripper."""
        min_gripper = self.config.gripper_width_range[0]

        if action.arm == Arm.LEFT:
            current_pose = self._pose_left
            other_pose = self._pose_right
            current_gripper = self._gripper_left
            other_gripper = self._gripper_right
        else:
            current_pose = self._pose_right
            other_pose = self._pose_left
            current_gripper = self._gripper_right
            other_gripper = self._gripper_left

        trajectory: list[EEFWaypoint] = []
        for i in range(1, action.steps + 1):
            t = i / action.steps
            interp_gripper = current_gripper * (1 - t) + min_gripper * t
            if action.arm == Arm.LEFT:
                trajectory.append((current_pose, other_pose, interp_gripper, other_gripper))  # type: ignore
            else:
                trajectory.append((other_pose, current_pose, other_gripper, interp_gripper))  # type: ignore

        return trajectory

    def hold(self, action: HoldAction) -> list[EEFWaypoint]:
        """Generate trajectory to hold current position."""
        left_gripper_target = 0 if "l" in action.gripper else self.config.gripper_width_range[1]
        right_gripper_target = 0 if "r" in action.gripper else self.config.gripper_width_range[1]
        return [
            (
                self._pose_left,
                self._pose_right,
                left_gripper_target,
                right_gripper_target,
            )
            for _ in range(action.steps)
        ]

    def return_to_home(self, action: ReturnToHomeAction) -> list[EEFWaypoint]:
        """Generate trajectory to return to home position."""
        if self._initial_pose_left is None or self._initial_pose_right is None:
            logger.warning("Initial poses not set, cannot return to home")
            return []

        if action.arm == Arm.LEFT:
            return self.move_to_position(
                MoveToPositionAction(
                    arm=Arm.LEFT,
                    target_position=self._initial_pose_left.vector.to_numpy(),
                    target_rotation=self._initial_pose_left.rotation,
                    steps=action.steps,
                )
            )
        else:
            return self.move_to_position(
                MoveToPositionAction(
                    arm=Arm.RIGHT,
                    target_position=self._initial_pose_right.vector.to_numpy(),
                    target_rotation=self._initial_pose_right.rotation,
                    steps=action.steps,
                )
            )


# =============================================================================
# VLM Planner (GPT-based planning)
# =============================================================================

VLM_SYSTEM_PROMPT = """You are a robot manipulation planner controlling a bimanual robot. You will receive observations from the robot's cameras and must decide actions to complete tasks.

TASK: {task_name}

WORKSPACE BOUNDS (world frame):
The robot can reliably reach x: [0.30, 0.72], y: [-0.45, 0.45] on the table surface.

AVAILABLE ACTIONS:
1. MOVETO_GRASP(arm, object_name, grasping_direction, offs, dlt_euler, gripper) - Move end-effector to object for grasping
   - arm: "left" or "right"
   - object_name: name of object to grasp
   - grasping_direction: gripper orientation string in format "{{gripper_faces}}_{{fingers_align}}_{{fingers_point}}"
     Common options:
     - "down_left-right_front" (grasp from top, fingers left-right)
     - "down_front-back_left" (grasp from top, fingers front-back)
     - "front_left-right_up" (grasp from front, fingers horizontal)
     - "back_left-right_up" (grasp from back, fingers horizontal)
     - "left_front-back_up" (grasp from left side)
     - "right_front-back_up" (grasp from right side)
     Full list: front_left-right_up, front_left-right_down, front_up-down_left, front_up-down_right,
                left_front-back_up, left_front-back_down, left_up-down_back, left_up-down_front,
                back_left-right_up, back_left-right_down, back_up-down_left, back_up-down_right,
                right_front-back_up, right_front-back_down, right_up-down_back, right_up-down_front,
                down_front-back_left, down_front-back_right, down_left-right_back, down_left-right_front,
                up_front-back_left, up_front-back_right, up_left-right_back, up_left-right_front
   - offs: [x, y, z] position offset from object center (default [0, 0, 0])
   - dlt_euler: [rx, ry, rz] additional rotation in degrees applied to grasp orientation (default [0, 0, 0])
   - gripper: which grippers to keep closed during motion ("" = both open, "l" = left closed, "r" = right closed, "lr" = both closed)

2. MOVETO_OBJECT(arm, object_name, dlt_pos, dlt_euler, steps, gripper) - Move arm to object position with offset
   - arm: "left" or "right"
   - object_name: name of object to move to
   - dlt_pos: [dx, dy, dz] position offset from object center (default [0, 0, 0])
   - dlt_euler: [rx, ry, rz] delta euler angles in degrees (default [0, 0, 0])
   - steps: number of interpolation steps (default 50)
   - gripper: which grippers to keep closed ("" = both open, "l" = left closed, "r" = right closed, "lr" = both closed)

3. OPEN_GRIPPER(arm) - Open the gripper
   - arm: "left" or "right"

4. CLOSE_GRIPPER(arm) - Close the gripper
   - arm: "left" or "right"

5. HOLD(steps, gripper) - Hold current position
   - steps: number of steps to hold (default 10)
   - gripper: which grippers to keep closed ("" = both open, "l" = left closed, "r" = right closed, "lr" = both closed)

6. RETURN_HOME(arm) - Return arm to initial/home position
   - arm: "left" or "right"

7. DONE() - Call this when the task appears to be completed
   - Use this when you believe the task goal has been achieved

COORDINATE SYSTEM:
- X: forward (positive) / backward (negative)
- Y: left (positive) / right (negative)
- Z: up (positive) / down (negative)
- Robot workspace is roughly: {{"x_min": 0.21, "x_max": 1.00, "y_min": -0.69, "y_max": 0.69, "z_min": 0.76, "z_max": 1.2}}

OUTPUT FORMAT:
Return a single JSON object for the next action. Examples:
```json
{{"action": "MOVETO_GRASP", "arm": "right", "object_name": "cube", "grasping_direction": "down_left-right_front", "offs": [0, 0, 0], "dlt_euler": [0, 0, 0], "gripper": ""}}
```
```json
{{"action": "CLOSE_GRIPPER", "arm": "right"}}
```
```json
{{"action": "MOVETO_OBJECT", "arm": "right", "object_name": "target", "dlt_pos": [0, 0, 0.1], "dlt_euler": [0, 0, 0], "steps": 50, "gripper": "r"}}
```
```json
{{"action": "RETURN_HOME", "arm": "right"}}
```
```json
{{"action": "DONE"}}
```

GUIDELINES:
- **VERIFY ACTION SUCCESS**: After each action, compare object positions with the previous observation. If an object's position didn't change, the action likely failed - try a different approach.
- **CHECK REACHABILITY**: Before attempting to manipulate an object, verify it's within workspace bounds (x: [0.30, 0.72], y: [-0.45, 0.45]). If outside bounds, the task may be impossible.
- **VERIFY GRIPPER PROXIMITY**: Before declaring success, check that your gripper actually reached near the target object (within ~0.05m).
- **PUSHING STRATEGY**: To push an object toward a target:
  1. Position the gripper BEHIND the object (opposite side from target direction)
  2. Lower to object height
  3. Push THROUGH the object toward the target
- **DON'T REPEAT FAILURES**: If the same action fails twice, try a fundamentally different approach (different arm, different direction, different offset).
- Analyze images carefully to understand the scene geometry
- Pay attention to object relationships and constraints"""

VLM_OBSERVATION_TEMPLATE = """OBSERVATION {observation_num}:

ROBOT STATE (EEF poses in world frame):
{robot_eef_pose}

OBJECTS (with pose in world frame):
{objects_info}

[Images attached: ego camera, left wrist, right wrist]

First, analyze: Did your previous action succeed? Check if object positions changed. Then decide your next action."""


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _to_numpy(value: Any) -> npt.NDArray[np.float64] | None:
    """Convert value to numpy array."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().astype(np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, list | tuple):
        return np.array(value, dtype=np.float64)
    return None


@dataclass
class RobotEEFPoseInfo:
    """Robot end-effector pose information."""

    left_pos: npt.NDArray[np.float64] | None = None  # (3,) position in world frame
    left_euler: npt.NDArray[np.float64] | None = None  # (3,) euler angles in degrees
    left_gripper: float | None = None  # gripper width
    right_pos: npt.NDArray[np.float64] | None = None  # (3,) position in world frame
    right_euler: npt.NDArray[np.float64] | None = None  # (3,) euler angles in degrees
    right_gripper: float | None = None  # gripper width


def format_robot_eef_pose(eef_info: RobotEEFPoseInfo | None) -> str:
    """Format robot EEF pose info for the VLM prompt.

    Args:
        eef_info: Robot EEF pose information

    Returns:
        Formatted string describing the robot's current EEF poses
    """
    if eef_info is None:
        return "(Robot state not available)"

    lines = []

    # Left gripper
    if eef_info.left_pos is not None:
        left_line = "- Left gripper:\n"
        left_line += f"    pos: ({eef_info.left_pos[0]:.3f}, {eef_info.left_pos[1]:.3f}, {eef_info.left_pos[2]:.3f})\n"
        if eef_info.left_euler is not None:
            left_line += f"    euler: ({eef_info.left_euler[0]:.1f}°, {eef_info.left_euler[1]:.1f}°, {eef_info.left_euler[2]:.1f}°)\n"
        if eef_info.left_gripper is not None:
            gripper_status = (
                "open"
                if eef_info.left_gripper > 0.04
                else "closed"
                if eef_info.left_gripper < 0.01
                else "partially open"
            )
            left_line += f"    gripper: {gripper_status} (width: {eef_info.left_gripper:.3f}m)"
        lines.append(left_line.rstrip())
    else:
        lines.append("- Left gripper: (not available)")

    # Right gripper
    if eef_info.right_pos is not None:
        right_line = "- Right gripper:\n"
        right_line += (
            f"    pos: ({eef_info.right_pos[0]:.3f}, {eef_info.right_pos[1]:.3f}, {eef_info.right_pos[2]:.3f})\n"
        )
        if eef_info.right_euler is not None:
            right_line += f"    euler: ({eef_info.right_euler[0]:.1f}°, {eef_info.right_euler[1]:.1f}°, {eef_info.right_euler[2]:.1f}°)\n"
        if eef_info.right_gripper is not None:
            gripper_status = (
                "open"
                if eef_info.right_gripper > 0.04
                else "closed"
                if eef_info.right_gripper < 0.01
                else "partially open"
            )
            right_line += f"    gripper: {gripper_status} (width: {eef_info.right_gripper:.3f}m)"
        lines.append(right_line.rstrip())
    else:
        lines.append("- Right gripper: (not available)")

    return "\n".join(lines)


def format_objects_info(objs_info: dict[str, dict[str, Any]]) -> str:
    """Format objects info for the VLM prompt.

    Args:
        objs_info: Dictionary of object info. Format:
            {
                "object_name": {
                    "material": "rigid" | "particle",
                    "pos": np.array,  # (3,) for rigid, (N, 3) for particle
                    "euler": np.array or None,  # (3,) for rigid, None for particle
                    "vel": np.array,
                    "bounds": np.array or None,  # (2, 3) AABB for rigid
                    "convex_hull_2d": np.array or None,  # (N, 2) for rigid
                }
            }

    Returns:
        Formatted string describing all objects
    """
    lines = []
    for obj_name, obj_info in objs_info.items():
        material = obj_info.get("material", "rigid")
        pos = _to_numpy(obj_info.get("pos"))
        euler = _to_numpy(obj_info.get("euler"))
        bounds = _to_numpy(obj_info.get("bounds"))

        if pos is None:
            lines.append(f"- {obj_name}: [unknown position]")
            continue

        if material == "particle":
            # Particle object: pos is (N, 3), compute center and bbox
            if pos.ndim == 2 and pos.shape[1] == 3:
                center = pos.mean(axis=0)
                bbox_min = pos.min(axis=0)
                bbox_max = pos.max(axis=0)

                lines.append(
                    f"- {obj_name} (particle):\n"
                    f"    center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n"
                    f"    bbox_min: ({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f})\n"
                    f"    bbox_max: ({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})"
                )
            else:
                lines.append(f"- {obj_name} (particle): [invalid pos shape]")
        else:
            # Rigid object: pos is (3,)
            if pos.ndim == 1 and pos.shape[0] == 3:
                obj_line = f"- {obj_name} (rigid):\n"
                obj_line += f"    pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"

                if euler is not None and euler.shape == (3,):
                    obj_line += f"    euler: ({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)\n"

                if bounds is not None and bounds.shape == (2, 3):
                    obj_line += (
                        f"    bounds: min({bounds[0, 0]:.3f}, {bounds[0, 1]:.3f}, {bounds[0, 2]:.3f}) "
                        f"max({bounds[1, 0]:.3f}, {bounds[1, 1]:.3f}, {bounds[1, 2]:.3f})"
                    )

                lines.append(obj_line.rstrip())
            else:
                lines.append(f"- {obj_name} (rigid): [invalid pos shape]")

    return "\n".join(lines) if lines else "No objects detected"


@dataclass
class VLMCallRecord:
    """Record of a single VLM call with prompt and response."""

    action_index: int
    prompt: str
    response: str
    parsed_action: dict[str, Any] | None
    images: dict[str, Image.Image] | None = None  # {"ego": img, "wrist_left": img, "wrist_right": img}


def encode_images_for_api(images: dict[str, Image.Image]) -> list[dict[str, Any]]:
    """Encode multiple images as base64 for the OpenAI API.

    Args:
        images: Dictionary of camera name to PIL Image

    Returns:
        List of content dicts for the API (text labels + images)
    """
    image_contents: list[dict[str, Any]] = []
    camera_labels = {
        "ego": "Camera: ego (front view)",
        "wrist_left": "Camera: wrist_left",
        "wrist_right": "Camera: wrist_right",
    }

    for cam_name in ["ego", "wrist_left", "wrist_right"]:
        if cam_name not in images:
            continue
        img = images[cam_name]
        image_base64 = encode_image_to_base64(img)

        # Add label text
        image_contents.append(
            {
                "type": "text",
                "text": camera_labels.get(cam_name, f"Camera: {cam_name}"),
            }
        )
        # Add image
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high",
                },
            }
        )

    return image_contents


def call_vlm_with_conversation(
    system_prompt: str,
    conversation_history: list[dict[str, Any]],
    new_observation: str,
    images: dict[str, Image.Image],
    action_index: int,
    model: str = "gpt-4o",
) -> tuple[dict[str, Any] | None, str, VLMCallRecord]:
    """Call VLM with conversation history (agent-style).

    Args:
        system_prompt: System prompt with task and action definitions
        conversation_history: List of previous messages (user observations + assistant responses)
        new_observation: New observation text to add as user message
        images: Dictionary of camera images {"ego": img, "wrist_left": img, "wrist_right": img}
        action_index: Current action index for logging
        model: OpenAI model to use

    Returns:
        Tuple of (action_dict, response_text, VLMCallRecord)
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI package not installed. Using fallback planner.")
        record = VLMCallRecord(
            action_index=action_index,
            prompt=new_observation,
            response="[OpenAI not installed - using fallback]",
            parsed_action=None,
            images=images,
        )
        return None, "", record

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Using fallback planner.")

        fallback_action = {
            "action": "GRASP",
            "arm": "right",
            "object_name": "heavy_large_object",
            "offs": [0, -0.1, 0],
            "grasping_direction": "front_left-right_up",
        }

        record = VLMCallRecord(
            action_index=action_index,
            prompt=new_observation,
            response="[No API key - using fallback]",
            parsed_action=fallback_action,
            images=images,
        )
        return fallback_action, "", record

    client = OpenAI(api_key=api_key)

    # Wrap with LangSmith tracing if available
    try:
        from langsmith import wrappers as langsmith_wrappers

        client = langsmith_wrappers.wrap_openai(client)
        logger.debug("LangSmith tracing enabled for OpenAI client")
    except ImportError:
        logger.debug("LangSmith not installed, skipping tracing")

    # Encode images for API
    image_contents = encode_images_for_api(images)

    # Build the new user message with observation text and images
    new_user_content: list[dict[str, Any]] = [{"type": "text", "text": new_observation}]
    new_user_content.extend(image_contents)

    # Build full message list: system + history + new observation
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": new_user_content})

    logger.info(f"Calling VLM ({model}) - conversation turn {len(conversation_history) // 2 + 1}...")
    logger.debug(f"=== NEW OBSERVATION ===\n{new_observation}\n=== END OBSERVATION ===")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content or ""
        logger.info(f"=== VLM RESPONSE ===\n{response_text}\n=== END RESPONSE ===")

        # Parse JSON from response - look for a single JSON object
        json_match = re.search(r"\{[^{}]*\}", response_text)
        if json_match:
            action_json = json.loads(json_match.group())
            record = VLMCallRecord(
                action_index=action_index,
                prompt=new_observation,
                response=response_text,
                parsed_action=action_json,
                images=images,
            )
            return action_json, response_text, record
        else:
            logger.warning("No JSON object found in VLM response")
            record = VLMCallRecord(
                action_index=action_index,
                prompt=new_observation,
                response=response_text,
                parsed_action=None,
                images=images,
            )
            return None, response_text, record

    except Exception as e:
        logger.error(f"VLM call failed: {e}")
        record = VLMCallRecord(
            action_index=action_index,
            prompt=new_observation,
            response=f"[Error: {e}]",
            parsed_action=None,
            images=images,
        )
        return None, "", record


@dataclass
class DoneAction(HighLevelAction):
    """Indicates the VLM believes the task is complete."""

    @property
    def name(self) -> str:
        return "done()"


def parse_vlm_action(
    action_dict: dict[str, Any],
) -> HighLevelAction | None:
    """Parse a single VLM action output into a HighLevelAction object.

    Args:
        action_dict: Single action dictionary from VLM

    Returns:
        HighLevelAction object, or None if parsing failed
    """
    action_type = action_dict.get("action", "").upper()
    arm_str = action_dict.get("arm", "left").lower()
    arm = Arm.LEFT if arm_str == "left" else Arm.RIGHT

    try:
        if action_type in ("GRASP", "MOVETO_GRASP"):
            object_name = action_dict.get("object_name", "")
            grasping_direction = action_dict.get(
                "grasping_direction", action_dict.get("grasp_direction", "down_left-right_front")
            )

            # Validate grasp direction
            if grasping_direction not in DIRECTIONS_TO_ROTATION:
                logger.warning(f"Unknown grasping_direction '{grasping_direction}', using 'down_left-right_front'")
                grasping_direction = "down_left-right_front"

            # Parse offs
            offs_raw = action_dict.get("offs", [0.0, 0.0, 0.0])
            if isinstance(offs_raw, list | tuple) and len(offs_raw) == 3:
                offs = (float(offs_raw[0]), float(offs_raw[1]), float(offs_raw[2]))
            else:
                offs = (0.0, 0.0, 0.0)

            # Parse gripper state
            gripper = str(action_dict.get("gripper", ""))

            # Parse steps
            pre_grasp_steps = int(action_dict.get("pre_grasp_steps", 40))
            grasp_steps = int(action_dict.get("grasp_steps", 30))

            # Parse dlt_euler
            dlt_euler_raw = action_dict.get("dlt_euler", [0.0, 0.0, 0.0])
            if isinstance(dlt_euler_raw, list | tuple) and len(dlt_euler_raw) == 3:
                dlt_euler = (float(dlt_euler_raw[0]), float(dlt_euler_raw[1]), float(dlt_euler_raw[2]))
            else:
                dlt_euler = (0.0, 0.0, 0.0)

            return GraspObjectAction(
                arm=arm,
                object_name=object_name,
                grasping_direction=grasping_direction,
                pre_grasp_steps=pre_grasp_steps,
                grasp_steps=grasp_steps,
                offs=offs,
                dlt_euler=dlt_euler,
                gripper=gripper,
            )

        elif action_type == "MOVETO_OBJECT":
            object_name = action_dict.get("object_name", "")

            # Parse dlt_pos
            dlt_pos_raw = action_dict.get("dlt_pos", [0.0, 0.0, 0.0])
            if isinstance(dlt_pos_raw, list | tuple) and len(dlt_pos_raw) == 3:
                dlt_pos = (float(dlt_pos_raw[0]), float(dlt_pos_raw[1]), float(dlt_pos_raw[2]))
            else:
                dlt_pos = (0.0, 0.0, 0.0)

            # Parse dlt_euler
            dlt_euler_raw = action_dict.get("dlt_euler", [0.0, 0.0, 0.0])
            if isinstance(dlt_euler_raw, list | tuple) and len(dlt_euler_raw) == 3:
                dlt_euler = (float(dlt_euler_raw[0]), float(dlt_euler_raw[1]), float(dlt_euler_raw[2]))
            else:
                dlt_euler = (0.0, 0.0, 0.0)

            steps = int(action_dict.get("steps", 50))
            gripper = str(action_dict.get("gripper", ""))

            return MoveToObjectAction(
                arm=arm,
                object_name=object_name,
                dlt_pos=dlt_pos,
                dlt_euler=dlt_euler,
                steps=steps,
                gripper=gripper,
            )

        elif action_type in ("MOVE_TO", "MOVE_EEF"):
            # Legacy absolute position movement
            x = float(action_dict.get("x", 0.4))
            y = float(action_dict.get("y", 0.0))
            z = float(action_dict.get("z", 0.9))
            target_pos = np.array([x, y, z], dtype=np.float64)

            # Parse orientation (axis-angle) if provided
            target_rotation: UnitQuaternion | None = None
            if "wx" in action_dict or "wy" in action_dict or "wz" in action_dict:
                wx = float(action_dict.get("wx", 0.0))
                wy = float(action_dict.get("wy", 0.0))
                wz = float(action_dict.get("wz", 0.0))
                axis_angle = np.array([wx, wy, wz], dtype=np.float64)
                target_rotation = axis_angle_to_quaternion(axis_angle)

            gripper = str(action_dict.get("gripper", ""))

            return MoveToPositionAction(
                arm=arm,
                target_position=target_pos,
                target_rotation=target_rotation,
                steps=40,
                gripper=gripper,
                is_world_frame=True,  # VLM provides positions in world frame
            )

        elif action_type == "OPEN_GRIPPER":
            return OpenGripperAction(arm=arm)

        elif action_type == "CLOSE_GRIPPER":
            return CloseGripperAction(arm=arm)

        elif action_type == "HOLD":
            steps = int(action_dict.get("steps", 30))
            gripper = str(action_dict.get("gripper", ""))
            return HoldAction(steps=steps, gripper=gripper)

        elif action_type == "RETURN_HOME":
            return ReturnToHomeAction(arm=arm)

        elif action_type == "DONE":
            return DoneAction()

        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to parse action {action_dict}: {e}")
        return None


# =============================================================================
# High-Level Planner
# =============================================================================


class HighLevelPlanner:
    """Generates high-level actions one at a time from observations using VLM.

    Uses GPT-4V/GPT-5 with ego camera image to generate manipulation actions.
    Each call to plan_next_action() queries the VLM with the current observation
    and returns a single action to execute.
    """

    def __init__(self, task_name: str, vlm_model: str = "gpt-4o", max_vlm_calls: int = 20) -> None:
        self.task_name = task_name
        self.vlm_model = vlm_model
        self.max_vlm_calls = max_vlm_calls
        self._action_history: list[str] = []  # String descriptions of actions taken
        self._action_count: int = 0
        self._current_action: HighLevelAction | None = None
        self._is_done: bool = False
        self._vlm_call_records: list[VLMCallRecord] = []  # Records of all VLM calls

        # Agent-style conversation history
        self._system_prompt = VLM_SYSTEM_PROMPT.format(task_name=task_name)
        self._conversation_history: list[dict[str, Any]] = []  # List of {role, content} messages
        self._observation_count: int = 0

    def reset(self) -> None:
        """Reset the planner for a new episode."""
        self._action_history = []
        self._action_count = 0
        self._current_action = None
        self._is_done = False
        self._vlm_call_records = []
        self._conversation_history = []
        self._observation_count = 0

    @property
    def is_done(self) -> bool:
        """Check if the planner has signaled completion (VLM returned DONE or max calls reached)."""
        return self._is_done

    @property
    def current_action(self) -> HighLevelAction | None:
        """Get the current action to execute."""
        return self._current_action

    @property
    def action_count(self) -> int:
        """Get the number of actions planned so far."""
        return self._action_count

    @property
    def action_history(self) -> list[str]:
        """Get the history of action descriptions."""
        return self._action_history

    @property
    def vlm_call_records(self) -> list[VLMCallRecord]:
        """Get the records of all VLM calls."""
        return self._vlm_call_records

    def plan_next_action(
        self,
        objs_info: dict[str, dict[str, Any]],
        images: dict[str, Image.Image],
        robot_eef_pose: RobotEEFPoseInfo | None = None,
    ) -> HighLevelAction | None:
        """Plan the next single action using VLM based on current observation.

        Uses agent-style conversation - maintains dialogue history across observations.

        Args:
            objs_info: Dictionary of object info from environment
            images: Dictionary of camera images {"ego": img, "wrist_left": img, "wrist_right": img}
            robot_eef_pose: Current robot EEF pose information

        Returns:
            Next high-level action to execute, or None if planning is complete/failed
        """
        if self._is_done:
            return None

        # Check max VLM calls limit
        if len(self._vlm_call_records) >= self.max_vlm_calls:
            logger.info(f"Max VLM calls ({self.max_vlm_calls}) reached, marking as done")
            self._is_done = True
            return None

        self._observation_count += 1

        # Format the new observation as a user message
        observation_text = VLM_OBSERVATION_TEMPLATE.format(
            observation_num=self._observation_count,
            robot_eef_pose=format_robot_eef_pose(robot_eef_pose),
            objects_info=format_objects_info(objs_info),
        )

        logger.info(
            f"Planning action {self._action_count + 1} (observation {self._observation_count}, VLM call {len(self._vlm_call_records) + 1}/{self.max_vlm_calls})..."
        )

        # Call VLM with conversation history
        action_dict, response_text, vlm_record = call_vlm_with_conversation(
            system_prompt=self._system_prompt,
            conversation_history=self._conversation_history,
            new_observation=observation_text,
            images=images,
            action_index=self._action_count + 1,
            model=self.vlm_model,
        )

        # Store the VLM call record
        self._vlm_call_records.append(vlm_record)

        # Add the observation (as text only, images are not stored in history to save tokens)
        # and the response to conversation history for future calls
        self._conversation_history.append(
            {
                "role": "user",
                "content": observation_text,  # Text only - images sent but not stored in history
            }
        )
        if response_text:
            self._conversation_history.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

        if action_dict is None:
            logger.error("VLM returned no action")
            return None

        action = parse_vlm_action(action_dict)
        if action is None:
            logger.error(f"Failed to parse VLM action: {action_dict}")
            return None

        # Check if VLM signaled completion
        if isinstance(action, DoneAction):
            logger.info("VLM signaled task completion (DONE)")
            self._is_done = True
            return None

        self._current_action = action
        self._action_count += 1

        # Record action in history (for logging purposes)
        self._action_history.append(action.name)

        logger.info(f"Planned action {self._action_count}: {action.name}")
        return action

    def save_vlm_records(self, output_dir: Path) -> None:
        """Save all VLM call records to the output directory.

        Args:
            output_dir: Directory to save records to
        """
        vlm_dir = output_dir / "vlm_calls"
        vlm_dir.mkdir(parents=True, exist_ok=True)

        # Save system prompt
        system_prompt_file = vlm_dir / "system_prompt.txt"
        with open(system_prompt_file, "w") as f:
            f.write(self._system_prompt)

        for record in self._vlm_call_records:
            # Save observation (user message)
            prompt_file = vlm_dir / f"action_{record.action_index:03d}_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(record.prompt)

            # Save response (assistant message)
            response_file = vlm_dir / f"action_{record.action_index:03d}_response.txt"
            with open(response_file, "w") as f:
                f.write(record.response)

            # Save parsed action as JSON
            if record.parsed_action is not None:
                action_file = vlm_dir / f"action_{record.action_index:03d}_parsed.json"
                with open(action_file, "w") as f:
                    json.dump(record.parsed_action, f, indent=2)

            # Save all camera images (ego, wrist_left, wrist_right)
            if record.images is not None:
                for cam_name, img in record.images.items():
                    image_file = vlm_dir / f"action_{record.action_index:03d}_{cam_name}.png"
                    img.save(image_file)

        # Save conversation history
        conversation_file = vlm_dir / "conversation_history.json"
        with open(conversation_file, "w") as f:
            json.dump(self._conversation_history, f, indent=2)

        # Save summary JSON
        summary = {
            "total_vlm_calls": len(self._vlm_call_records),
            "total_observations": self._observation_count,
            "calls": [
                {
                    "action_index": r.action_index,
                    "parsed_action": r.parsed_action,
                }
                for r in self._vlm_call_records
            ],
        }
        summary_file = vlm_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved {len(self._vlm_call_records)} VLM call records to {vlm_dir}")


# =============================================================================
# Scripted Controller (Orchestrator)
# =============================================================================


class ScriptedController:
    """Orchestrates high-level planning and low-level control.

    This is the main controller that:
    1. Takes observations from the environment
    2. Plans ONE high-level action at a time using VLM
    3. Updates the low-level controller's state
    4. Executes the current low-level trajectory
    5. When trajectory completes, requests new observation and plans next action
    6. Returns a 14D EE_ABS action for the environment
    """

    def __init__(
        self,
        task_name: str,
        gripper_width_range: tuple[float, float] = (0.0, 0.08),
        vlm_model: str = "gpt-4o",
        max_vlm_calls: int = 20,
    ) -> None:
        self.task_name = task_name

        # Create planner and controller
        self.high_level_planner = HighLevelPlanner(task_name, vlm_model=vlm_model, max_vlm_calls=max_vlm_calls)
        self.low_level_controller = LowLevelController(
            LowLevelControllerConfig(gripper_width_range=gripper_width_range)
        )

        # Track whether we need to plan the next action
        self._needs_new_action: bool = True

    def reset(self) -> None:
        """Reset the controller for a new episode."""
        self.high_level_planner.reset()
        self.low_level_controller.reset()
        self._needs_new_action = True

    @property
    def is_done(self) -> bool:
        """Check if planning and execution is complete."""
        return self.high_level_planner.is_done and self.low_level_controller.trajectory_done

    def _extract_camera_images(self, observation: dict[str, Any]) -> dict[str, Image.Image]:
        """Extract images from all three cameras (ego, wrist_left, wrist_right).

        Args:
            observation: gs-gym observation dict with a "pixels" mapping.

        Returns:
            Dictionary of camera name to PIL Image
        """
        images: dict[str, Image.Image] = {}

        try:
            # gs-gym returns ready-to-use (H, W, 3) uint8 arrays per camera, so
            # there is no sensor-object unwrapping to do.
            pixels = observation.get("pixels", {})
            for cam_name in ["ego", "wrist_left", "wrist_right"]:
                frame = pixels.get(cam_name)
                if frame is None:
                    logger.warning(f"No '{cam_name}' camera found in observation")
                    continue
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                images[cam_name] = Image.fromarray(frame)
        except Exception as e:
            logger.error(f"Failed to extract camera images: {e}")

        return images

    def _extract_robot_eef_pose(
        self,
        pose_left: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        pose_right: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        gripper_left: float,
        gripper_right: float,
    ) -> RobotEEFPoseInfo:
        """Extract robot EEF pose information for VLM prompt.

        Args:
            pose_left: Left arm pose in robot base frame
            pose_right: Right arm pose in robot base frame
            gripper_left: Left gripper width
            gripper_right: Right gripper width

        Returns:
            RobotEEFPoseInfo with poses converted to world frame
        """
        # Robot base position in world frame
        robot_base_pos = self.low_level_controller._robot_base_pos

        # Convert poses from robot base frame to world frame
        # World position = robot_base_pos + pose_position
        def pose_to_world(
            pose: Pose6D[UnitQuaternion, RobotBaseRefFrame],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            vec = pose.vector
            pos_array = np.array([vec.x, vec.y, vec.z], dtype=np.float64)
            world_pos = robot_base_pos + pos_array

            # Convert quaternion to euler angles (degrees)
            quat = pose.rotation
            euler = quat.to_euler_angle()
            euler_deg = np.degrees(np.array([euler.roll, euler.pitch, euler.yaw], dtype=np.float64))

            return world_pos, euler_deg

        left_pos, left_euler = pose_to_world(pose_left)
        right_pos, right_euler = pose_to_world(pose_right)

        return RobotEEFPoseInfo(
            left_pos=left_pos,
            left_euler=left_euler,
            left_gripper=gripper_left,
            right_pos=right_pos,
            right_euler=right_euler,
            right_gripper=gripper_right,
        )

    def get_action(
        self,
        observation: dict[str, Any],
        objs_info: dict[str, dict[str, Any]],
    ) -> npt.NDArray[np.float64]:
        """Get the next action from the controller.

        Args:
            observation: gs-gym observation dict from RoboWitsEnv.get_observations().
            objs_info: Object info from the environment (world frame).

        Returns:
            A 14D EE_ABS action for RoboWitsEnv.apply_action.
        """
        # gs-gym packs EE state as agent_pos_ee:
        #   [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
        # already relative to the robot base, which is the frame this controller
        # works in, so the poses transfer without conversion.
        state = np.asarray(observation["agent_pos_ee"], dtype=np.float64)

        pose_right = Pose6D(
            vector=Vector3D.from_numpy(state[0:3]),
            rotation=axis_angle_to_quaternion(state[6:9]),
            ref_frame=RobotBaseRefFrame(),
        )
        pose_left = Pose6D(
            vector=Vector3D.from_numpy(state[3:6]),
            rotation=axis_angle_to_quaternion(state[9:12]),
            ref_frame=RobotBaseRefFrame(),
        )
        gripper_right = float(state[12])
        gripper_left = float(state[13])

        # Update low-level controller state
        self.low_level_controller.update_state(
            pose_left=pose_left,
            pose_right=pose_right,
            gripper_left=gripper_left,
            gripper_right=gripper_right,
            objs_info=objs_info,
        )

        # If current low-level trajectory is done, plan next high-level action
        if self.low_level_controller.trajectory_done and self._needs_new_action and not self.high_level_planner.is_done:
            # Extract all camera images for VLM planning
            images = self._extract_camera_images(observation)
            if not images:
                logger.error("Failed to extract any camera images for planning")
            else:
                # Extract robot EEF pose info for VLM planning
                robot_eef_pose = self._extract_robot_eef_pose(pose_left, pose_right, gripper_left, gripper_right)

                # Plan the next single action
                next_action = self.high_level_planner.plan_next_action(objs_info, images, robot_eef_pose)

                if next_action is not None:
                    logger.info(f"Executing: {next_action.name}")
                    trajectory = self.low_level_controller.execute_action(next_action)
                    self.low_level_controller.set_trajectory(trajectory)
                    self._needs_new_action = False
                else:
                    # Planner is done or failed - don't request new action anymore
                    self._needs_new_action = False

        # Check if we just finished a trajectory and need to request new action
        if (
            self.low_level_controller.trajectory_done
            and not self._needs_new_action
            and not self.high_level_planner.is_done
        ):
            self._needs_new_action = True

        # Get next waypoint from low-level controller
        waypoint = self.low_level_controller.get_next_waypoint()
        left_pose, right_pose, left_gripper, right_gripper = waypoint

        # Emit the waypoint as a gs-gym EE_ABS action.
        action = np.zeros(EE_ABS_ACTION_DIM, dtype=np.float64)
        action[0:3] = right_pose.vector.to_numpy()
        action[3:6] = left_pose.vector.to_numpy()
        action[6:9] = right_pose.rotation.to_axis_angle()
        action[9:12] = left_pose.rotation.to_axis_angle()
        action[12] = right_gripper
        action[13] = left_gripper
        return action


# ==============================================================================
# Per-task evaluation loop
# ==============================================================================


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy values so the state dumps are JSON-writable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_task_eval(
    task_name: str,
    output_dir: Path,
    model: str = "gpt-4o",
    num_episodes: int = 1,
    max_episode_steps: int = 1000,
    max_vlm_calls: int = 20,
    eval_dataset_json: str | None = None,
    show_viewer: bool = False,
    overwrite: bool = False,
    genesis_backend: str = "cpu",
) -> dict[str, Any]:
    """Run the VLM controller baseline on one gs-gym task.

    Args:
        task_name: Registered gs-gym task id (e.g. "robowits/02-retrieve-cube-v0").
        output_dir: Directory to write per-episode artifacts and metrics into.
        model: VLM model name.
        num_episodes: Number of episodes to evaluate.
        max_episode_steps: Maximum env steps per episode.
        max_vlm_calls: Maximum VLM calls (high-level actions) per episode.
        eval_dataset_json: Eval dataset JSON with per-episode initial placements,
            as used by the policy evals.
        show_viewer: Whether to show the Genesis viewer.
        overwrite: Re-evaluate episodes that already have results.
        genesis_backend: Genesis physics backend ("cpu" or "cuda"). The
            SPH/MPM tasks are impractically slow on cpu.

    Returns:
        The metrics dict (also written to eval_metrics/metrics.json).
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown gs-gym task: '{task_name}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running VLM controller eval for task: {task_name}")
    logger.info(f"VLM model: {model}, max VLM calls: {max_vlm_calls}")

    if overwrite:
        logger.info("Overwrite mode enabled, will re-evaluate all episodes")
        for episode_idx in range(num_episodes):
            episode_dir = output_dir / f"episode_{episode_idx:03d}"
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
        metrics_dir = output_dir / "eval_metrics"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)

    episode_results: list[dict[str, Any]] = []
    episode_object_states: list[dict[str, Any]] = []

    # Episodes with a per-episode summary are complete. Their results are loaded
    # back rather than merely skipped: dropping them would compute the success
    # rate over only the newly run episodes and silently understate n.
    completed_episodes: set[int] = set()
    if not overwrite:
        for episode_idx in range(num_episodes):
            summary_file = output_dir / f"episode_{episode_idx:03d}" / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        episode_results.append(json.load(f))
                    completed_episodes.add(episode_idx)
                except Exception as e:
                    logger.warning(f"Failed to load summary for episode {episode_idx}: {e}")
        if completed_episodes:
            logger.info(f"Found {len(completed_episodes)} completed episodes, continuing from remaining...")

    # Nothing left to do: return the existing results without building a scene.
    # Each Genesis scene build is expensive and, across a long sequential sweep,
    # eventually segfaults the process -- so a resume must not rebuild scenes for
    # variants it is only going to skip.
    if len(completed_episodes) >= num_episodes:
        logger.info(f"All {num_episodes} episodes already complete for {task_name}; skipping env build")
        episode_results.sort(key=lambda r: r["episode"])
        max_rewards = [float(r.get("max_reward", 0.0)) for r in episode_results]
        num_success = sum(1 for r in episode_results if r["success"])
        num_crashed = sum(1 for r in episode_results if r.get("crashed", False))
        num_valid = len(episode_results) - num_crashed
        metrics = {
            "task_id": task_name,
            "task_name": task_name,
            "model": model,
            "num_episodes": num_episodes,
            "num_success": num_success,
            "num_crashed": num_crashed,
            "success_rate": num_success / num_valid if num_valid > 0 else 0.0,
            "avg_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
            "episode_results": episode_results,
        }
        metrics_path = output_dir / "eval_metrics" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        write_eval_info(output_dir, task_name, [r["success"] for r in episode_results], max_rewards)
        return metrics

    env = TASK_REGISTRY[task_name](
        show_viewer=show_viewer,
        control_mode="EE_ABS",
        observation_mode="EE",
        genesis_backend=genesis_backend,
        max_episode_steps=max_episode_steps,
        eval_dataset_json=eval_dataset_json,
    )
    task_instruction = env.task_description
    logger.info(f"Task Instruction: {task_instruction}")

    controller = ScriptedController(
        task_name=task_instruction,
        gripper_width_range=(0.0, 0.08),
        vlm_model=model,
        max_vlm_calls=max_vlm_calls,
    )

    for episode_idx in range(num_episodes):
        if episode_idx in completed_episodes:
            logger.info(f"Skipping episode {episode_idx + 1}/{num_episodes} (already completed)")
            continue

        logger.info(f"Starting episode {episode_idx + 1}/{num_episodes}")

        env.set_eval_dataset_idx(episode_idx)
        env.reset_env()
        controller.reset()

        objs_info = env.collect_objs_info()
        episode_success = False
        episode_length = 0
        episode_crashed = False
        # Progress score: the max milestone reward reached during the episode,
        # matching how lerobot-eval reports `max_reward` for policy evals.
        episode_max_reward = 0.0

        try:
            for step in range(max_episode_steps):
                observation = env.get_observations()
                objs_info = env.collect_objs_info()

                action = controller.get_action(observation, objs_info)
                env.apply_action(torch.as_tensor(action, dtype=torch.float32, device=env.device))
                episode_length += 1

                reward, _ = env.get_reward()
                episode_max_reward = max(episode_max_reward, float(reward.item()))

                if bool(env.get_terminated().item()):
                    episode_success = True
                    logger.info(f"✅ Episode {episode_idx + 1} SUCCEEDED at step {episode_length}")
                    break

                # The VLM signalled DONE and the last trajectory has drained.
                if controller.is_done:
                    logger.info(
                        f"Planner finished (VLM signaled DONE) after "
                        f"{controller.high_level_planner.action_count} actions"
                    )
                    break

                if step > 0 and step % 100 == 0:
                    logger.info(
                        f"Step {step}/{max_episode_steps}, "
                        f"actions planned: {controller.high_level_planner.action_count}, "
                        f"trajectory progress: {controller.low_level_controller.trajectory_progress}/"
                        f"{controller.low_level_controller.trajectory_length}"
                    )

        except Exception as e:
            episode_crashed = True
            logger.error(f"💥 Episode {episode_idx + 1} CRASHED at step {episode_length}: {type(e).__name__}: {e}")
            try:
                env.reset_env()
            except Exception as reset_error:
                logger.warning(f"Failed to reset environment after crash: {reset_error}")

        if not episode_success and not episode_crashed:
            logger.warning(
                f"❌ Episode {episode_idx + 1} FAILED after {episode_length} steps "
                f"({controller.high_level_planner.action_count} high-level actions)"
            )

        episode_dir = output_dir / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        try:
            controller.high_level_planner.save_vlm_records(episode_dir)
        except Exception as save_error:
            logger.warning(f"Failed to save VLM records for episode {episode_idx}: {save_error}")

        episode_summary = {
            "episode": episode_idx,
            "success": episode_success,
            "crashed": episode_crashed,
            "length": episode_length,
            "num_actions": controller.high_level_planner.action_count,
            "max_reward": episode_max_reward,
            "action_history": controller.high_level_planner.action_history,
        }
        with open(episode_dir / "summary.json", "w") as f:
            json.dump(episode_summary, f, indent=2)
        episode_results.append(episode_summary)
        episode_object_states.append(
            {
                "episode_index": episode_idx,
                "episode_success": episode_success,
                "task_name": task_instruction,
                "objects_info": _to_serializable(objs_info),
            }
        )

    episode_results.sort(key=lambda r: r["episode"])
    num_success = sum(1 for r in episode_results if r["success"])
    num_crashed = sum(1 for r in episode_results if r.get("crashed", False))
    num_valid = len(episode_results) - num_crashed

    max_rewards = [float(r.get("max_reward", 0.0)) for r in episode_results]

    metrics = {
        "task_id": task_name,
        "task_name": task_name,
        "task_instruction": task_instruction,
        "model": model,
        "num_episodes": num_episodes,
        "num_success": num_success,
        "num_crashed": num_crashed,
        "success_rate": num_success / num_valid if num_valid > 0 else 0.0,
        "avg_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
        "episode_results": episode_results,
    }

    metrics_path = output_dir / "eval_metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "episode_object_states.json", "w") as f:
        json.dump(episode_object_states, f, indent=2)

    # Also emit eval_info.json so scripts/robowits/eval/summarize_results.py
    # picks these runs up alongside the policy evals.
    write_eval_info(output_dir, task_name, [r["success"] for r in episode_results], max_rewards)

    logger.info(f"Success rate: {num_success}/{num_valid} ({100 * metrics['success_rate']:.1f}%)")
    logger.info(f"Progress score (avg max reward): {metrics['avg_max_reward']:.3f}")
    logger.info(f"Results saved to {output_dir}")
    return metrics


# ==============================================================================
# Command line interface
# ==============================================================================


# Suite names come from summarize_results.SUITES so the seed and mutation paths
# cannot drift apart (they did once: --suite rejected "normal" while
# --mutations accepted it).
NAMED_SUITES = SUITES


def _successes_from_metrics(metrics: dict) -> list[bool]:
    """Per-episode success flags from a controller metrics dict."""
    return [bool(e.get("success", False)) and not e.get("crashed", False) for e in metrics.get("episode_results", [])]


def _rewards_from_metrics(metrics: dict) -> list[float]:
    """Per-episode progress scores (max milestone reward)."""
    return [float(e.get("max_reward", 0.0)) for e in metrics.get("episode_results", [])]


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
    # The controller reports progress through logger.info; stdlib logging
    # defaults to WARNING, which would silence it.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="Task id ('02') or registered task name ('robowits/02_03-v0')")
    group.add_argument("--suite", choices=sorted(NAMED_SUITES), help="Named task suite to sweep")
    group.add_argument(
        "--mutations",
        help="Run mutation variants for a suite ('robowits_10', 'robowits_20', 'all') or a task id like '02'",
    )
    parser.add_argument("--model", default="gpt-4o", help="VLM model name")
    parser.add_argument("--num-episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="Max env steps per episode (the VLA evals use 1000)"
    )
    parser.add_argument("--max-vlm-calls", type=int, default=20, help="Max high-level actions per episode")
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

    output_root = args.output_dir or Path("tmp/eval/vlm_controller")

    if args.mutations:

        def _run_one(task_name, out_dir, ds_json):
            return run_task_eval(
                task_name=task_name,
                output_dir=out_dir,
                model=args.model,
                num_episodes=args.num_episodes,
                max_episode_steps=args.max_episode_steps,
                max_vlm_calls=args.max_vlm_calls,
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

        eval_dataset_json = args.eval_dataset_json
        if eval_dataset_json is not None and Path(eval_dataset_json).is_dir():
            candidate = Path(eval_dataset_json) / f"{task_id}.json"
            eval_dataset_json = str(candidate) if candidate.exists() else None
            if eval_dataset_json is None:
                print(f"No eval dataset for task {task_id} in {args.eval_dataset_json}; using random placement")

        print(f"\n{'=' * 60}\n{task_name}\n{'=' * 60}")
        try:
            metrics = run_task_eval(
                task_name=task_name,
                output_dir=output_root / task_id.replace("/", "_"),
                model=args.model,
                num_episodes=args.num_episodes,
                max_episode_steps=args.max_episode_steps,
                max_vlm_calls=args.max_vlm_calls,
                eval_dataset_json=eval_dataset_json,
                show_viewer=args.show_viewer,
                overwrite=args.overwrite,
                genesis_backend=args.genesis_backend,
            )
            results[task_id] = metrics
        except Exception as e:  # a crashing task must not abort the sweep
            print(f"\n{'!' * 60}\nTASK CRASHED: {task_name}\nError: {e}\n{'!' * 60}")
            failed.append((task_id, str(e)))

    if len(task_ids) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        sweep = {
            "model": args.model,
            "num_episodes": args.num_episodes,
            "tasks": {tid: {"success_rate": m.get("success_rate")} for tid, m in results.items()},
            "failed_tasks": dict(failed),
        }
        with open(output_root / "sweep_summary.json", "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"\nSweep summary written to {output_root / 'sweep_summary.json'}")
        for tid, m in results.items():
            print(f"  {tid}: {m.get('success_rate', 0) * 100:.1f}%")
        for tid, err in failed:
            print(f"  {tid}: CRASHED ({err})")

    # Same convention as run_mutations: a sweep that evaluated most tasks is not
    # a failed job. Exit non-zero only when nothing at all produced results.
    if failed and not results:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
