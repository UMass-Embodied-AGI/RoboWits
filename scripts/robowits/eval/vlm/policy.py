#!/usr/bin/env python
"""Evaluate a VLM as a policy on RoboWits tasks, through motion tools on the env's own action interface.

The VLM sees what the VLA policies see (the three RGB cameras, the end-effector
and joint state, the task instruction) plus task-independent scene geometry and
the arm joint limits, and acts only through its harness's motion tools
(`move_eef` with absolute grasp-point targets, `move_joints` with absolute joint
targets) or `give_up`. There are no grasp/place macros and no object poses: a
non-learned harness validates each target's format and bounds, interpolates it
into env actions, and plays them open-loop at the env control rate. Like a VLA's
actions, an end-effector target the arm cannot reach is not rejected up front;
inverse kinematics gets as close as it can.

Each harness chooses the env's control mode and defines its tools on it: a tool
turns the model's arguments into a Motion, whose actions are computed step by
step in that control mode. The env is driven exactly as lerobot-eval drives a
VLA: built with `gs_gym.make` (the harness's control mode, "auto" physics
backend, 30 Hz), reset with seed 1000 + episode index on that eval-dataset
entry, one `env.step()` per action, and the episode ends when `step()` reports
`terminated` (the task's success check) or `truncated` (the step limit).

The interaction loop follows the RoboProbe L3 "Inspect EEF" harness used to
evaluate GPT-6 Astra on RoboDojo (arXiv:2609.24170, Appendix A): one tool call
per turn, unnamed targets hold their observed values, rejected targets return
an error without moving the robot, motions spend env steps in proportion to
their length; RoboProbe keeps images only on the two most recent observations, ours keeps them all.
Two harnesses share that loop and differ only in how the embodiment and the
images are represented to the model (--harness):

- `roboprobe`: the paper's interface, through the env's EE_ABS interface. World-frame
  targets named per dimension (`right_x`, ..., pitch/roll/yaw in degrees from a
  straight-down grasp, gripper 0-1), and three separate camera images per observation.
- `ours`: the env's own end-effector and joint interfaces. `move_eef` takes per-arm
  grasp-point position and axis-angle rotation in the robot base frame and drives the
  env's EE_ABS interface (its own IK); `move_joints` takes per-arm joint angles and
  drives the env's JOINT_ABS interface; both take the gripper opening in metres. The state reports both
  the end-effector pose and the joint angles. The prompt has Robot / Actions / Frames and
  geometry / Images sections that
  describe the end-effector frame and the wrist-camera image axes, and one
  composed image per observation (ego on top, both wrists at half size below,
  each labelled). The task instruction opens the first user message, together
  with the initial observation, and an accepted move_eef call returns its
  resulting observation as the tool output once the motion has played.
  RoboProbe instead sends the goal as a user turn of its own, acknowledges each
  call, and sends every observation as a separate user message.

Both harnesses pass their instructions as the request's `instructions`.

Unlike scripts/robowits/eval/vlm/controller.py, the model gets no object
state and no motion primitives.

Usage:
    OPENAI_API_KEY=... python scripts/robowits/eval/vlm/policy.py --task 02
    ANTHROPIC_API_KEY=... python scripts/robowits/eval/vlm/policy.py --task 02 --model claude-fable-5-2
    python scripts/robowits/eval/vlm/policy.py --harness roboprobe --suite robowits_10 --num-episodes 5
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import io
import json
import logging
import math
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
from langsmith import traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree

# Private, but it is exactly how wrap_openai records a Responses API output (incl. token usage for cost);
# recheck it on a langsmith upgrade.
from langsmith.wrappers._openai import _process_responses_api_output
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

import gs_gym
from gs_gym.envs.registry import TASK_REGISTRY

# controller.py (same directory) owns the suite/mutation sweep plumbing, and
# puts the parent eval/ directory on sys.path for summarize_results.
# isort: off
from controller import NAMED_SUITES, resolve_task, run_mutations  # noqa: E402
from summarize_results import resolve_suite, variant_stem, write_eval_info  # noqa: E402
# isort: on

logger = logging.getLogger(__name__)

# The VLM baselines render with Nyx unless GS_GYM_CAMERA_TYPE says otherwise (RoboWitsEnv reads it); Nyx runs on
# the CUDA backend, which "auto" picks for it.
os.environ.setdefault("GS_GYM_CAMERA_TYPE", "nyx")

# Every episode is one LangSmith trace in this project, with its model calls nested under it.
LANGSMITH_PROJECT = "robowits"


# ==============================================================================
# Embodiment geometry and the move_eef coordinate convention
# ==============================================================================

# World = robot base + (0, 0, 1.08); the base is not rotated (robowits_default.yaml).
ROBOT_BASE_POS = np.array([0.0, 0.0, 1.08])

# Grasp-point (Gripper_Tip_*) orientation for a straight-down grasp, in the world
# frame: approach (tip local z) along -z, jaws opening (tip local x) along +y.
# Derived from FK at the rest pose, where the tip points forward (local z ~ +x,
# local x = +y); this is that frame pitched 90 deg down about +y.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

ARMS = ("left", "right")
CAMERAS = ("ego", "wrist_left", "wrist_right")
# World-frame workspace bounds; the ours harness states the same box in the robot base frame.
POSITION_BOUNDS = {"x": (0.0, 1.0), "y": (-0.75, 0.75), "z": (0.76, 1.5)}
ANGLE_BOUNDS = {"pitch_deg": (-180.0, 180.0), "roll_deg": (-90.0, 90.0), "yaw_deg": (-180.0, 180.0)}
GRIPPER_BOUNDS = (0.0, 1.0)
DIMENSIONS = [f"{arm}_{d}" for arm in ARMS for d in (*POSITION_BOUNDS, *ANGLE_BOUNDS, "gripper")]

# Gripper command 1.0 = fully open = 0.1 m total jaw width (0.05 m per finger).
MAX_GRIPPER_WIDTH = 0.1

# Harness timing at the 30 Hz env control rate.
CONTROL_HZ = 30.0
MAX_LINEAR_SPEED = 0.3  # m/s
MAX_ANGULAR_SPEED = math.radians(120.0)  # rad/s
MIN_MOTION_STEPS = 3
SETTLE_STEPS = 5
GRIPPER_STEPS = 15

# lerobot-eval's default eval.seed: episode i of a VLA eval resets with seed 1000 + i.
EVAL_START_SEED = 1000


def pitch_roll_yaw_to_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> npt.NDArray[np.float64]:
    """World-frame grasp-point rotation for the move_eef angles.

    Zero is a straight-down grasp. Positive pitch tilts the approach toward +x
    (forward), positive roll toward +y (left), and yaw turns the grasp about the
    vertical (counter-clockwise seen from above).
    """
    tilt = Rotation.from_euler("ZXY", [yaw_deg, roll_deg, -pitch_deg], degrees=True)
    return tilt.as_matrix() @ R_DOWN


def matrix_to_pitch_roll_yaw(rot: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Inverse of pitch_roll_yaw_to_matrix."""
    yaw, roll, neg_pitch = Rotation.from_matrix(rot @ R_DOWN.T).as_euler("ZXY", degrees=True)
    return float(-neg_pitch), float(roll), float(yaw)


@dataclass
class ArmPose:
    """One arm's grasp-point pose in the world frame plus its gripper opening (0-1)."""

    pos: npt.NDArray[np.float64]
    rot: npt.NDArray[np.float64]
    gripper: float

    def dims(self) -> dict[str, float]:
        pitch, roll, yaw = matrix_to_pitch_roll_yaw(self.rot)
        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),
            "pitch_deg": pitch,
            "roll_deg": roll,
            "yaw_deg": yaw,
            "gripper": self.gripper,
        }


def unbatch(observation: dict[str, Any]) -> dict[str, Any]:
    """The single env's observation from a vector env's batched one (leading dim of 1, nested dicts kept)."""
    return {k: unbatch(v) if isinstance(v, dict) else np.asarray(v)[0] for k, v in observation.items()}


def state_from_observation(observation: dict[str, Any]) -> dict[str, ArmPose]:
    """Read both arms' poses from `agent_pos_ee`, the env's 14D EE state (base-relative, axis-angle).

    The env returns agent_pos_ee, agent_pos_ee_rot6d and agent_pos_joint in every observation mode (the mode only
    picks which of them `agent_pos` repeats, for LeRobot), so the harness reads the explicit keys it needs.
    """
    s = np.asarray(observation["agent_pos_ee"], dtype=np.float64)
    return {
        "right": ArmPose(
            s[0:3] + ROBOT_BASE_POS, Rotation.from_rotvec(s[6:9]).as_matrix(), float(s[12] / MAX_GRIPPER_WIDTH)
        ),
        "left": ArmPose(
            s[3:6] + ROBOT_BASE_POS, Rotation.from_rotvec(s[9:12]).as_matrix(), float(s[13] / MAX_GRIPPER_WIDTH)
        ),
    }


def ee_abs_action(poses: dict[str, ArmPose]) -> npt.NDArray[np.float64]:
    """Pack world-frame poses into the env's 14D EE_ABS action."""
    right, left = poses["right"], poses["left"]
    return np.concatenate(
        [
            right.pos - ROBOT_BASE_POS,
            left.pos - ROBOT_BASE_POS,
            Rotation.from_matrix(right.rot).as_rotvec(),
            Rotation.from_matrix(left.rot).as_rotvec(),
            [right.gripper * MAX_GRIPPER_WIDTH, left.gripper * MAX_GRIPPER_WIDTH],
        ]
    )


def rotation_angle(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    return float(Rotation.from_matrix(a.T @ b).magnitude())


# ==============================================================================
# RoboProbe harness: prompts, tools, state, targets
# ==============================================================================

ROBOPROBE_SYSTEM_PROMPT = """You are controlling a simulated bimanual robot: two 7-joint arms with parallel-jaw grippers, mounted side by side behind a table.
You receive RGB camera images, the current world-frame grasp-point state of both arms in the same 14 dimensions move_eef takes, arm joint angles as context you cannot command, and a task instruction.
Move with move_eef by naming only the world-frame dimensions you want to change. Cartesian targets must be estimated from RGB; no depth or world-coordinate query is available. Respond with exactly one tool call per turn. After each motion the next observation reports how far the grasp point ended up from what you asked for, so check it before assuming a motion landed. Two budgets run down at once and whichever empties first ends the episode: {max_calls} LLM calls, one per turn, and the environment's own step limit, reported with each observation as the env steps remaining. A motion spends env steps in proportion to how far it travels, so a small correction is cheap and a long reach is not.
Embodiment notes:
- World axes: +x points forward from the robot toward the far side of the table, +y to the robot's left, +z up. Positions are in metres.
- The robot base is at (0, 0, 1.08). The left arm's shoulder is at about (0, 0.21, 1.2) and the right arm's at about (0, -0.21, 1.2).
- The table top is at z = 0.76. It spans x from 0.17 to 1.02 and y from -0.75 to 0.75.
- The grasp point is midway between the fingertips.
- Orientation is in degrees from a straight-down grasp. All zero: the gripper points straight down with its jaws opening along y. Positive pitch tilts the fingertips toward +x (pitch 90 points them horizontally forward), positive roll tilts them toward +y, and yaw turns the gripper about the vertical axis (positive is counter-clockwise seen from above).
- Gripper: 0 is fully closed, 1 is fully open (0.10 m between the jaws). A gripper change is applied after the arm motion of the same call has finished.
- Cameras: 'ego' is a head camera looking down at the table; 'wrist_left' and 'wrist_right' are mounted on each gripper.
- Bounds: x in [{x_lo}, {x_hi}], y in [{y_lo}, {y_hi}], z in [{z_lo}, {z_hi}], pitch_deg in [-180, 180], roll_deg in [-90, 90], yaw_deg in [-180, 180], gripper in [0, 1]. A target outside the bounds or outside an arm's reach returns an error and does not move the robot.
- The episode succeeds as soon as the task is done; there is no way to declare success yourself."""

GOAL_TEMPLATE = "Goal: {instruction}"

IMAGE_OMITTED = "[earlier camera image omitted to save context]"

NO_TOOL_CALL_REMINDER = "Error: that reply called no tool, so the robot did not move. Call exactly one tool."

GIVE_UP_TOOL = {
    "type": "function",
    "name": "give_up",
    "description": "End the episode because it is unrecoverable. This does not declare success.",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why the episode is unrecoverable."},
            "hindsight": {"type": "string", "description": "What would have been needed."},
        },
        "required": ["reason", "hindsight"],
        "additionalProperties": False,
    },
    "strict": True,
}

ROBOPROBE_TOOLS = [
    {
        "type": "function",
        "name": "move_eef",
        "description": (
            "Move one or both grasp points to absolute world-frame targets. Name only the dimensions to change; "
            "unnamed dimensions hold their observed values. Naming dimensions of both arms moves them together."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "description": "Nonempty map from dimension name to target value.",
                    "properties": {d: {"type": "number"} for d in DIMENSIONS},
                    "additionalProperties": False,
                },
                "note": {"type": "string", "description": "Current observation and reason for this motion."},
            },
            "required": ["targets", "note"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    GIVE_UP_TOOL,
]


def joint_lines(joints: npt.NDArray[np.float64]) -> list[str]:
    # agent_pos_joint layout: [right_joints(7), right_gripper, left_joints(7), left_gripper]
    return [
        "Arm joint angles, radians, for context; not commandable:",
        " ".join(f"left_joint{i + 1}={joints[8 + i]:.3f}" for i in range(7)),
        " ".join(f"right_joint{i + 1}={joints[i]:.3f}" for i in range(7)),
    ]


def roboprobe_state_lines(poses: dict[str, ArmPose], _observation: dict[str, Any] | None = None) -> list[str]:
    lines = ["World-frame grasp-point state (metres; degrees from straight down):"]
    for arm in ARMS:
        d = poses[arm].dims()
        lines.append(f"{arm}_x={d['x']:.3f} {arm}_y={d['y']:.3f} {arm}_z={d['z']:.3f}")
        lines.append(
            f"{arm}_pitch_deg={d['pitch_deg']:.1f} {arm}_roll_deg={d['roll_deg']:.1f} {arm}_yaw_deg={d['yaw_deg']:.1f}"
        )
        lines.append(f"{arm}_gripper={d['gripper']:.2f}")
    return lines


def encode_jpeg(frame: npt.NDArray[np.uint8], quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def roboprobe_images(pixels: dict[str, npt.NDArray[np.uint8]], step: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for cam in CAMERAS:
        content.append({"type": "input_text", "text": f"camera '{cam}' (step {step}):"})
        content.append({"type": "input_image", "image_url": encode_jpeg(np.asarray(pixels[cam])), "detail": "high"})
    return content


def strip_images(item: dict[str, Any]) -> None:
    """Replace the images of an older observation with a placeholder, keeping its text.

    The observation is either a user message (parts under "content") or a tool output (parts under "output").
    """
    key = "output" if item.get("type") == "function_call_output" else "content"
    item[key] = [
        {"type": "input_text", "text": IMAGE_OMITTED} if part["type"] == "input_image" else part for part in item[key]
    ]


# ==============================================================================
# Harness: validate, execute
# ==============================================================================


class RejectedTarget(Exception):
    """A malformed or out-of-bounds move_eef request; the robot does not move."""


def resolve_roboprobe_targets(args: Any, current: dict[str, ArmPose]) -> tuple[dict[str, ArmPose], set[str]]:
    """Merge the named dimensions into the observed poses, validating names and bounds."""
    if not isinstance(args, dict):
        raise RejectedTarget("arguments must be a JSON object with 'targets' and 'note'")
    targets = args.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise RejectedTarget("'targets' must be a nonempty object of dimension -> number")
    goal: dict[str, ArmPose] = {}
    for arm in ARMS:
        dims = current[arm].dims()
        for key, value in targets.items():
            if key not in DIMENSIONS:
                raise RejectedTarget(f"unknown dimension '{key}'; valid dimensions are {', '.join(DIMENSIONS)}")
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                raise RejectedTarget(f"target for '{key}' must be a finite number")
            if not key.startswith(f"{arm}_"):
                continue
            dim = key[len(arm) + 1 :]
            lo, hi = POSITION_BOUNDS.get(dim) or ANGLE_BOUNDS.get(dim) or GRIPPER_BOUNDS
            if not lo <= value <= hi:
                raise RejectedTarget(f"target for '{key}' is outside [{lo:g}, {hi:g}]")
            dims[dim] = float(value)
        goal[arm] = ArmPose(
            pos=np.array([dims["x"], dims["y"], dims["z"]]),
            rot=pitch_roll_yaw_to_matrix(dims["pitch_deg"], dims["roll_deg"], dims["yaw_deg"]),
            gripper=dims["gripper"],
        )
    return goal, {k.split("_", 1)[0] for k in targets}


def plan_waypoints(start: dict[str, ArmPose], goal: dict[str, ArmPose]) -> list[dict[str, ArmPose]]:
    """Time-aligned straight-line/slerp arm motion, then a gripper phase, then a short settle."""
    duration = 0.0
    for arm in ARMS:
        duration = max(
            duration,
            float(np.linalg.norm(goal[arm].pos - start[arm].pos)) / MAX_LINEAR_SPEED,
            rotation_angle(start[arm].rot, goal[arm].rot) / MAX_ANGULAR_SPEED,
        )
    n = max(MIN_MOTION_STEPS, math.ceil(duration * CONTROL_HZ))
    ts = np.arange(1, n + 1) / n
    rots = {arm: Slerp([0.0, 1.0], Rotation.from_matrix([start[arm].rot, goal[arm].rot]))(ts) for arm in ARMS}
    waypoints = [
        {
            arm: ArmPose(
                pos=start[arm].pos + t * (goal[arm].pos - start[arm].pos),
                rot=rots[arm][i].as_matrix(),
                gripper=start[arm].gripper,
            )
            for arm in ARMS
        }
        for i, t in enumerate(ts)
    ]
    final = {arm: ArmPose(goal[arm].pos, goal[arm].rot, goal[arm].gripper) for arm in ARMS}
    gripper_changes = any(abs(goal[arm].gripper - start[arm].gripper) > 1e-3 for arm in ARMS)
    waypoints.extend([final] * ((GRIPPER_STEPS if gripper_changes else 0) + SETTLE_STEPS))
    return waypoints


def arrival_check(
    goal: dict[str, ArmPose],
    reached: dict[str, ArmPose],
    moving: set[str],
    gripper_text: Callable[[float], str] = lambda g: f"{g:.2f}",
    distance_text: Callable[[float], str] = lambda d: f"{1000 * d:.0f} mm",
    angle_text: Callable[[float], str] = lambda a: f"{math.degrees(a):.1f} deg",
    per_line: bool = False,
) -> str:
    """How far each moved arm ended up from its target: one line per arm ("left: ...") with `per_line`, else
    "left ...; right ..." (RoboProbe's)."""
    parts = []
    for arm in ARMS:
        if arm not in moving:
            continue
        pos_err = float(np.linalg.norm(reached[arm].pos - goal[arm].pos))
        rot_err = rotation_angle(reached[arm].rot, goal[arm].rot)
        part = f"{arm}{':' if per_line else ''} {distance_text(pos_err)} and {angle_text(rot_err)} away"
        if abs(reached[arm].gripper - goal[arm].gripper) > 0.05:
            part += f", gripper at {gripper_text(reached[arm].gripper)} (asked {gripper_text(goal[arm].gripper)})"
        parts.append(part)
    return ("\n" if per_line else "; ").join(parts)


# ==============================================================================
# Motions: what an accepted tool call does to the env
# ==============================================================================


@dataclass
class Motion:
    """An accepted tool call: `steps` env steps whose actions, in the env control mode `control_mode`, are computed
    one step at a time from the latest observation, then an arrival check on the observation after the last one.

    The env implements every control mode (RoboWitsEnv.set_control_mode); the loop switches it to `control_mode`
    before playing the motion, so each tool drives the env through the env's own interface for that mode.
    """

    control_mode: str
    steps: int
    action: Callable[[int, dict[str, Any]], npt.NDArray[np.float64]]  # (step within the motion, observation)
    arrival: Callable[[dict[str, Any]], str]
    # Each arm's commanded gripper opening (m) once the motion is done; the next motion starts from it.
    gripper: dict[str, float]


# A tool's handler: (parsed arguments, current observation, each arm's commanded gripper opening in m) -> Motion;
# raises RejectedTarget for a bad call. The gripper is position-controlled, so its grip force comes from the
# commanded opening being below the observed one: holding the *commanded* opening (not the observed one) is what
# keeps a closed gripper squeezing an object while the arm moves.
ToolHandler = Callable[[Any, dict[str, Any], dict[str, float]], Motion]


def ee_tool(
    resolve: Callable[[Any, dict[str, ArmPose]], tuple[dict[str, ArmPose], set[str]]],
    gripper_text: Callable[[float], str],
    hold_commanded_gripper: bool = True,
    distance_text: Callable[[float], str] = lambda d: f"{1000 * d:.0f} mm",
    angle_text: Callable[[float], str] = lambda a: f"{math.degrees(a):.1f} deg",
    per_line: bool = False,
) -> ToolHandler:
    """move_eef: a straight-line/slerp end-effector path played through the env's EE_ABS interface (its own IK).

    With `hold_commanded_gripper`, an omitted gripper, and the gripper during the arm motion, keep the last
    commanded opening; without it (RoboProbe's "unnamed dimensions hold their observed values"), the observed one.
    """

    def handle(args: Any, observation: dict[str, Any], commanded: dict[str, float]) -> Motion:
        current = state_from_observation(observation)
        if hold_commanded_gripper:
            current = {arm: ArmPose(c.pos, c.rot, commanded[arm] / MAX_GRIPPER_WIDTH) for arm, c in current.items()}
        goal, moving = resolve(args, current)
        waypoints = plan_waypoints(current, goal)
        return Motion(
            control_mode="EE_ABS",
            steps=len(waypoints),
            action=lambda i, _obs: ee_abs_action(waypoints[i]),
            arrival=lambda obs: arrival_check(
                goal, state_from_observation(obs), moving, gripper_text, distance_text, angle_text, per_line
            ),
            gripper={arm: goal[arm].gripper * MAX_GRIPPER_WIDTH for arm in ARMS},
        )

    return handle


# agent_pos_joint layout: [right_joints(7), right_gripper, left_joints(7), left_gripper]
JOINT_SLICES = {"right": slice(0, 7), "left": slice(8, 15)}
GRIPPER_INDEX = {"right": 7, "left": 15}

# Arm joint limits (rad) from marvin_pika.urdf, Joint1-7, identical for both arms. The env reads the same values
# from the built URDF (BimanualMarvinRobot.init_dof_indices), overriding its hard-coded defaults.
JOINT_LOWER = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
JOINT_UPPER = np.array([3.1067, 2.0944, 3.1067, 0.0, 3.1067, 1.0472, 1.5708])
MAX_JOINT_SPEED = math.radians(90.0)  # rad/s per joint, for move_joints


def joint_abs_action(
    joints: dict[str, npt.NDArray[np.float64]], gripper_m: dict[str, float]
) -> npt.NDArray[np.float64]:
    """The env's 16D JOINT_ABS action: [R_joints(7), R_gripper_width, L_joints(7), L_gripper_width]."""
    return np.concatenate([joints["right"], [gripper_m["right"]], joints["left"], [gripper_m["left"]]])


def full_arm_targets(args: Any, fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Both arms' targets from a full command: every call names both arms, each with every field in `fields`."""
    if not isinstance(args, dict):
        raise RejectedTarget("arguments must be a JSON object with 'right', 'left' and 'reasoning'")
    unknown = set(args) - {"right", "left", "reasoning"}
    if unknown:
        raise RejectedTarget(f"unknown argument(s) {sorted(unknown)}; use 'right', 'left' and 'reasoning'")
    names = ", ".join(f"'{f}'" for f in fields)
    targets = {}
    for arm in ARMS:
        target = args.get(arm)
        if not isinstance(target, dict):
            raise RejectedTarget(f"give the full command for both arms: '{arm}' must be an object with {names}")
        unknown = set(target) - set(fields)
        if unknown:
            raise RejectedTarget(f"unknown field(s) {sorted(unknown)} for '{arm}'; use {names}")
        missing = [f for f in fields if f not in target]
        if missing:
            raise RejectedTarget(f"'{arm}' is missing {', '.join(repr(f) for f in missing)}; every call needs {names}")
        targets[arm] = target
    return targets


def _gripper_target(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise RejectedTarget(f"'{name}' must be a finite number")
    if not 0.0 <= value <= MAX_GRIPPER_WIDTH:
        raise RejectedTarget(f"target for '{name}' is outside [0, {MAX_GRIPPER_WIDTH:.2f}]")
    return float(value)


def resolve_joint_targets(
    args: Any, observation: dict[str, Any], commanded: dict[str, float]
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, float], set[str]]:
    """Both arms' joint and gripper targets from a full command, validated against the limits."""
    targets = full_arm_targets(args, ("joints", "gripper"))
    joints, gripper = {}, {}
    for arm in ARMS:
        q = np.array(_finite_list(targets[arm]["joints"], 7, f"{arm}.joints"))
        for j, (value, lo, hi) in enumerate(zip(q, JOINT_LOWER, JOINT_UPPER, strict=True)):
            if not lo <= value <= hi:
                raise RejectedTarget(f"target for '{arm}.joints' joint {j + 1} is outside [{lo:.2f}, {hi:.2f}]")
        joints[arm], gripper[arm] = q, _gripper_target(targets[arm]["gripper"], f"{arm}.gripper")
    return joints, gripper, set(ARMS)


def joint_arrival_check(
    goal: dict[str, npt.NDArray[np.float64]], gripper: dict[str, float], observation: dict[str, Any], moving: set[str]
) -> str:
    reached = np.asarray(observation["agent_pos_joint"], dtype=np.float64)
    parts = []
    for arm in ARMS:
        if arm not in moving:
            continue
        err = np.abs(reached[JOINT_SLICES[arm]] - goal[arm])
        part = f"{arm}: joints within {err.max():.3f} rad of their targets"
        if err.max() >= 0.01:
            part += f" (largest at joint {int(err.argmax()) + 1})"
        g = float(reached[GRIPPER_INDEX[arm]])
        if abs(g - gripper[arm]) > 0.005:
            part += f", gripper at {g:.3f} m (asked {gripper[arm]:.3f} m)"
        parts.append(part)
    return "\n".join(parts)


def handle_move_joints(args: Any, observation: dict[str, Any], commanded: dict[str, float]) -> Motion:
    """move_joints: a straight joint-space line at up to MAX_JOINT_SPEED, played through the env's JOINT_ABS
    interface, then the gripper, then a settle. Grippers hold their commanded opening during the arm motion."""
    goal, gripper, moving = resolve_joint_targets(args, observation, commanded)
    obs_joints = np.asarray(observation["agent_pos_joint"], dtype=np.float64)
    start = {arm: obs_joints[JOINT_SLICES[arm]] for arm in ARMS}
    start_gripper = dict(commanded)
    travel = max(float(np.abs(goal[arm] - start[arm]).max()) for arm in ARMS)
    n = max(MIN_MOTION_STEPS, math.ceil(travel / MAX_JOINT_SPEED * CONTROL_HZ))
    actions = [
        joint_abs_action({arm: start[arm] + t * (goal[arm] - start[arm]) for arm in ARMS}, start_gripper)
        for t in np.arange(1, n + 1) / n
    ]
    gripper_changes = any(abs(gripper[arm] - start_gripper[arm]) > 1e-3 for arm in ARMS)
    actions += [joint_abs_action(goal, gripper)] * ((GRIPPER_STEPS if gripper_changes else 0) + SETTLE_STEPS)
    return Motion(
        control_mode="JOINT_ABS",
        steps=len(actions),
        action=lambda i, _obs: actions[i],
        arrival=lambda obs: joint_arrival_check(goal, gripper, obs, moving),
        gripper=gripper,
    )


# ==============================================================================
# Ours harness
# ==============================================================================

# Bounds of the ours harness: the world-frame workspace box in the robot base frame.
BASE_POSITION_BOUNDS = {
    k: (lo - ROBOT_BASE_POS[i], hi - ROBOT_BASE_POS[i]) for i, (k, (lo, hi)) in enumerate(POSITION_BOUNDS.items())
}

# End-effector frame and wrist-camera axes measured by FK on the Marvin URDF (Gripper_Tip_* and
# the wrist camera's offset_T on Pika_Gripper_Base_*, both arms identical): fingertips at
# x = +-0.045 in the tip frame, camera at (0.009, 0.049, -0.125) looking along +z with image
# right = -x and image up = +y (Genesis cameras look along their -z with +y up).
JOINT_LIMITS_TABLE = "\n".join(
    [
        "| joint | " + " | ".join(str(j) for j in range(1, 8)) + " |",
        "|---" * 8 + "|",
        "| lower | " + " | ".join(f"{v:.2f}" for v in JOINT_LOWER) + " |",
        "| upper | " + " | ".join(f"{v:.2f}" for v in JOINT_UPPER) + " |",
    ]
)

OURS_INSTRUCTIONS = (
    """\
You are an expert in robot manipulation. You control a bimanual robot to complete a tabletop \
task, one motion at a time. Each turn you see the robot's cameras and state and call \
exactly one tool: `move_eef` to move one or both grippers to absolute end-effector targets, \
`move_joints` to move one or both arms to absolute joint targets, or `give_up` if the episode \
cannot be recovered. The episode ends as soon as the task is done.

# Robot
Two 7-DoF arms (right, left), each ending in a parallel-jaw gripper. Each observation reports, per arm, \
the end-effector state (`pos`, `axis_angle`, `gripper`) and the joint angles (`joints`), in the formats \
described below.

End-effector frame (each arm's `pos` and `axis_angle`): the grasp point, midway between the fingertips. \
Its axes: +z points out through the fingertips (the approach direction; the wrist is behind the \
grasp point along -z); +x is the jaw axis, along which the two fingers open and close (each \
fingertip is 4.5 cm from the grasp point along +x or -x when fully open); +y is the side of the \
gripper the wrist camera is mounted on.

Gripper (`gripper`): the opening between the jaws in metres, 0.00 (closed) to 0.10 (fully open). It \
is position-controlled: jaws closing on an object stop at the object's width, so the observed \
opening stays above 0 while the gripper holds something, and they squeeze the object only while the \
commanded opening is below its width. Commanding the observed opening of a held object releases the \
squeeze.

Wrist cameras (`wrist_left`, `wrist_right`): each is fixed to its gripper, 12.5 cm behind the grasp \
point along -z and 4.9 cm toward +y, looking along +z through the fingers. Image up is the end \
effector's +y and image right is its -x, so the fingertips reach into the view from the bottom. \
Moving the gripper toward its own +x shifts what its wrist camera sees toward the image's right.

Arm joints (each arm's `joints`): 7 revolute joints, joint 1 at the shoulder to joint 7 at the wrist, \
angles in radians. You can command them directly with `move_joints`, or command the end effector with \
`move_eef` and let inverse kinematics pick the joint angles: it starts from the current ones at every \
control step and never moves a joint past its limit, so an end-effector motion can stop short of its \
target when a joint reaches a limit. Every joint moves at most 1.57 rad/s, so a motion that needs a large \
change of joint angles (such as the wrist turning over) lags behind its path and can end away from its \
target.

Collisions between the arms: the two arms share the space above the table, and nothing checks a motion \
for collisions. Each arm moves along its path even if the other arm is in the way, and the arms are \
solid, so a collision stops or deflects them, can knock objects over, and shows up as a large miss in \
the arrival check. The forearm and wrist trail behind the grasp point toward the arm's own shoulder, so \
an arm reaching across the robot's midline (y = 0 in the base frame) sweeps its forearm through the \
space in front of the other arm. To avoid collisions:
- Keep each arm on its own side of the midline (left arm at y > 0, right arm at y < 0) unless the other \
arm is out of the way.
- Before one arm crosses the midline or works near the other, first move the other arm back and up \
toward its own side, in a separate call.
- When both arms move in one call, they move at the same time along straight lines: make sure the \
paths do not cross and the arms, forearms included, stay apart along the way.
- Keep the two grasp points at least 15 cm apart unless the task needs the arms to work together \
closely, and then bring them together slowly, one arm at a time.

Joint limits in radians (both arms; the joints cannot move past them):
"""
    + JOINT_LIMITS_TABLE
    + """

# Actions
Every motion call gives the full command for both arms, plus a `reasoning`: for each of `right` and \
`left`, every field of that tool. Values are absolute targets, not deltas. A call with a missing arm \
or field returns an error and the robot does not move. A gripper change runs after the arm motion, at the 30 Hz control rate \
like the rest of the motion.

`move_eef` takes `pos` (x, y, z) in metres and `axis_angle` (rx, ry, rz), a rotation vector \
whose direction is the rotation axis and whose length is the angle in radians, both in the robot base \
frame, and `gripper`. This is the robot's own end-effector command and state format. The harness \
moves each named arm along a straight line (orientation interpolated spherically) at up to 0.3 m/s and \
2.09 rad/s, tracking the target with inverse kinematics. Targets must lie within x in [{x_lo:.2f}, \
{x_hi:.2f}], y in [{y_lo:.2f}, {y_hi:.2f}], z in [{z_lo:.2f}, {z_hi:.2f}] (robot base frame) and gripper in \
[0, 0.10]; otherwise the call returns an error and the robot does not move. A target the arm cannot \
reach is still executed: inverse kinematics gets as close as it can.

`move_joints` takes `joints` (7 angles in radians, joint 1 to joint 7) and `gripper`. This is \
the robot's own joint command and state format. The harness moves all joints of the named arms along a \
straight line in joint space, at up to 1.57 rad/s per joint, so they arrive together. Joint targets must \
lie within the joint limits and gripper in [0, 0.10]; otherwise the call returns an error and the robot \
does not move.

An accepted motion call returns once the motion has finished, and its result is the new observation. \
For `move_eef` it reports how far each grasp point ended up from its target, for `move_joints` how \
far each arm's joints ended up from theirs, and for either the gripper opening if it differs \
from the target. Check it before assuming a motion landed.

# Budgets
Two budgets run down at once and whichever empties first ends the episode: {max_calls} tool calls, \
one per turn, and {max_episode_steps} control steps at 30 Hz, the remainder reported with each \
observation. A motion spends control steps in proportion to how far it travels, so a small \
correction is cheap and a long reach is not.

# Frames and geometry (the same in every episode)
Robot base frame: x points away from the robot across the table, y to the robot's left, z up. It is \
the world frame translated by -(0, 0, 1.08) m with the same orientation: world = base + (0, 0, 1.08).
The table top is at z = 0.76 m in the world frame, i.e. z = -0.32 m in the robot base frame. It \
spans x from 0.17 to 1.02 m and y from -0.75 to 0.75 m.
The shoulders are at about (0, 0.21, 0.12) m (left arm) and (0, -0.21, 0.12) m (right arm) in the \
robot base frame.

# Images
Each observation has one image combining the three cameras at that step: `ego` (head camera looking \
down at the table) on top, `wrist_left` bottom left and `wrist_right` bottom right (cameras on each \
gripper, shown at half size). It is preceded by its step index."""
)

OURS_GOAL_TEMPLATE = "# Task Instruction\n{instruction}"
# Heads the initial observation in the first user message, after the task instruction.
INITIAL_OBSERVATION_HEADER = "# Initial Observation"


def _arm_target_schema(arm: str) -> dict[str, Any]:
    number = {"type": "number"}
    return {
        "type": "object",
        "description": f"Full target for the {arm} arm (its current values to keep it where it is).",
        "required": ["pos", "axis_angle", "gripper"],
        "properties": {
            "pos": {
                "type": "array",
                "items": number,
                "minItems": 3,
                "maxItems": 3,
                "description": "Grasp-point position (x, y, z) in metres, robot base frame.",
            },
            "axis_angle": {
                "type": "array",
                "items": number,
                "minItems": 3,
                "maxItems": 3,
                "description": "Grasp-point orientation as a rotation vector (rx, ry, rz) in radians, robot base frame.",
            },
            "gripper": {"type": "number", "description": "Jaw opening in metres, 0 (closed) to 0.10 (open)."},
        },
        "additionalProperties": False,
    }


def _arm_joint_schema(arm: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": f"Full target for the {arm} arm (its current values to keep it where it is).",
        "required": ["joints", "gripper"],
        "properties": {
            "joints": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 7,
                "maxItems": 7,
                "description": "Joint angles in radians, joint 1 (shoulder) to joint 7 (wrist), within the limits.",
            },
            "gripper": {"type": "number", "description": "Jaw opening in metres, 0 (closed) to 0.10 (open)."},
        },
        "additionalProperties": False,
    }


OURS_TOOLS = [
    {
        "type": "function",
        "name": "move_eef",
        "description": (
            "Move both grippers to absolute end-effector targets in the robot base frame. Give the full target "
            "for both arms on every call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "right": _arm_target_schema("right"),
                "left": _arm_target_schema("left"),
                "reasoning": {"type": "string", "description": "Current observation and reason for this motion."},
            },
            "required": ["right", "left", "reasoning"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "move_joints",
        "description": (
            "Move both arms to absolute joint-angle targets. Give the full target for both arms on every call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "right": _arm_joint_schema("right"),
                "left": _arm_joint_schema("left"),
                "reasoning": {"type": "string", "description": "Current observation and reason for this motion."},
            },
            "required": ["right", "left", "reasoning"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    GIVE_UP_TOOL,
]


# agent_pos_ee layout: [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
EE_SLICES = {"right": (slice(0, 3), slice(6, 9), 12), "left": (slice(3, 6), slice(9, 12), 13)}


def ours_state_lines(poses: dict[str, ArmPose], observation: dict[str, Any]) -> list[str]:
    """The env's own agent_pos_ee and agent_pos_joint values, labelled per arm (not recomputed).

    Both are observation vectors of the env, so this is the state a VLA trained on the EE or JOINT
    observation modes reads.
    """
    ee = np.asarray(observation["agent_pos_ee"], dtype=np.float64)
    joints = np.asarray(observation["agent_pos_joint"], dtype=np.float64)
    # The formats (frame, units, joint order) are stated once in the instructions, not in every observation.
    lines = ["End-effector state:"]
    for arm in ARMS:
        pos_s, rot_s, grip_i = EE_SLICES[arm]
        p, r = ee[pos_s], ee[rot_s]
        lines.append(
            f"{arm}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
            f"axis_angle=({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}) gripper={ee[grip_i]:.3f}"
        )
    lines.append("Joint angles:")
    for arm in ARMS:
        lines.append(f"{arm}: joints=({', '.join(f'{q:.3f}' for q in joints[JOINT_SLICES[arm]])})")
    return lines


def _finite_list(value: Any, n: int, name: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != n
        or not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    ):
        raise RejectedTarget(f"'{name}' must be a list of {n} finite numbers")
    return [float(v) for v in value]


def resolve_ours_targets(args: Any, current: dict[str, ArmPose]) -> tuple[dict[str, ArmPose], set[str]]:
    """Both arms' pos/axis_angle/gripper targets (robot base frame) from a full command, validated."""
    targets = full_arm_targets(args, ("pos", "axis_angle", "gripper"))
    goal: dict[str, ArmPose] = {}
    for arm in ARMS:
        base = _finite_list(targets[arm]["pos"], 3, f"{arm}.pos")
        for value, (axis, (lo, hi)) in zip(base, BASE_POSITION_BOUNDS.items(), strict=True):
            if not lo <= value <= hi:
                raise RejectedTarget(f"target for '{arm}.pos' {axis} is outside [{lo:.2f}, {hi:.2f}]")
        rot = Rotation.from_rotvec(_finite_list(targets[arm]["axis_angle"], 3, f"{arm}.axis_angle")).as_matrix()
        gripper = _gripper_target(targets[arm]["gripper"], f"{arm}.gripper") / MAX_GRIPPER_WIDTH
        goal[arm] = ArmPose(pos=np.array(base) + ROBOT_BASE_POS, rot=rot, gripper=gripper)
    return goal, set(ARMS)


WRIST_SIZE = (424, 240)
"""Half-width wrist views under the 848x480 ego view (all three cameras render at 848x480)."""


# Pixel size of the view labels; PIL's default bitmap font (~11 px) is hard to read in the composed image.
LABEL_FONT_SIZE = 22
LABEL_PAD = 6


@functools.cache
def _label_font() -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=LABEL_FONT_SIZE)


def label(frame: npt.NDArray[np.uint8], text: str) -> npt.NDArray[np.uint8]:
    """Draw `text` on a black box in the top-left corner."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _label_font()
    _, _, right, bottom = draw.textbbox((LABEL_PAD, LABEL_PAD), text, font=font)
    draw.rectangle((0, 0, right + LABEL_PAD, bottom + LABEL_PAD), fill=(0, 0, 0))
    draw.text((LABEL_PAD, LABEL_PAD), text, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def compose_views(pixels: dict[str, Any], title: str) -> npt.NDArray[np.uint8]:
    """Ego on top (labelled with `title`), wrist_left and wrist_right below at half size, each labelled."""
    ego = label(np.asarray(pixels["ego"]), title)
    wrists = [
        label(np.asarray(Image.fromarray(np.asarray(pixels[cam])).resize(WRIST_SIZE)), cam)
        for cam in ("wrist_left", "wrist_right")
    ]
    frame = np.concatenate([ego, np.concatenate(wrists, axis=1)], axis=0)
    return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]


def step_title(step: int) -> str:
    return f"step {step} (t = {step / CONTROL_HZ:.2f} s)"


def ours_images(pixels: dict[str, npt.NDArray[np.uint8]], step: int) -> list[dict[str, Any]]:
    title = step_title(step)
    return [
        {"type": "input_text", "text": title},
        {"type": "input_image", "image_url": encode_jpeg(compose_views(pixels, title), quality=85), "detail": "high"},
    ]


# ==============================================================================
# Harness registry
# ==============================================================================


@dataclass(frozen=True)
class Harness:
    """How the model interacts with the env: its tools (each driving one of the env's control interfaces) and how
    the embodiment and images are represented. The episode loop is shared."""

    name: str
    # The request's `instructions`, from (max_calls, max_episode_steps, image_horizon or None for every image).
    system_prompt: Callable[[int, int, int], str]
    goal_template: str
    # Tool schemas for the model, and a handler per motion tool (give_up is handled by the loop).
    tools: list[dict[str, Any]]
    tool_handlers: dict[str, ToolHandler]
    state_lines: Callable[[dict[str, ArmPose], dict[str, Any]], list[str]]
    images: Callable[[dict[str, npt.NDArray[np.uint8]], int], list[dict[str, Any]]]
    # RoboProbe repeats the instruction at the top of every observation (arXiv:2609.24170, Appendix A).
    repeat_instruction: bool = False
    # How the per-observation step budget is named; ours avoids sim-only wording so it also suits real robots.
    steps_label: str = "Env steps remaining before the episode ends"
    # True: an accepted move_eef call returns, once the motion has finished, the resulting observation as its
    # tool output. False (RoboProbe): it returns an acknowledgement and the observation follows as a user message.
    observation_as_tool_output: bool = False
    # True: the goal opens the first observation's user message. False (RoboProbe): it is a user turn of its own.
    goal_with_first_observation: bool = False
    # Appends RoboProbe's own joint-angle lines (the paper's format); ours reports joint angles in its
    # state_lines instead, next to the end-effector state.
    joint_state: bool = False
    # Observations that keep their images (older ones keep only their text); None keeps every image. Replacing an
    # old image with text changes an early part of the conversation, so every later token misses the prompt cache:
    # dropping images costs more than it saves. --image-horizon overrides it.
    image_horizon: int | None = None


def _bounds_kwargs(bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {f"{axis}_{side}": v for axis, (lo, hi) in bounds.items() for side, v in (("lo", lo), ("hi", hi))}


HARNESSES = {
    # The paper's interface: move_eef only, through the env's EE_ABS interface.
    "roboprobe": Harness(
        name="roboprobe",
        system_prompt=lambda max_calls, _steps, _horizon: ROBOPROBE_SYSTEM_PROMPT.format(
            max_calls=max_calls, **_bounds_kwargs(POSITION_BOUNDS)
        ),
        goal_template=GOAL_TEMPLATE,
        tools=ROBOPROBE_TOOLS,
        # The paper's "unnamed dimensions hold their observed values", the gripper included.
        tool_handlers={
            "move_eef": ee_tool(resolve_roboprobe_targets, lambda g: f"{g:.2f}", hold_commanded_gripper=False)
        },
        state_lines=roboprobe_state_lines,
        images=roboprobe_images,
        repeat_instruction=True,
        joint_state=True,
        image_horizon=2,  # the paper's: the two most recent observations
    ),
    # Both end-effector and joint control: move_eef through the env's EE_ABS interface, move_joints through its
    # JOINT_ABS interface (the env switches between them per motion).
    "ours": Harness(
        name="ours",
        system_prompt=lambda max_calls, steps, horizon: OURS_INSTRUCTIONS.format(
            max_calls=max_calls,
            max_episode_steps=steps,
            image_horizon=horizon,
            **_bounds_kwargs(BASE_POSITION_BOUNDS),
        ),
        goal_template=OURS_GOAL_TEMPLATE,
        tools=OURS_TOOLS,
        tool_handlers={
            # Metres and radians, like the rest of the ours state (RoboProbe keeps the paper's mm and deg)
            "move_eef": ee_tool(
                resolve_ours_targets,
                lambda g: f"{g * MAX_GRIPPER_WIDTH:.3f} m",
                distance_text=lambda d: f"{d:.3f} m",
                angle_text=lambda a: f"{a:.3f} rad",
                per_line=True,
            ),
            "move_joints": handle_move_joints,
        },
        state_lines=ours_state_lines,
        images=ours_images,
        steps_label="Control steps remaining before the episode ends",
        observation_as_tool_output=True,
        goal_with_first_observation=True,
        image_horizon=None,  # every image stays, so the conversation only ever grows and stays cached
    ),
}


# ==============================================================================
# Per-episode loop
# ==============================================================================


@dataclass
class Episode:
    success: bool = False
    crashed: bool = False
    gave_up: bool = False
    steps: int = 0
    llm_calls: int = 0
    max_reward: float = 0.0
    cost_usd: float | None = 0.0
    turns: list[dict[str, Any]] = field(default_factory=list)
    langsmith_run_id: str | None = None


# USD per 1M tokens: (input, cached input, output, cache write), as in LangSmith's price table and Anthropic's
# price list (5-minute cache writes). Other models get no cost.
PRICES_PER_MTOK = {
    "gpt-6-astra": (10.0, 1.0, 50.0, 10.0),  # OpenAI bills no cache writes
    "claude-fable-5-2": (8.0, 0.20, 40.0, 10.0),
}


def call_usage(response: Any) -> dict[str, int] | None:
    """Token counts of a Responses API reply (None for a replay, which has no usage)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    cache_write = getattr(details, "cache_write_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    return {
        "input": usage.input_tokens,
        "cached": cached,
        "cache_write": cache_write,
        "output": usage.output_tokens,
        "reasoning": reasoning,
    }


def call_cost(model: str, usage: dict[str, int] | None) -> float | None:
    if usage is None or model not in PRICES_PER_MTOK:
        return None
    price_in, price_cached, price_out, price_write = PRICES_PER_MTOK[model]
    uncached = usage["input"] - usage["cached"] - usage["cache_write"]
    return (
        uncached * price_in
        + usage["cached"] * price_cached
        + usage["cache_write"] * price_write
        + usage["output"] * price_out
    ) / 1e6


EVENTS_FILE = "events.jsonl"


class EventLog:
    """The episode as viewer/server.py shows it: <episode_dir>/events.jsonl, one JSON record per line.

    Every record is flushed as it is written, so the viewer can follow a running episode. Message images are
    written once, exactly as the model received them, to <episode_dir>/images/ and referenced by path.
    """

    def __init__(self, episode_dir: Path) -> None:
        self.dir = episode_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.dir / EVENTS_FILE, "w", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self._seq = 0
        self._images = 0

    def emit(self, kind: str, **fields: Any) -> None:
        self._seq += 1
        record = {"seq": self._seq, "ts": time.time(), "kind": kind, **fields}
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def parts(self, content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A message's content as text parts and image file references (images/NNNN.jpg)."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "input_text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                header, data = part["image_url"].split(",", 1)
                path = f"images/{self._images:04d}.{'png' if 'png' in header else 'jpg'}"
                (self.dir / "images").mkdir(exist_ok=True)
                (self.dir / path).write_bytes(base64.b64decode(data))
                self._images += 1
                parts.append({"type": "image", "path": path})
        return parts

    def close(self) -> None:
        self._file.close()


class ReplayClient:
    """Stands in for the OpenAI client by returning the tool calls of a recorded turns.json in order.

    The scene and harness are deterministic, so replaying a pilot's calls reproduces its trajectory
    (e.g. to record a video) without any API calls.
    """

    def __init__(self, turns_path: Path) -> None:
        self._turns = [t for t in json.loads(Path(turns_path).read_text()) if "tool" in t]
        self._next = 0
        self.responses = self

    def create(self, **_: Any) -> Any:
        if self._next >= len(self._turns):
            raise RuntimeError(f"replay ran out of recorded calls after {self._next}")
        turn = self._turns[self._next]
        self._next += 1
        call = SimpleNamespace(
            type="function_call",
            name=turn["tool"],
            arguments=json.dumps(turn["arguments"]),
            call_id=f"replay_{self._next}",
        )
        return SimpleNamespace(output=[call])


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


# Thinking counts toward max_tokens; streaming keeps a reply this long clear of HTTP timeouts.
CLAUDE_MAX_TOKENS = 64000


def _claude_blocks(parts: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses content parts as Messages API content blocks (text, base64 image)."""
    if isinstance(parts, str):
        return [{"type": "text", "text": parts}]
    blocks = []
    for part in parts:
        if part["type"] == "input_text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["type"] == "input_image":
            header, data = part["image_url"].split(",", 1)
            media_type = header.removeprefix("data:").removesuffix(";base64")
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        else:
            raise ValueError(f"unsupported content part {part['type']!r}")
    return blocks


def _claude_tool(tool: dict[str, Any]) -> dict[str, Any]:
    spec = {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
    if tool.get("strict"):
        spec["strict"] = True
    return spec


class ClaudeClient:
    """Serves the loop's Responses API calls (`client.responses.create`) from the Claude Messages API.

    Each call rebuilds the Messages request from the loop's Responses-format context. That context only ever
    grows (every image stays, see Harness.image_horizon), so each request extends the previous one unchanged:
    the thinking blocks sent back stay valid under Claude's prefix check, and the prompt cache covers the
    whole earlier conversation. A reply comes back as Responses-style output items: the assistant message
    itself (`claude_message`, sent back verbatim on later calls) plus `reasoning` / `message` /
    `function_call` views of it that the loop reads and the request builder skips.

    No refusal fallback: a declined turn is recorded as a turn without a tool call rather than silently
    answered by another model.
    """

    def __init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(timeout=900, max_retries=3)
        self.responses = self

    @staticmethod
    def _messages(items: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        def add(role: str, blocks: list[Any]) -> None:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": list(blocks)})

        for item in items:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    result = {"type": "tool_result", "tool_use_id": item["call_id"]}
                    add("user", [{**result, "content": _claude_blocks(item["output"])}])
                elif item.get("role") == "user":
                    add("user", _claude_blocks(item["content"]))
                else:
                    raise ValueError(f"unsupported request item {item!r:.200}")
            elif getattr(item, "type", None) == "claude_message":
                add("assistant", item.content)
        return messages

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: list[Any],  # noqa: A002 - the Responses API's own parameter name
        tools: list[dict[str, Any]],
        tool_choice: str,
        parallel_tool_calls: bool,
        reasoning: dict[str, Any],
    ) -> Any:
        if tool_choice != "required" or parallel_tool_calls:
            raise ValueError("ClaudeClient serves single, required tool calls only")
        with self._client.messages.stream(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=instructions,
            messages=self._messages(input),
            tools=[_claude_tool(t) for t in tools],
            # Forced tool use (`any`) is a 400 on these models; the prompts already ask for exactly one tool call
            # per turn, and a reply without one is a rejected turn.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": reasoning["effort"]},
            cache_control={"type": "ephemeral"},
        ) as stream:
            message = stream.get_final_message()
        return self._as_response(message)

    @staticmethod
    def _as_response(message: Any) -> Any:
        output: list[Any] = []
        if message.content:  # an empty assistant turn cannot be sent back
            output.append(SimpleNamespace(type="claude_message", content=message.content))
        for block in message.content:
            if block.type == "thinking" and block.thinking:
                output.append(SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=block.thinking)]))
            elif block.type == "text":
                text = SimpleNamespace(type="output_text", text=block.text)
                output.append(SimpleNamespace(type="message", role="assistant", content=[text]))
            elif block.type == "tool_use":
                output.append(
                    SimpleNamespace(
                        type="function_call", name=block.name, arguments=json.dumps(block.input), call_id=block.id
                    )
                )
        usage = message.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        return SimpleNamespace(
            output=output,
            message=message,
            stop_reason=message.stop_reason,
            usage=SimpleNamespace(
                # As in the Responses API: input_tokens is every prompt token, cached or not.
                input_tokens=usage.input_tokens + cache_read + cache_write,
                output_tokens=usage.output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=cache_read, cache_write_tokens=cache_write),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


def make_client(model: str) -> Any:
    if is_claude(model):
        return ClaudeClient()

    from openai import OpenAI

    # Not wrap_openai: create_response traces each call itself, in a format LangSmith renders.
    return OpenAI(timeout=600, max_retries=3)


def _field(item: Any, key: str) -> Any:
    """A key of a request item, which is a dict (ours) or an SDK object (the model's own output items)."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _chat_parts(parts: Any) -> Any:
    """Responses content parts as Chat Completions parts (text / image_url), which LangSmith renders."""
    if isinstance(parts, str):
        return parts
    converted = []
    for part in parts:
        kind = _field(part, "type")
        if kind in ("input_text", "output_text"):
            converted.append({"type": "text", "text": _field(part, "text")})
        elif kind == "input_image":
            converted.append({"type": "image_url", "image_url": {"url": _field(part, "image_url")}})
        else:
            converted.append(part)
    return converted


def chat_messages_for_trace(instructions: str, items: list[Any]) -> list[dict[str, Any]]:
    """The request as Chat Completions messages, for the LangSmith trace only.

    LangSmith's trace view renders images in Chat Completions messages, including tool messages, but shows
    the images of a Responses API `function_call_output` as raw base64. The model still gets the Responses
    request; the copies built here also keep later in-place edits (strip_images) out of the trace.
    Reasoning items are left out: they carry no readable content.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    for item in items:
        kind, role = _field(item, "type"), _field(item, "role")
        if kind == "function_call":
            call = {
                "id": _field(item, "call_id"),
                "type": "function",
                "function": {"name": _field(item, "name"), "arguments": _field(item, "arguments")},
            }
            if messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _field(item, "call_id"),
                    "content": _chat_parts(_field(item, "output")),
                }
            )
        elif role is not None:
            content = _chat_parts(_field(item, "content"))
            if role == "assistant" and isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            messages.append({"role": role, "content": content})
    return messages


def _trace_llm_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": chat_messages_for_trace(inputs["instructions"], inputs["input"]),
        "tools": [
            {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
            for t in inputs["tools"]
        ],
        "model": inputs["model"],
        "reasoning": inputs["reasoning"],
    }


def _trace_llm_outputs(response: Any) -> dict[str, Any]:
    """The reply as LangSmith records it: a Responses API output, or a Claude message with its usage."""
    message = getattr(response, "message", None)
    if message is None:
        return _process_responses_api_output(response)
    output = message.model_dump(exclude_none=True, mode="json")
    output.pop("usage", None)
    usage = response.usage
    output["usage_metadata"] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_token_details": {
            "cache_read": usage.input_tokens_details.cached_tokens,
            "cache_creation": usage.input_tokens_details.cache_write_tokens,
        },
    }
    return output


@traceable(
    name="responses.create",
    run_type="llm",
    process_inputs=_trace_llm_inputs,
    process_outputs=_trace_llm_outputs,
)
def create_response(
    client: Any,
    model: str,
    instructions: str,
    input: list[Any],  # noqa: A002 - the Responses API's own parameter name
    tools: list[dict[str, Any]],
    reasoning: dict[str, Any],
) -> Any:
    """One Responses API call, traced as an LLM run whose messages LangSmith renders (see chat_messages_for_trace)."""
    return client.responses.create(
        model=model,
        instructions=instructions,
        input=input,
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        reasoning=reasoning,
    )


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the env and client objects out of the trace."""
    keys = ("model", "reasoning_effort", "max_calls", "max_episode_steps", "image_horizon")
    return {
        "harness": inputs["harness"].name,
        "episode_dir": str(inputs["episode_dir"]),
        **{k: inputs[k] for k in keys},
    }


def _trace_outputs(ep: Episode) -> dict[str, Any]:
    keys = ("success", "gave_up", "steps", "llm_calls", "max_reward")
    return {k: getattr(ep, k) for k in keys}


@traceable(
    name="vlm_policy_episode",
    run_type="chain",
    project_name=LANGSMITH_PROJECT,
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
def run_episode(
    env: Any,
    initial_obs: dict[str, Any],
    client: Any,
    harness: Harness,
    episode_dir: Path,
    model: str,
    reasoning_effort: str,
    max_calls: int,
    max_episode_steps: int,
    image_horizon: int | None,
    record_video: bool = True,
    task_name: str = "",
    episode_index: int = -1,
) -> Episode:
    """One episode on a freshly reset single-env vector env (`initial_obs` is what its reset() returned)."""
    events = EventLog(episode_dir)
    try:
        return _run_episode(
            env, initial_obs, client, harness, episode_dir, model, reasoning_effort, max_calls,
            max_episode_steps, image_horizon, record_video, task_name, episode_index, events,
        )  # fmt: skip
    except Exception as e:
        events.emit("episode_end", crashed=True, error=f"{type(e).__name__}: {e}")
        raise
    finally:
        events.close()


def _run_episode(
    env: Any,
    initial_obs: dict[str, Any],
    client: Any,
    harness: Harness,
    episode_dir: Path,
    model: str,
    reasoning_effort: str,
    max_calls: int,
    max_episode_steps: int,
    image_horizon: int | None,
    record_video: bool,
    task_name: str,
    episode_index: int,
    events: EventLog,
) -> Episode:
    ep = Episode()
    run_tree = get_current_run_tree()
    ep.langsmith_run_id = None if run_tree is None else str(run_tree.id)
    instruction = env.call("task_description")[0]
    obs = initial_obs
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    instructions = harness.system_prompt(max_calls, max_episode_steps, image_horizon)
    goal_text = harness.goal_template.format(instruction=instruction)
    context: list[Any] = [] if harness.goal_with_first_observation else [{"role": "user", "content": goal_text}]
    events.emit(
        "episode_start",
        task=task_name,
        episode=episode_index,
        instruction=instruction,
        harness=harness.name,
        model=model,
        reasoning_effort=reasoning_effort,
        max_calls=max_calls,
        max_episode_steps=max_episode_steps,
        image_horizon=image_horizon,
        control_hz=CONTROL_HZ,
        instructions=instructions,
        tools=harness.tools,
    )
    if context:
        events.emit("message", role="user", step=0, parts=events.parts(goal_text))
    observation_messages: list[dict[str, Any]] = []
    arrival: str | None = None
    need_observation = True
    # The accepted move_eef call whose output is the next observation (observation_as_tool_output harnesses).
    pending_call_id: str | None = None
    motion_summary: str | None = None
    # Each arm's commanded gripper opening (m), from the observed one at reset; see ToolHandler.
    reset_joints = np.asarray(obs["agent_pos_joint"], dtype=np.float64)
    commanded_gripper = {arm: float(reset_joints[GRIPPER_INDEX[arm]]) for arm in ARMS}
    # Every env step at the control rate, so the video plays in real time.
    video = imageio.get_writer(episode_dir / "video.mp4", fps=CONTROL_HZ) if record_video else None
    if video is not None:
        video.append_data(compose_views(obs["pixels"], step_title(0)))
    truncated = False

    def step_env(action: npt.NDArray[np.float64]) -> bool:
        """One env.step(), as lerobot-eval takes it; returns True once the episode has ended."""
        nonlocal obs, truncated
        batch_obs, reward, terminated, trunc, _ = env.step(action[None].astype(np.float32))
        obs = unbatch(batch_obs)
        ep.steps += 1
        if video is not None:
            video.append_data(compose_views(obs["pixels"], step_title(ep.steps)))
        ep.max_reward = max(ep.max_reward, float(reward[0]))
        ep.success = bool(terminated[0])
        truncated = bool(trunc[0])
        return ep.success or truncated

    while ep.llm_calls < max_calls and not (ep.success or truncated):
        current = state_from_observation(obs)
        if need_observation:
            lines = [f"Instruction: {instruction}"] if harness.repeat_instruction else []
            if pending_call_id is not None:
                lines += [motion_summary, "Arrival check:", arrival]
            lines += harness.state_lines(current, obs)
            if harness.joint_state:
                lines += joint_lines(np.asarray(obs["agent_pos_joint"]))
            if arrival is not None and pending_call_id is None:
                lines.append(f"Arrival check for the previous move_eef target: {arrival}.")
            lines.append(f"{harness.steps_label}: {max_episode_steps - ep.steps}")
            if harness.goal_with_first_observation and not observation_messages:
                lines.insert(0, INITIAL_OBSERVATION_HEADER)
            content = [{"type": "input_text", "text": "\n".join(lines)}, *harness.images(obs["pixels"], ep.steps)]
            if harness.goal_with_first_observation and not observation_messages:
                content.insert(0, {"type": "input_text", "text": goal_text})
            if pending_call_id is not None:
                msg = {"type": "function_call_output", "call_id": pending_call_id, "output": content}
                events.emit(
                    "message", role="tool", call=ep.llm_calls, call_id=pending_call_id, step=ep.steps,
                    parts=events.parts(content),
                )  # fmt: skip
                pending_call_id = None
            else:
                msg = {"role": "user", "content": content}
                events.emit("message", role="user", call=ep.llm_calls, step=ep.steps, parts=events.parts(content))
            observation_messages.append(msg)
            context.append(msg)
            if image_horizon is not None:
                for old in observation_messages[:-image_horizon]:
                    strip_images(old)
            for cam in CAMERAS:
                Image.fromarray(np.asarray(obs["pixels"][cam])).save(frames_dir / f"call{ep.llm_calls:03d}_{cam}.jpg")
            need_observation = False

        started = time.monotonic()
        response = create_response(
            client,
            model,
            instructions,
            context,
            harness.tools,
            {"effort": reasoning_effort},
            langsmith_extra={
                # What wrap_openai attaches, so LangSmith prices the call.
                "metadata": {
                    "ls_provider": "anthropic" if is_claude(model) else "openai",
                    "ls_model_type": "chat",
                    "ls_model_name": model,
                    "ls_invocation_params": {"reasoning": {"effort": reasoning_effort}, "use_responses_api": True},
                }
            },
        )
        latency = time.monotonic() - started
        ep.llm_calls += 1
        context.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        usage = call_usage(response)
        cost = call_cost(model, usage)
        ep.cost_usd = None if cost is None or ep.cost_usd is None else ep.cost_usd + cost
        turn: dict[str, Any] = {"call": ep.llm_calls, "step": ep.steps}
        call = calls[0] if calls else None
        args = None
        if call is not None:
            with contextlib.suppress(json.JSONDecodeError):
                args = json.loads(call.arguments)
        text = "".join(
            getattr(c, "text", "")
            for item in response.output
            if getattr(item, "type", None) == "message"
            for c in (getattr(item, "content", None) or [])
        )
        reasoning = [
            getattr(s, "text", "")
            for item in response.output
            if getattr(item, "type", None) == "reasoning"
            for s in (getattr(item, "summary", None) or [])
        ]
        events.emit(
            "model_call",
            call=ep.llm_calls,
            step=ep.steps,
            latency_s=round(latency, 2),
            stop_reason=getattr(response, "stop_reason", None),
            usage=usage,
            cost_usd=cost,
            text=text or None,
            reasoning=reasoning or None,
            tool=None if call is None else call.name,
            call_id=None if call is None else call.call_id,
            arguments=args if args is not None or call is None else call.arguments,
        )
        if call is None:
            # tool_choice="required" should prevent this; treat it as a rejected turn. Claude takes no forced tool
            # choice (and a refusal ends a turn without one), and a request may not end on an assistant turn, so
            # it is told why the turn was rejected.
            stop_reason = getattr(response, "stop_reason", None)
            turn["error"] = "no tool call in response" + (f" (stop_reason {stop_reason})" if stop_reason else "")
            ep.turns.append(turn)
            if is_claude(model):
                content = [{"type": "input_text", "text": NO_TOOL_CALL_REMINDER}]
                context.append({"role": "user", "content": content})
                events.emit("message", role="user", call=ep.llm_calls, step=ep.steps, parts=events.parts(content))
            continue
        turn["tool"] = call.name
        turn["arguments"] = args

        if call.name == "give_up":
            ep.gave_up = True
            ep.turns.append(turn)
            break

        handler = harness.tool_handlers.get(call.name)
        try:
            if handler is None:
                raise RejectedTarget(f"unknown tool '{call.name}'")
            motion = handler(args, obs, commanded_gripper)
        except RejectedTarget as e:
            result = f"Error: {e}. The robot did not move."
            turn["result"] = result
            context.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
            events.emit(
                "message", role="tool", call=ep.llm_calls, call_id=call.call_id, step=ep.steps, rejected=True,
                parts=events.parts(result),
            )  # fmt: skip
            ep.turns.append(turn)
            continue

        if harness.observation_as_tool_output:
            # Answered by the next observation, once the motion has played (see the top of the loop).
            pending_call_id = call.call_id
        else:
            result = (
                f"Accepted: playing {motion.steps} waypoints ({motion.steps / CONTROL_HZ:.1f}s). "
                "The next observation reports how close the grasp point landed."
            )
            context.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
            events.emit(
                "message", role="tool", call=ep.llm_calls, call_id=call.call_id, step=ep.steps,
                parts=events.parts(result),
            )  # fmt: skip
            turn["result"] = result

        if motion.control_mode != env.call("control_mode")[0]:
            env.call("set_control_mode", motion.control_mode)
        aborted = None
        for i in range(motion.steps):
            try:
                done = step_env(motion.action(i, obs))
            except ValueError as e:  # the robot's IK raises ValueError("IK failed") if its solver errors
                aborted = str(e)
                break
            if done:
                break
        arrival = motion.arrival(obs)
        commanded_gripper = motion.gripper
        if aborted:
            arrival += (
                f"\nMotion aborted: {aborted}."
                if harness.observation_as_tool_output
                else f" (motion aborted: {aborted})"
            )
        turn["arrival"] = arrival
        turn["steps_used"] = ep.steps - turn["step"]
        if harness.observation_as_tool_output:
            motion_summary = (
                f"Motion finished after {turn['steps_used']} control steps ({turn['steps_used'] / CONTROL_HZ:.1f} s)."
            )
            turn["result"] = motion_summary
        events.emit(
            "motion",
            call=ep.llm_calls,
            control_mode=motion.control_mode,
            start_step=turn["step"],
            step=ep.steps,
            steps_used=turn["steps_used"],
            arrival=arrival,
            aborted=aborted,
            max_reward=ep.max_reward,
            success=ep.success,
            truncated=truncated,
        )
        ep.turns.append(turn)
        need_observation = True

    if video is not None:
        video.close()
    events.emit(
        "episode_end",
        success=ep.success,
        gave_up=ep.gave_up,
        truncated=truncated,
        crashed=False,
        steps=ep.steps,
        llm_calls=ep.llm_calls,
        max_reward=ep.max_reward,
        cost_usd=ep.cost_usd,
        video="video.mp4" if record_video else None,
    )
    return ep


def run_task_eval(
    task_name: str,
    output_dir: Path,
    harness: str = "ours",
    model: str = "gpt-6-astra",
    reasoning_effort: str = "medium",
    num_episodes: int = 1,
    episodes: list[int] | None = None,
    max_episode_steps: int = 1000,
    max_calls: int = 100,
    image_horizon: int | None = None,
    eval_dataset_json: str | None = None,
    show_viewer: bool = False,
    overwrite: bool = False,
    genesis_backend: str = "auto",
    record_video: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the move_eef VLM policy on one gs-gym task; returns the metrics dict.

    `client` defaults to the OpenAI client; pass a ReplayClient to re-execute recorded calls.
    `episodes` runs only those eval-dataset indices instead of 0..num_episodes-1.
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown gs-gym task: '{task_name}'")
    harness_spec = HARNESSES[harness]
    image_horizon = harness_spec.image_horizon if image_horizon is None else image_horizon
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in [*output_dir.glob("episode_*"), output_dir / "eval_metrics"]:
            if path.exists():
                shutil.rmtree(path)

    # Episodes with a summary are complete; load them back so the success rate
    # covers every episode, and skip building a scene if nothing is left to run.
    results: dict[int, dict[str, Any]] = {}
    indices = list(episodes) if episodes is not None else list(range(num_episodes))
    for idx in indices:
        summary = output_dir / f"episode_{idx:03d}" / "summary.json"
        if summary.exists():
            results[idx] = json.loads(summary.read_text())

    if len(results) < len(indices):
        # The same constructor and kwargs lerobot-eval uses for --env.type=gs_gym (GsGymEnv.gym_kwargs in
        # third_party/lerobot/src/lerobot/envs/configs.py), in the harness's control mode. The observation mode
        # only picks which vector `agent_pos` repeats; the harness reads agent_pos_ee / agent_pos_joint directly.
        env = gs_gym.make(
            task_name,
            n_envs=1,
            show_viewer=show_viewer,
            max_episode_steps=max_episode_steps,
            control_freq=CONTROL_HZ,
            # The constructor's mode; each motion switches the env to the interface its tool uses.
            control_mode="EE_ABS",
            observation_mode="EE",
            genesis_backend=genesis_backend,
            device=genesis_backend,
            eval_dataset_json=eval_dataset_json,
        )
        client = client or make_client(model)
        for idx in indices:
            if idx in results:
                continue
            logger.info(f"{task_name}: episode index {idx} ({indices.index(idx) + 1}/{len(indices)})")
            # lerobot-eval resets in order, so its episode i plays eval-dataset entry i with seed 1000 + i;
            # pointing the env at entry idx first lets an --episodes subset land on the same scenes.
            env.call("set_eval_dataset_idx", idx)
            initial_obs, _ = env.reset(seed=EVAL_START_SEED + idx)
            episode_dir = output_dir / f"episode_{idx:03d}"
            # A replay makes no model calls, so it is not traced.
            try:
                with tracing_context(enabled=False if isinstance(client, ReplayClient) else None):
                    ep = run_episode(
                        env,
                        unbatch(initial_obs),
                        client,
                        harness_spec,
                        episode_dir,
                        model,
                        reasoning_effort,
                        max_calls,
                        max_episode_steps,
                        image_horizon,
                        record_video,
                        task_name,
                        idx,
                        langsmith_extra={
                            "name": f"{harness} {variant_stem(task_name)} ep{idx:03d}",
                            "metadata": {"harness": harness, "task": task_name, "episode": idx, "model": model},
                        },
                    )
            except Exception as e:
                logger.error(f"Episode {idx + 1} crashed: {type(e).__name__}: {e}")
                ep = Episode(crashed=True)
            summary = {
                "episode": idx,
                "harness": harness,
                "success": ep.success,
                "crashed": ep.crashed,
                "gave_up": ep.gave_up,
                "length": ep.steps,
                "llm_calls": ep.llm_calls,
                "max_reward": ep.max_reward,
                "cost_usd": ep.cost_usd,
                "langsmith_run_id": ep.langsmith_run_id,
            }
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "turns.json").write_text(json.dumps(ep.turns, indent=2))
            # A crashed episode (usually an API failure) gets no summary, so a rerun retries it.
            if not ep.crashed:
                (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            results[idx] = summary
            logger.info(
                f"Episode index {idx}: success={ep.success} steps={ep.steps} calls={ep.llm_calls} gave_up={ep.gave_up}"
            )

    # The metrics describe the whole task directory, not just this call: fold in every episode an earlier call
    # completed, or re-running one episode (--episodes 1) would replace the task's aggregate with that episode.
    for summary in output_dir.glob("episode_*/summary.json"):
        record = json.loads(summary.read_text())
        results.setdefault(int(record["episode"]), record)

    episodes = [results[i] for i in sorted(results)]
    max_rewards = [float(r["max_reward"]) for r in episodes]
    num_success = sum(1 for r in episodes if r["success"])
    num_valid = sum(1 for r in episodes if not r["crashed"])
    metrics = {
        "task_id": task_name,
        "task_name": task_name,
        "harness": harness,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "num_episodes": len(episodes),
        "episode_indices": sorted(results),
        "num_success": num_success,
        "num_crashed": len(episodes) - num_valid,
        "success_rate": num_success / num_valid if num_valid else 0.0,
        "avg_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
        "episode_results": episodes,
    }
    metrics_path = output_dir / "eval_metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    write_eval_info(output_dir, task_name, [r["success"] for r in episodes], max_rewards)
    logger.info(f"{task_name}: success {num_success}/{num_valid}, results in {output_dir}")
    return metrics


# ==============================================================================
# Command line interface
# ==============================================================================


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="Task id ('02') or registered task name ('robowits/02_03-v0')")
    group.add_argument(
        "--suite", help=f"Named task suite to sweep ({', '.join(sorted(NAMED_SUITES))}) or a single task id like '02'"
    )
    group.add_argument(
        "--mutations",
        help="Run mutation variants for a suite ('robowits_10', 'robowits_20', 'all') or a task id like '02'",
    )
    parser.add_argument(
        "--harness",
        default="ours",
        choices=sorted(HARNESSES),
        help="How the embodiment and images are represented: 'roboprobe' (arXiv:2609.24170) or 'ours' "
        "(the env's EE_ABS/EE format, sectioned prompt and composed images)",
    )
    parser.add_argument(
        "--model",
        default="gpt-6-astra",
        help="VLM model name; a claude-* model (e.g. claude-fable-5-2) runs through the Claude Messages API",
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", help="Reasoning effort (Responses API reasoning / Claude effort)"
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="Episodes per task (indices 0..N-1)")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Run only these eval-dataset indices (overrides --num-episodes)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="Env step budget per episode (the VLA evals use 1000)"
    )
    parser.add_argument("--max-calls", type=int, default=100, help="LLM call budget per episode")
    parser.add_argument(
        "--image-horizon",
        type=int,
        default=None,
        help="Observations that keep their images (default: the harness's: every image for ours, 2 for roboprobe)",
    )
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
        help="Genesis physics backend; 'auto' (as in the VLA evals) uses each task's preferred backend",
    )
    parser.add_argument("--show-viewer", action="store_true", help="Show the Genesis viewer")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate existing episodes")
    parser.add_argument("--no-video", action="store_true", help="Skip the per-episode video.mp4")
    parser.add_argument(
        "--replay-turns",
        type=Path,
        default=None,
        help="Re-execute the tool calls of a recorded episode_*/turns.json (no API calls); needs --task and "
        "--num-episodes 1 with the same --eval-dataset-json",
    )
    args = parser.parse_args()
    horizon = args.image_horizon if args.image_horizon is not None else HARNESSES[args.harness].image_horizon
    if is_claude(args.model) and horizon is not None and not args.replay_turns:
        # Dropping old images edits earlier turns, which invalidates Claude's thinking blocks (a 400 error).
        parser.error(f"--model {args.model} needs every image kept: use --harness ours without --image-horizon")
    n_runs = len(args.episodes) if args.episodes else args.num_episodes
    if args.replay_turns and (not args.task or n_runs != 1):
        parser.error("--replay-turns needs --task and a single episode (--num-episodes 1 or one --episodes index)")

    output_root = args.output_dir or Path("tmp/eval/vlm_policy") / args.harness

    def run_one(task_name: str, out_dir: Path, ds_json: str | None) -> dict[str, Any]:
        return run_task_eval(
            task_name=task_name,
            output_dir=out_dir,
            harness=args.harness,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            num_episodes=args.num_episodes,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            max_calls=args.max_calls,
            image_horizon=args.image_horizon,
            eval_dataset_json=ds_json,
            show_viewer=args.show_viewer,
            overwrite=args.overwrite,
            genesis_backend=args.genesis_backend,
            record_video=not args.no_video,
            client=ReplayClient(args.replay_turns) if args.replay_turns else None,
        )

    if args.mutations:
        return run_mutations(args, run_one, output_root)

    task_ids = resolve_suite(args.suite) if args.suite else [args.task]
    failed: list[str] = []
    for task_id in task_ids:
        eval_dataset_json = args.eval_dataset_json
        if eval_dataset_json is not None and Path(eval_dataset_json).is_dir():
            candidate = Path(eval_dataset_json) / f"{task_id}.json"
            eval_dataset_json = str(candidate) if candidate.exists() else None
            if eval_dataset_json is None:
                print(f"No eval dataset for task {task_id} in {args.eval_dataset_json}; using random placement")
        try:
            run_one(resolve_task(task_id), output_root / task_id.replace("/", "_"), eval_dataset_json)
        except Exception as e:  # a crashing task must not abort the sweep
            print(f"TASK CRASHED: {task_id}: {e}")
            failed.append(task_id)
    return 1 if failed and len(failed) == len(task_ids) else 0


if __name__ == "__main__":
    sys.exit(main())
