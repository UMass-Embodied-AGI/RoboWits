"""Base class for bimanual robots in Genesis environments.

Two arms of ``ARM_DOF`` revolute joints each, and a parallel gripper of two prismatic fingers per arm
(each finger moves half the gripper width, in the direction given by ``FINGER_SIGNS``).
Subclasses supply the robot's data (joint and link names, gains, limits, reset pose, URDF, rest-pose
footprint, wrist camera mount); everything else (control modes, IK, the joint-state layout) lives here.

Layouts used throughout, for ``n = ARM_DOF``:

- physical / sequential (``2 * (n + 2)``): ``[R_arm(n), R_finger(2), L_arm(n), L_finger(2)]``
- user-facing joint vector (``JOINT_DIM = 2 * (n + 1)``): ``[R_arm(n), R_grip(1), L_arm(n), L_grip(1)]``,
  where the gripper value is the total width (``FINGER_SIGNS``-weighted sum of both fingers)
- EE actions: 14D (EE_ABS / EE_DELTA) or 20D (EE_ABS_ROT6D), independent of ``ARM_DOF``
"""

import math
from typing import ClassVar

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces

from gs_gym.common.utils.math_utils import quat_mul, rot6d_to_quat

#: An axis-aligned box in world coordinates: {"x": (lo, hi), "y": (lo, hi), "z": (lo, hi)}
BBox = dict[str, tuple[float, float]]


class BimanualRobot:
    """Bimanual robot with two ``ARM_DOF``-joint arms and two-finger grippers.

    Supports control modes:
    - JOINT_ABS: Absolute joint position control (JOINT_DIM: arm + 1 gripper width per arm)
    - JOINT_DELTA: Delta joint position control (JOINT_DIM, added to current)
    - EE_ABS: Absolute end-effector control with IK (14D)
    - EE_ABS_ROT6D: Absolute end-effector control with IK, 6D rotations (20D)
    - EE_DELTA: Delta end-effector control with IK (14D)
    """

    #: Available control modes for this robot
    CONTROL_MODES: ClassVar[list[str]] = ["JOINT_ABS", "JOINT_DELTA", "EE_ABS", "EE_ABS_ROT6D", "EE_DELTA"]

    #: Fastest an arm joint's position target may move (rad/s), in every control mode. The real robots'
    #: controllers limit joint commands to 1.57 rad/s; here the limit also
    #: applies to the IK's output, which otherwise can jump to a distant solution in one step when the nearby one
    #: runs into joint limits, snapping the arm across at tens of rad/s.
    MAX_JOINT_SPEED: ClassVar[float] = math.radians(90.0)

    # ── Supplied by subclasses ────────────────────────────────────────────
    #: Joint names per arm, base to tip; each list has ARM_DOF entries
    RIGHT_ARM_JOINTS: ClassVar[list[str]]
    LEFT_ARM_JOINTS: ClassVar[list[str]]
    #: The two prismatic finger joints per gripper
    RIGHT_GRIPPER_JOINTS: ClassVar[list[str]]
    LEFT_GRIPPER_JOINTS: ClassVar[list[str]]

    #: End-effector link names (the frames EE actions and observations refer to)
    EE_LINK_RIGHT: ClassVar[str]
    EE_LINK_LEFT: ClassVar[str]
    #: Gripper base link names (used for wrist camera attachment)
    GRIPPER_BASE_LINK_RIGHT: ClassVar[str]
    GRIPPER_BASE_LINK_LEFT: ClassVar[str]
    #: Wrist camera pose relative to the gripper base link (4x4)
    WRIST_CAMERA_OFFSET_T: ClassVar[np.ndarray]

    #: PD gains and force range per arm: ARM_DOF arm joints, then the 2 finger joints
    KP: ClassVar[tuple[float, ...]]
    KV: ClassVar[tuple[float, ...]]
    FORCE_RANGE: ClassVar[tuple[float, ...]]

    #: Arm joint limits used until init_dof_indices() reads them from the built URDF (both arms)
    ARM_LOWER: ClassVar[tuple[float, ...]]
    ARM_UPPER: ClassVar[tuple[float, ...]]
    #: Upper limit of one finger joint; a gripper's total width spans [0, 2 * FINGER_MAX]
    FINGER_MAX: ClassVar[float]
    #: Direction each finger joint opens in: +1 if its position grows with the width, -1 if it
    #: shrinks (a finger whose URDF range is [-FINGER_MAX, 0])
    FINGER_SIGNS: ClassVar[tuple[float, float]] = (1.0, 1.0)
    #: Extra gs.morphs.URDF options, e.g. decompose_robot_error_threshold for links whose convex hull is a poor
    #: collision shape (Genesis never decomposes robot links by default)
    URDF_MORPH_OPTIONS: ClassVar[dict] = {}
    #: Friction against MPM/SPH particles (Genesis's legacy coupler). Its default, 0.1, cannot lift a grasped
    #: deformable: the coupler only applies friction while a finger moves into the particles, so a closed grip
    #: that lifts barely holds anything (task 20's foam ball stayed on the table; at 1.0 it lifts 17.7 cm).
    #: Rigid-rigid contact does not use it.
    COUP_FRICTION: ClassVar[float] = 1.0

    #: Reset pose: arm joints per arm, and the per-finger opening
    RESET_RIGHT_ARM: ClassVar[tuple[float, ...]]
    RESET_LEFT_ARM: ClassVar[tuple[float, ...]]
    RESET_FINGER: ClassVar[float]

    #: World-frame link-center bounds of each arm at the reset pose, with the default base position.
    #: Keys: "forearm_right", "forearm_left" (forearm/wrist/gripper) and "full_right", "full_left".
    REST_FOOTPRINT: ClassVar[dict[str, BBox]]

    # ── Derived ────────────────────────────────────────────────────────────
    #: Arm joints per arm, and the user-facing joint vector size (arm + gripper width, both arms)
    ARM_DOF: ClassVar[int]
    JOINT_DIM: ClassVar[int]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "RIGHT_ARM_JOINTS" not in cls.__dict__:
            return  # intermediate class without robot data
        cls.ARM_DOF = len(cls.RIGHT_ARM_JOINTS)
        cls.JOINT_DIM = 2 * (cls.ARM_DOF + 1)
        n = cls.ARM_DOF
        checks = {
            "LEFT_ARM_JOINTS": len(cls.LEFT_ARM_JOINTS) == n,
            "RIGHT_GRIPPER_JOINTS": len(cls.RIGHT_GRIPPER_JOINTS) == 2,
            "LEFT_GRIPPER_JOINTS": len(cls.LEFT_GRIPPER_JOINTS) == 2,
            "KP": len(cls.KP) == n + 2,
            "KV": len(cls.KV) == n + 2,
            "FORCE_RANGE": len(cls.FORCE_RANGE) == n + 2,
            "ARM_LOWER": len(cls.ARM_LOWER) == n,
            "ARM_UPPER": len(cls.ARM_UPPER) == n,
            "RESET_RIGHT_ARM": len(cls.RESET_RIGHT_ARM) == n,
            "RESET_LEFT_ARM": len(cls.RESET_LEFT_ARM) == n,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise TypeError(f"{cls.__name__}: wrong length for {bad} (ARM_DOF={n}, 2 fingers per gripper)")

    @classmethod
    def default_urdf_path(cls) -> str:
        """Path to the robot's URDF."""
        raise NotImplementedError

    def __init__(
        self,
        n_envs: int,
        scene,  # Genesis scene object
        args,  # Robot args
        device: torch.device,
        control_mode: str = "JOINT_ABS",
        ee_link_right: str | None = None,
        ee_link_left: str | None = None,
        urdf_path: str | None = None,
        control_dt: float | None = None,
    ) -> None:
        """Initialize the robot.

        Args:
            n_envs: Number of parallel environments
            scene: Genesis scene object
            args: Robot configuration arguments
            device: Torch device
            control_mode: Control mode (one of CONTROL_MODES)
            ee_link_right: Override EE link name for right arm (default: class constant)
            ee_link_left: Override EE link name for left arm (default: class constant)
            urdf_path: Override URDF path (default: default_urdf_path())
            control_dt: Seconds between actions. With it, every arm joint target moves at most
                MAX_JOINT_SPEED * control_dt per action (see _limit_arm_targets); without it, unlimited.
        """
        self.n_envs = n_envs
        self.scene = scene
        self.args = args
        self.device = device
        self.max_joint_step: float | None = None if control_dt is None else self.MAX_JOINT_SPEED * control_dt
        # The last commanded arm joint targets (right, left), each (n_envs, ARM_DOF): what the next step's limit
        # is measured from. Set at reset; None until then (the limit then starts from the measured joints).
        self._arm_cmd: tuple[torch.Tensor, torch.Tensor] | None = None

        self._ee_link_right_name = ee_link_right or self.EE_LINK_RIGHT
        self._ee_link_left_name = ee_link_left or self.EE_LINK_LEFT

        # Validate control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control mode '{control_mode}'. Available: {self.CONTROL_MODES}")
        self.control_mode = control_mode

        self._urdf_path = urdf_path or self.default_urdf_path()

        # Initialize IK solver (lazy - created on first use)
        self._ik_solver = None

        # Create robot entity from URDF
        # Keep gripper base links so wrist cameras can attach to them
        morph = gs.morphs.URDF(
            file=self._urdf_path,
            pos=args.position,
            euler=args.euler,
            quat=None,
            visualization=True,
            collision=True,
            requires_jac_and_IK=True,
            scale=1.0,
            convexify=True,
            merge_fixed_links=False,
            fixed=True,  # Base is fixed to world
            links_to_keep=list(
                {
                    self.GRIPPER_BASE_LINK_RIGHT,
                    self.GRIPPER_BASE_LINK_LEFT,
                    self._ee_link_right_name,
                    self._ee_link_left_name,
                }
            ),
            prioritize_urdf_material=True,
            **self.URDF_MORPH_OPTIONS,
        )

        self.robot_entity = scene.add_entity(morph=morph, material=gs.materials.Rigid(coup_friction=self.COUP_FRICTION))

        # End-effector links - will be populated by init_dof_indices() after scene.build()
        # get_link() requires the scene to be built first
        self.ee_link_right = None
        self.ee_link_left = None

        # DOF indices - will be populated by init_dof_indices() after scene.build()
        # Initial values are fallback based on typical URDF order
        n = self.ARM_DOF
        self.n_physical_dofs = 2 * (n + 2)
        self.right_arm_dofs = list(range(0, n))
        self.right_gripper_dofs = list(range(n, n + 2))
        self.left_arm_dofs = list(range(n + 2, 2 * n + 2))
        self.left_gripper_dofs = list(range(2 * n + 2, 2 * n + 4))

        # All arm DOFs (for convenience)
        self.arm_dofs = self.right_arm_dofs + self.left_arm_dofs
        self.gripper_dofs = self.right_gripper_dofs + self.left_gripper_dofs
        self.all_dofs = list(range(self.n_physical_dofs))

        # Flag to track if DOF indices have been initialized from joint names
        self._dofs_initialized = False

        # Joint limits (replaced by the URDF's own in init_dof_indices)
        self.right_arm_lower = torch.tensor(list(self.ARM_LOWER), device=device)
        self.right_arm_upper = torch.tensor(list(self.ARM_UPPER), device=device)
        self.left_arm_lower = self.right_arm_lower.clone()
        self.left_arm_upper = self.right_arm_upper.clone()
        self.gripper_lower = torch.tensor([0.0, 0.0], device=device)
        self.gripper_upper = torch.tensor([self.FINGER_MAX, self.FINGER_MAX], device=device)

        # Reset positions in sequential order
        fingers = [self.RESET_FINGER * self.FINGER_SIGNS[0], self.RESET_FINGER * self.FINGER_SIGNS[1]]
        self.reset_joint_positions = torch.tensor(
            list(self.RESET_RIGHT_ARM) + fingers + list(self.RESET_LEFT_ARM) + fingers,
            device=device,
        ).repeat(n_envs, 1)

        # Control parameters for EE delta mode
        self.pos_scale = 0.05  # Position delta scale (meters)
        self.rot_scale = 0.1  # Rotation delta scale (radians)

        # Mapping from sequential position to DOF index
        # This will be updated by init_dof_indices() after scene.build()
        # Initial values assume DOF indices match sequential positions
        # NOTE: Keep on CPU for indexing operations (works with both CPU and CUDA tensors)
        self._seq_pos_to_dof = torch.arange(self.n_physical_dofs, dtype=torch.long)

        # IK solver q-vector layout, set up with the solver in _get_ik_solver()
        self._ik_q_dofs: np.ndarray | None = None
        self._ik_q_right_arm: np.ndarray | None = None
        self._ik_q_left_arm: np.ndarray | None = None

    def init_dof_indices(self) -> None:
        """Initialize DOF indices and EE links by querying the built robot entity.

        This should be called after scene.build() to get accurate DOF indices
        and EE link references.
        Falls back gracefully if joints can't be queried.
        """
        if self._dofs_initialized:
            return

        try:
            # Get end-effector links (requires scene to be built)
            if self.ee_link_right is None:
                self.ee_link_right = self.robot_entity.get_link(self._ee_link_right_name)
            if self.ee_link_left is None:
                self.ee_link_left = self.robot_entity.get_link(self._ee_link_left_name)

            def _dofs(joint_names: list[str]) -> list[int]:
                # dofs_idx_local returns a list; [0] for 1-DOF joints
                dofs = []
                for joint_name in joint_names:
                    joint = self.robot_entity.get_joint(joint_name)
                    if joint is not None:
                        dofs.append(joint.dofs_idx_local[0])
                return dofs

            right_arm_dofs = _dofs(self.RIGHT_ARM_JOINTS)
            right_gripper_dofs = _dofs(self.RIGHT_GRIPPER_JOINTS)
            left_arm_dofs = _dofs(self.LEFT_ARM_JOINTS)
            left_gripper_dofs = _dofs(self.LEFT_GRIPPER_JOINTS)

            # Only update if we got all expected joints
            if (
                len(right_arm_dofs) == self.ARM_DOF
                and len(right_gripper_dofs) == 2
                and len(left_arm_dofs) == self.ARM_DOF
                and len(left_gripper_dofs) == 2
            ):
                self.right_arm_dofs = right_arm_dofs
                self.right_gripper_dofs = right_gripper_dofs
                self.left_arm_dofs = left_arm_dofs
                self.left_gripper_dofs = left_gripper_dofs

                # Update convenience lists
                self.arm_dofs = self.right_arm_dofs + self.left_arm_dofs
                self.gripper_dofs = self.right_gripper_dofs + self.left_gripper_dofs
                self.all_dofs = (
                    self.right_arm_dofs + self.right_gripper_dofs + self.left_arm_dofs + self.left_gripper_dofs
                )

                # Build mapping from sequential position to DOF index
                # Sequential order: [R_arm, R_fingers, L_arm, L_fingers]
                # Keep on CPU for indexing operations (works with both CPU and CUDA tensors)
                self._seq_pos_to_dof = torch.tensor(self.all_dofs, dtype=torch.long)

                self._dofs_initialized = True

                # Read joint limits directly from the built URDF
                lower_all, upper_all = self.robot_entity.get_dofs_limit()
                self.right_arm_lower = lower_all[self.right_arm_dofs].to(self.device)
                self.right_arm_upper = upper_all[self.right_arm_dofs].to(self.device)
                self.left_arm_lower = lower_all[self.left_arm_dofs].to(self.device)
                self.left_arm_upper = upper_all[self.left_arm_dofs].to(self.device)
                self.gripper_lower = lower_all[self.right_gripper_dofs].to(self.device)
                self.gripper_upper = upper_all[self.right_gripper_dofs].to(self.device)

                # Set PD gains and force range (sequential order: right arm/gripper, left arm/gripper)
                kp_seq = np.array(self.KP + self.KP, dtype=float)
                kv_seq = np.array(self.KV + self.KV, dtype=float)
                fr_seq = np.array(self.FORCE_RANGE + self.FORCE_RANGE, dtype=float)
                dof_order = self._seq_pos_to_dof.cpu().numpy()
                kp_dof = np.empty(self.n_physical_dofs)
                kv_dof = np.empty(self.n_physical_dofs)
                fr_dof = np.empty(self.n_physical_dofs)
                kp_dof[dof_order] = kp_seq
                kv_dof[dof_order] = kv_seq
                fr_dof[dof_order] = fr_seq
                self.robot_entity.set_dofs_kp(kp_dof)
                self.robot_entity.set_dofs_kv(kv_dof)
                self.robot_entity.set_dofs_force_range(-fr_dof, fr_dof)

        except (AttributeError, RuntimeError):
            # Fall back to hard-coded indices if querying fails
            pass

    def _dof_to_sequential(self, dof_data: torch.Tensor) -> torch.Tensor:
        """Convert DOF-indexed data to sequential position order.

        Genesis returns joint data indexed by DOF. This reorders it to
        sequential position order [R_arm, R_fingers, L_arm, L_fingers].

        Args:
            dof_data: (n_envs, n_physical_dofs) data indexed by DOF

        Returns:
            (n_envs, n_physical_dofs) data in sequential position order
        """
        # _seq_pos_to_dof[seq_pos] = dof_idx
        # We want the reverse: for each sequential position, get the value at that DOF
        return dof_data[:, self._seq_pos_to_dof]

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Current arm joints and gripper widths.

        Returns:
            (right_arm (n_envs, ARM_DOF), right_width (n_envs, 1), left_arm (n_envs, ARM_DOF),
            left_width (n_envs, 1)); a width is the FINGER_SIGNS-weighted sum of its two finger positions.
        """
        seq = self._dof_to_sequential(self.robot_entity.get_dofs_position())
        n = self.ARM_DOF
        left = n + 2
        right_arm = seq[:, :n]
        s0, s1 = self.FINGER_SIGNS
        right_width = seq[:, n : n + 1] * s0 + seq[:, n + 1 : n + 2] * s1
        left_arm = seq[:, left : left + n]
        left_width = seq[:, left + n : left + n + 1] * s0 + seq[:, left + n + 1 : left + n + 2] * s1
        return right_arm, right_width, left_arm, left_width

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset robot state for given environment indices.

        Args:
            envs_idx: Tensor of environment indices to reset
        """
        n_reset = len(envs_idx)
        # Reset joint velocities first
        zero_vel = torch.zeros(n_reset, self.n_physical_dofs, device=self.device)

        # Get DOF indices list for set_dofs_position/velocity calls
        # reset_joint_positions is in sequential order, and _seq_pos_to_dof maps to DOF indices
        dof_indices = self._seq_pos_to_dof.tolist()

        # For single-env (non-parallelized) scenes, Genesis doesn't support envs_idx
        if self.n_envs == 1:
            self.robot_entity.set_dofs_velocity(
                zero_vel.squeeze(0),
                dofs_idx_local=dof_indices,
            )
            self.robot_entity.set_dofs_position(
                self.reset_joint_positions[envs_idx].squeeze(0),
                dofs_idx_local=dof_indices,
                zero_velocity=False,
            )
        else:
            self.robot_entity.set_dofs_velocity(
                zero_vel,
                dofs_idx_local=dof_indices,
                envs_idx=envs_idx,
            )
            self.robot_entity.set_dofs_position(
                self.reset_joint_positions[envs_idx],
                dofs_idx_local=dof_indices,
                envs_idx=envs_idx,
                zero_velocity=False,
            )
        self._arm_cmd = self._reset_arm_targets()

    def hold_reset_position(self) -> None:
        """Apply PD control targeting the reset joint positions (used during settle steps)."""
        dof_indices = self._seq_pos_to_dof.tolist()
        self.robot_entity.control_dofs_position(
            self.reset_joint_positions[0],
            dofs_idx_local=dof_indices,
        )
        self._arm_cmd = self._reset_arm_targets()

    def _reset_arm_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The reset pose's arm joints (right, left), each (n_envs, ARM_DOF)."""
        n = self.ARM_DOF
        return self.reset_joint_positions[:, :n].clone(), self.reset_joint_positions[:, n + 2 : 2 * n + 2].clone()

    def _limit_arm_targets(self, right: torch.Tensor, left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Arm joint targets moved at most max_joint_step per joint from the last commanded ones, and recorded.

        Every control mode sends its arm targets through here, so no joint target -- an IK solution included --
        can move faster than MAX_JOINT_SPEED. A target further away is approached over several actions.
        """
        # A non-finite target joint (a NaN IK solution, a NaN action) is replaced by the previous command. Without
        # this, NaN was stored as the last command and every later target, prev + clamp(...), stayed NaN: the arm
        # flailed and ignored all further actions for the rest of the episode (17_01 episode 2).
        if self._arm_cmd is None or not all(bool(torch.isfinite(c).all()) for c in self._arm_cmd):
            q = self.robot_entity.get_dofs_position()
            self._arm_cmd = (q[..., self.right_arm_dofs], q[..., self.left_arm_dofs])
        prev_right, prev_left = self._arm_cmd
        right = torch.where(torch.isfinite(right), right, prev_right)
        left = torch.where(torch.isfinite(left), left, prev_left)
        if self.max_joint_step is not None:
            step = self.max_joint_step
            right = prev_right + (right - prev_right).clamp(-step, step)
            left = prev_left + (left - prev_left).clamp(-step, step)
        self._arm_cmd = (right.detach().clone(), left.detach().clone())
        return right, left

    def _limit_physical_arm_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """`_limit_arm_targets` on a (n_envs, n_physical_dofs) vector in sequential order; fingers unchanged."""
        n = self.ARM_DOF
        right, left = self._limit_arm_targets(targets[:, :n], targets[:, n + 2 : 2 * n + 2])
        return torch.cat([right, targets[:, n : n + 2], left, targets[:, 2 * n + 2 :]], dim=-1)

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action based on current control mode.

        Args:
            action: Action tensor. Shape depends on control mode (n = ARM_DOF):
                - JOINT_ABS: (n_envs, JOINT_DIM) [R_arm(n), R_grip(1), L_arm(n), L_grip(1)]
                - JOINT_DELTA: (n_envs, JOINT_DIM) same format, added to current positions
                - EE_ABS: (n_envs, 14) [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
                - EE_ABS_ROT6D: (n_envs, 20) [R_pos(3), L_pos(3), R_rot6d(6), L_rot6d(6), R_grip(1), L_grip(1)]
                - EE_DELTA: (n_envs, 14) [R_delta_pos(3), R_delta_euler(3), R_grip(1),
                                          L_delta_pos(3), L_delta_euler(3), L_grip(1)]
        """
        if self.control_mode == "JOINT_ABS":
            self._apply_joint_pos(action)
        elif self.control_mode == "JOINT_DELTA":
            self._apply_joint_delta(action)
        elif self.control_mode == "EE_ABS":
            self._apply_ee_abs(action)
        elif self.control_mode == "EE_ABS_ROT6D":
            self._apply_ee_abs_rot6d(action)
        elif self.control_mode == "EE_DELTA":
            self._apply_ee_delta(action)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def _apply_joint_pos(self, action: torch.Tensor) -> None:
        """Apply direct joint position control.

        Args:
            action: (n_envs, JOINT_DIM) joint position targets [R_arm, R_grip, L_arm, L_grip].
                Gripper width is split evenly across both finger joints.
        """
        action_phys = self._limit_physical_arm_targets(self._expand_to_physical(action))

        # Reorder from sequential positions to DOF indices
        dof_indices = self._seq_pos_to_dof.tolist()
        self.robot_entity.control_dofs_position(
            action_phys,
            dofs_idx_local=dof_indices,
        )

    def _apply_joint_delta(self, action: torch.Tensor) -> None:
        """Apply delta joint position control.

        Args:
            action: (n_envs, JOINT_DIM) joint position deltas [R_arm, R_grip, L_arm, L_grip].
                Deltas are added to current positions.
        """
        delta_phys = self._expand_to_physical(action)

        # Get current joint positions in sequential order
        current_pos = self._dof_to_sequential(self.robot_entity.get_dofs_position())

        # Add delta to current positions
        target_pos = self._limit_physical_arm_targets(current_pos + delta_phys)

        # Reorder from sequential positions to DOF indices
        dof_indices = self._seq_pos_to_dof.tolist()
        self.robot_entity.control_dofs_position(
            target_pos,
            dofs_idx_local=dof_indices,
        )

    def _apply_ee_abs(self, action: torch.Tensor) -> None:
        """Apply absolute end-effector control with IK.

        Args:
            action: (n_envs, 14) [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
        """
        # Parse action
        right_pos = action[:, 0:3]
        left_pos = action[:, 3:6]
        right_aa = action[:, 6:9]
        left_aa = action[:, 9:12]
        right_grip = action[:, 12:13]
        left_grip = action[:, 13:14]

        # Convert axis-angle to quaternion
        right_quat = self._axis_angle_to_quat(right_aa)
        left_quat = self._axis_angle_to_quat(left_aa)

        self._apply_ee_pose_targets(right_pos, right_quat, left_pos, left_quat, right_grip, left_grip)

    def _apply_ee_abs_rot6d(self, action: torch.Tensor) -> None:
        """Apply absolute end-effector control with IK, using the 6D rotation representation.

        Args:
            action: (n_envs, 20) [R_pos(3), L_pos(3), R_rot6d(6), L_rot6d(6), R_grip(1), L_grip(1)]
        """
        # Parse action
        right_pos = action[:, 0:3]
        left_pos = action[:, 3:6]
        right_rot6d = action[:, 6:12]
        left_rot6d = action[:, 12:18]
        right_grip = action[:, 18:19]
        left_grip = action[:, 19:20]

        # Convert 6D rotation to quaternion (Gram-Schmidt, tolerates non-orthonormal policy output)
        right_quat = rot6d_to_quat(right_rot6d)
        left_quat = rot6d_to_quat(left_rot6d)

        self._apply_ee_pose_targets(right_pos, right_quat, left_pos, left_quat, right_grip, left_grip)

    def _apply_ee_pose_targets(
        self,
        right_pos: torch.Tensor,
        right_quat: torch.Tensor,
        left_pos: torch.Tensor,
        left_quat: torch.Tensor,
        right_grip: torch.Tensor,
        left_grip: torch.Tensor,
    ) -> None:
        """Solve bimanual IK for the given EE pose targets and command arms and grippers."""
        # Bimanual IK (solves both arms simultaneously)
        right_joints, left_joints = self.run_bimanual_ik(right_pos, right_quat, left_pos, left_quat)
        right_joints, left_joints = self._limit_arm_targets(right_joints, left_joints)

        # Apply arm joint targets
        self.robot_entity.control_dofs_position(right_joints, dofs_idx_local=self.right_arm_dofs)
        self.robot_entity.control_dofs_position(left_joints, dofs_idx_local=self.left_arm_dofs)

        # Apply grippers (width -> per-finger position)
        self._apply_gripper(right_grip, self.right_gripper_dofs)
        self._apply_gripper(left_grip, self.left_gripper_dofs)

    def _expand_to_physical(self, action: torch.Tensor) -> torch.Tensor:
        """Expand a JOINT_DIM vector to the physical layout by splitting each gripper width across its fingers.

        Args:
            action: (n_envs, JOINT_DIM) [R_arm(n), R_grip(1), L_arm(n), L_grip(1)]

        Returns:
            (n_envs, n_physical_dofs) [R_arm(n), R_fingers(2), L_arm(n), L_fingers(2)]
        """
        n = self.ARM_DOF
        right_arm = action[:, :n]
        right_gripper = action[:, n : n + 1]
        left_arm = action[:, n + 1 : 2 * n + 1]
        left_gripper = action[:, 2 * n + 1 : 2 * n + 2]

        # Each finger gets half the gripper width
        return torch.cat(
            [right_arm, self._finger_pair(right_gripper / 2), left_arm, self._finger_pair(left_gripper / 2)],
            dim=-1,
        )

    def _finger_pair(self, half_width: torch.Tensor) -> torch.Tensor:
        """(n_envs, 1) half width -> (n_envs, 2) finger joint positions, each opening in its FINGER_SIGNS direction."""
        s0, s1 = self.FINGER_SIGNS
        return torch.cat([half_width * s0, half_width * s1], dim=-1)

    @staticmethod
    def _axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to quaternion (wxyz)."""
        angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axis_angle / angle
        half = angle * 0.5
        return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)

    def _apply_ee_delta(self, action: torch.Tensor) -> None:
        """Apply end-effector delta control with IK for both arms.

        Args:
            action: (n_envs, 14) [right_delta_pos(3), right_delta_euler(3), right_gripper(1),
                                  left_delta_pos(3), left_delta_euler(3), left_gripper(1)]
        """
        # Parse action for right arm
        right_delta_pos = action[:, 0:3] * self.pos_scale
        right_delta_euler = action[:, 3:6] * self.rot_scale
        right_gripper = action[:, 6:7]

        # Parse action for left arm
        left_delta_pos = action[:, 7:10] * self.pos_scale
        left_delta_euler = action[:, 10:13] * self.rot_scale
        left_gripper = action[:, 13:14]

        # Get current EE poses via FK on the IK solver's model (same frames targeted by IK),
        # rather than reading a nearby link from Genesis and carrying its offset error
        solver = self._get_ik_solver()
        solver.update_configuration(self._ik_current_q())

        right_tip_pos_np, right_tip_rot_np = solver.forward_kinematics(self._ee_link_right_name)
        left_tip_pos_np, left_tip_rot_np = solver.forward_kinematics(self._ee_link_left_name)

        right_current_pos = torch.from_numpy(right_tip_pos_np).float().to(self.device).unsqueeze(0)
        right_current_quat = self._rot_mat_to_quat(right_tip_rot_np).to(self.device).unsqueeze(0)
        left_current_pos = torch.from_numpy(left_tip_pos_np).float().to(self.device).unsqueeze(0)
        left_current_quat = self._rot_mat_to_quat(left_tip_rot_np).to(self.device).unsqueeze(0)

        # Compute targets for right arm
        right_target_pos = right_current_pos + right_delta_pos
        right_delta_quat = self._euler_to_quat(right_delta_euler)
        right_target_quat = quat_mul(right_delta_quat, right_current_quat)

        # Compute targets for left arm
        left_target_pos = left_current_pos + left_delta_pos
        left_delta_quat = self._euler_to_quat(left_delta_euler)
        left_target_quat = quat_mul(left_delta_quat, left_current_quat)

        # Compute joint targets with IK
        right_joint_targets = self._simple_ik(
            right_target_pos,
            right_target_quat,
            self.ee_link_right,
            self.right_arm_dofs,
            self.right_arm_lower,
            self.right_arm_upper,
        )
        left_joint_targets = self._simple_ik(
            left_target_pos,
            left_target_quat,
            self.ee_link_left,
            self.left_arm_dofs,
            self.left_arm_lower,
            self.left_arm_upper,
        )

        right_joint_targets, left_joint_targets = self._limit_arm_targets(right_joint_targets, left_joint_targets)

        # Apply arm joint targets
        self.robot_entity.control_dofs_position(
            right_joint_targets,
            dofs_idx_local=self.right_arm_dofs,
        )
        self.robot_entity.control_dofs_position(
            left_joint_targets,
            dofs_idx_local=self.left_arm_dofs,
        )

        # Apply grippers
        self._apply_gripper(right_gripper, self.right_gripper_dofs)
        self._apply_gripper(left_gripper, self.left_gripper_dofs)

    def _apply_gripper(self, gripper_action: torch.Tensor, gripper_dofs: list[int]) -> None:
        """Apply gripper control.

        Args:
            gripper_action: (n_envs, 1) gripper total width in meters [0, 2 * FINGER_MAX]
            gripper_dofs: List of gripper DOF indices
        """
        # Total width -> per-finger position (each finger gets half)
        gripper_width = (gripper_action / 2.0).clamp(min=0.0, max=self.FINGER_MAX)
        gripper_pos = self._finger_pair(gripper_width)
        self.robot_entity.control_dofs_position(
            gripper_pos,
            dofs_idx_local=gripper_dofs,
        )

    def _euler_to_quat(self, euler: torch.Tensor) -> torch.Tensor:
        """Convert euler angles (xyz convention) to quaternion (wxyz).

        Args:
            euler: (n_envs, 3) roll, pitch, yaw in radians

        Returns:
            (n_envs, 4) quaternion in wxyz format
        """
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    def _init_ik_layout(self) -> None:
        """Map the IK solver's q vector (the URDF's joint order) onto Genesis DOFs, by joint name."""
        import pinocchio as pin

        model = pin.buildModelFromUrdf(self._urdf_path)
        q_names = list(model.names)[1:]  # drop "universe"
        joint_names = (
            self.RIGHT_ARM_JOINTS + self.RIGHT_GRIPPER_JOINTS + self.LEFT_ARM_JOINTS + self.LEFT_GRIPPER_JOINTS
        )
        if model.nq != len(q_names) or sorted(q_names) != sorted(joint_names):
            raise ValueError(
                f"{type(self).__name__}: IK model joints {q_names} (nq={model.nq}) do not match the robot's "
                f"1-DOF joints {joint_names}"
            )
        dof_by_name = dict(zip(joint_names, self.all_dofs, strict=True))
        q_index = {name: i for i, name in enumerate(q_names)}
        self._ik_q_dofs = np.array([dof_by_name[name] for name in q_names])
        self._ik_q_right_arm = np.array([q_index[name] for name in self.RIGHT_ARM_JOINTS])
        self._ik_q_left_arm = np.array([q_index[name] for name in self.LEFT_ARM_JOINTS])

    def _ik_current_q(self) -> np.ndarray:
        """Current joint positions as the IK solver's q vector."""
        all_joints = self.robot_entity.get_dofs_position()[0].cpu().numpy()
        return all_joints[self._ik_q_dofs]

    def _get_ik_solver(self):
        """Get or create the IK solver (lazy initialization)."""
        if self._ik_solver is None:
            from gs_gym.solvers.kinematics import KinematicsSolver

            self._init_ik_layout()
            self._ik_solver = KinematicsSolver(
                self._urdf_path,
                initial_q=self._ik_current_q(),
                position_cost=10.0,
                rotation_cost=1.0,
            )

        return self._ik_solver

    def _quat_to_rotation_matrix(self, quat: torch.Tensor):
        """Convert quaternion (wxyz) to rotation matrix."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Rotation matrix from quaternion
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return rot

    @staticmethod
    def _rot_mat_to_quat(rot) -> torch.Tensor:
        """Convert 3x3 rotation matrix to quaternion (wxyz) using Shepperd's method."""
        trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot[2, 1] - rot[1, 2]) * s
            y = (rot[0, 2] - rot[2, 0]) * s
            z = (rot[1, 0] - rot[0, 1]) * s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
        return torch.tensor([w, x, y, z], dtype=torch.float32)

    def _simple_ik(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        ee_link,
        arm_dofs: list[int],
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IK using Pink/Pinocchio solver.

        Args:
            target_pos: (n_envs, 3) target end-effector position
            target_quat: (n_envs, 4) target end-effector quaternion (wxyz)
            ee_link: End-effector link object (used to determine which arm)
            arm_dofs: List of arm DOF indices
            joint_lower: Lower joint limits
            joint_upper: Upper joint limits

        Returns:
            (n_envs, ARM_DOF) joint position targets
        """
        if self.n_envs > 1:
            raise ValueError("Pink IK solver only supports n_envs=1")

        solver = self._get_ik_solver()

        # Get current joint configuration and update solver
        solver.update_configuration(self._ik_current_q())

        # Determine which arm based on ee_link
        is_right_arm = ee_link == self.ee_link_right
        frame_name = self._ee_link_right_name if is_right_arm else self._ee_link_left_name

        # Convert target to numpy
        target_pos_np = target_pos[0].cpu().numpy()
        target_rot_np = self._quat_to_rotation_matrix(target_quat[0].cpu())

        result_q = solver.inverse_kinematics(
            frame_name=frame_name,
            target_position=target_pos_np,
            target_rotation=target_rot_np,
            max_solver_iters=200,
        )

        target_joints = result_q[self._ik_q_right_arm] if is_right_arm else result_q[self._ik_q_left_arm]
        target_joints = torch.from_numpy(target_joints).float().to(self.device).unsqueeze(0)

        # Clamp to joint limits
        target_joints = torch.clamp(
            target_joints,
            joint_lower.unsqueeze(0),
            joint_upper.unsqueeze(0),
        )

        return target_joints

    def run_bimanual_ik(
        self,
        right_target_pos: torch.Tensor,
        right_target_quat: torch.Tensor,
        left_target_pos: torch.Tensor,
        left_target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IK for both arms simultaneously.

        Args:
            right_target_pos: (n_envs, 3) right EE target position
            right_target_quat: (n_envs, 4) right EE target quaternion (wxyz)
            left_target_pos: (n_envs, 3) left EE target position
            left_target_quat: (n_envs, 4) left EE target quaternion (wxyz)

        Returns:
            Tuple of (right_joints, left_joints), each (n_envs, ARM_DOF)
        """
        if self.n_envs > 1:
            raise ValueError("Pink IK solver only supports n_envs=1")

        solver = self._get_ik_solver()

        # Get current joint configuration and update solver
        solver.update_configuration(self._ik_current_q())

        # Convert targets to numpy
        right_pos_np = right_target_pos[0].cpu().numpy()
        right_rot_np = self._quat_to_rotation_matrix(right_target_quat[0].cpu())
        left_pos_np = left_target_pos[0].cpu().numpy()
        left_rot_np = self._quat_to_rotation_matrix(left_target_quat[0].cpu())

        try:
            # Run bimanual IK (both arms at once)
            result_q = solver.inverse_kinematics(
                frame_name=[self._ee_link_right_name, self._ee_link_left_name],
                target_position=[right_pos_np, left_pos_np],
                target_rotation=[right_rot_np, left_rot_np],
                max_solver_iters=200,
            )
        except Exception:
            raise ValueError("IK failed") from None

        left_joints = torch.from_numpy(result_q[self._ik_q_left_arm]).float().to(self.device).unsqueeze(0)
        right_joints = torch.from_numpy(result_q[self._ik_q_right_arm]).float().to(self.device).unsqueeze(0)

        # Clamp to joint limits
        right_joints = torch.clamp(
            right_joints,
            self.right_arm_lower.unsqueeze(0),
            self.right_arm_upper.unsqueeze(0),
        )
        left_joints = torch.clamp(
            left_joints,
            self.left_arm_lower.unsqueeze(0),
            self.left_arm_upper.unsqueeze(0),
        )

        return right_joints, left_joints

    def get_proprioception(self) -> torch.Tensor:
        """Get proprioceptive state.

        Returns:
            (n_envs, 2 * n_physical_dofs + 14) tensor containing:
            - joint positions (n_physical_dofs) in sequential order [R_arm, R_fingers, L_arm, L_fingers]
            - joint velocities (n_physical_dofs) in sequential order
            - right EE position (3)
            - right EE orientation quaternion (4)
            - left EE position (3)
            - left EE orientation quaternion (4)
        """
        dofs_pos = self._dof_to_sequential(self.robot_entity.get_dofs_position())
        dofs_vel = self._dof_to_sequential(self.robot_entity.get_dofs_velocity())

        right_ee_pos = self.ee_link_right.get_pos(relative=False)
        right_ee_quat = self.ee_link_right.get_quat(relative=False)
        left_ee_pos = self.ee_link_left.get_pos(relative=False)
        left_ee_quat = self.ee_link_left.get_quat(relative=False)

        return torch.cat(
            [dofs_pos, dofs_vel, right_ee_pos, right_ee_quat, left_ee_pos, left_ee_quat],
            dim=-1,
        )

    @property
    def joint_positions(self) -> torch.Tensor:
        """Get current joint positions in sequential order (n_physical_dofs)."""
        return self._dof_to_sequential(self.robot_entity.get_dofs_position())

    @property
    def right_ee_pose(self) -> torch.Tensor:
        """Right end-effector pose (position + quaternion).

        Returns:
            (n_envs, 7) tensor with [x, y, z, qw, qx, qy, qz]
        """
        pos = self.ee_link_right.get_pos(relative=False)
        quat = self.ee_link_right.get_quat(relative=False)
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_ee_pose(self) -> torch.Tensor:
        """Left end-effector pose (position + quaternion).

        Returns:
            (n_envs, 7) tensor with [x, y, z, qw, qx, qy, qz]
        """
        pos = self.ee_link_left.get_pos(relative=False)
        quat = self.ee_link_left.get_quat(relative=False)
        return torch.cat([pos, quat], dim=-1)

    @property
    def action_space(self):
        """Action space of the robot."""
        if self.control_mode in ("JOINT_ABS", "JOINT_DELTA"):
            # [R_arm, R_grip(1), L_arm, L_grip(1)]
            # Gripper is total width (0 to 2 * FINGER_MAX); each finger gets half internally
            gripper_low = np.array([0.0])
            gripper_high = np.array([2 * self.FINGER_MAX])
            low = np.concatenate(
                [self.right_arm_lower.cpu().numpy(), gripper_low, self.left_arm_lower.cpu().numpy(), gripper_low]
            )
            high = np.concatenate(
                [self.right_arm_upper.cpu().numpy(), gripper_high, self.left_arm_upper.cpu().numpy(), gripper_high]
            )
            return spaces.Box(low=low, high=high, shape=(self.JOINT_DIM,), dtype=np.float32)
        else:  # EE_DELTA, EE_ABS or EE_ABS_ROT6D
            # 14-DOF: [right: 3 pos + 3 rot + 1 gripper, left: same]; 20-DOF when rot is 6D
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        """Dimension of action space."""
        if self.control_mode == "EE_ABS_ROT6D":
            return 20  # 10+10 (pos+rot6d+gripper per arm)
        if self.control_mode in ("EE_DELTA", "EE_ABS"):
            return 14  # 7+7 (pos+rot+gripper per arm)
        return self.JOINT_DIM  # arm + single gripper width per arm
