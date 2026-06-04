"""Bimanual Marvin robot implementation for Genesis environments.

A dual-arm robot with 7 DOF per arm and 2-finger gripper per arm.
"""

from typing import ClassVar

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces

from gs_gym.common.utils.math_utils import quat_mul
from gs_gym.robots.registry import register_robot


@register_robot("bimanual_marvin")
class BimanualMarvinRobot:
    """Bimanual Marvin robot implementation for Genesis environments.

    A dual-arm robot with:
    - 2 arms: 7 DOF each (Joint1-7_R and Joint1-7_L)
    - 2 grippers: 2 prismatic joints each (Joint8-9_R and Joint8-9_L)
    - Total: 18 physical DOFs (but user-facing API uses 16D with single gripper width)

    Supports control modes:
    - JOINT_ABS: Absolute joint position control (16D: 7 arm + 1 gripper per arm)
    - JOINT_DELTA: Delta joint position control (16D, added to current)
    - EE_ABS: Absolute end-effector control with IK (14D)
    - EE_DELTA: Delta end-effector control with IK (14D)
    """

    #: Available control modes for this robot
    CONTROL_MODES: ClassVar[list[str]] = ["JOINT_ABS", "JOINT_DELTA", "EE_ABS", "EE_DELTA"]

    #: Joint names in order (right arm, then left arm)
    #: This matches the order Genesis assigns DOF indices from the URDF
    RIGHT_ARM_JOINTS: ClassVar[list[str]] = [
        "Joint1_R",
        "Joint2_R",
        "Joint3_R",
        "Joint4_R",
        "Joint5_R",
        "Joint6_R",
        "Joint7_R",
    ]
    RIGHT_GRIPPER_JOINTS: ClassVar[list[str]] = ["Joint8_R", "Joint9_R"]

    LEFT_ARM_JOINTS: ClassVar[list[str]] = [
        "Joint1_L",
        "Joint2_L",
        "Joint3_L",
        "Joint4_L",
        "Joint5_L",
        "Joint6_L",
        "Joint7_L",
    ]
    LEFT_GRIPPER_JOINTS: ClassVar[list[str]] = ["Joint8_L", "Joint9_L"]

    #: End-effector link names (actual gripper tip frames from URDF)
    EE_LINK_RIGHT: ClassVar[str] = "Gripper_Tip_R"
    EE_LINK_LEFT: ClassVar[str] = "Gripper_Tip_L"

    #: Gripper base link names (used for wrist camera attachment)
    GRIPPER_BASE_LINK_RIGHT: ClassVar[str] = "Pika_Gripper_Base_R"
    GRIPPER_BASE_LINK_LEFT: ClassVar[str] = "Pika_Gripper_Base_L"

    #: PD gains and force range
    # : Per arm: 7 arm joints + 2 gripper joints
    # KP: ClassVar[tuple[float, ...]] = (720, 720, 720, 360, 360, 360, 360, 200, 200)
    KP: ClassVar[tuple[float, ...]] = (7200, 7200, 7200, 3600, 3600, 3600, 3600, 200, 200)
    KV: ClassVar[tuple[float, ...]] = (600, 600, 600, 400, 200, 200, 200, 20, 20)
    # FORCE_RANGE: ClassVar[tuple[float, ...]] = (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000)
    FORCE_RANGE: ClassVar[tuple[float, ...]] = (2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000)
    # def _rad_to_deg(rad: tuple[float, ...]) -> tuple[float, ...]:
    #     return tuple(np.rad2deg(rad))

    # KP: ClassVar[tuple[float, ...]] = _rad_to_deg((5, 5, 5, 2.5, 2.5, 2.5, 2.5)) + (200.0, 200.0)
    # KV: ClassVar[tuple[float, ...]] = _rad_to_deg((0.5, 0.5, 0.5, 0.4, 0.2, 0.2, 0.2)) + (20.0, 20.0)
    # FORCE_RANGE: ClassVar[tuple[float, ...]] = (90, 90, 45, 45, 24, 24, 24, 100, 100)

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
    ) -> None:
        """Initialize the Bimanual Marvin robot.

        Args:
            n_envs: Number of parallel environments
            scene: Genesis scene object
            args: Robot configuration arguments
            device: Torch device
            control_mode: Control mode (JOINT_ABS, JOINT_DELTA, EE_ABS, EE_DELTA)
            ee_link_right: Override EE link name for right arm (default: class constant)
            ee_link_left: Override EE link name for left arm (default: class constant)
            urdf_path: Override URDF path (default: HuggingFace gs-gym-assets)
        """
        self.n_envs = n_envs
        self.scene = scene
        self.args = args
        self.device = device

        # Allow overriding EE link names (e.g. benchgen uses Gripper_Tip_R/L)
        self._ee_link_right_name = ee_link_right or self.EE_LINK_RIGHT
        self._ee_link_left_name = ee_link_left or self.EE_LINK_LEFT

        # Validate control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control mode '{control_mode}'. Available: {self.CONTROL_MODES}")
        self.control_mode = control_mode

        # Get URDF path from override, HuggingFace, or vendored assets
        if urdf_path is None:
            from gs_gym.common.utils.asset_utils import get_bimanual_marvin_urdf_path

            urdf_path = get_bimanual_marvin_urdf_path()
        self._urdf_path = urdf_path

        # Initialize IK solver (lazy - created on first use)
        self._ik_solver = None

        # Create robot entity from URDF
        # Keep gripper base links so wrist cameras can attach to them
        morph = gs.morphs.URDF(
            file=urdf_path,
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
        )

        self.robot_entity = scene.add_entity(morph=morph)

        # End-effector links - will be populated by init_dof_indices() after scene.build()
        # get_link() requires the scene to be built first
        self.ee_link_right = None
        self.ee_link_left = None

        # DOF indices - will be populated by init_dof_indices() after scene.build()
        # Initial values are fallback based on typical URDF order
        self.right_arm_dofs = list(range(0, 7))
        self.right_gripper_dofs = list(range(7, 9))
        self.left_arm_dofs = list(range(9, 16))
        self.left_gripper_dofs = list(range(16, 18))

        # All arm DOFs (for convenience)
        self.arm_dofs = self.right_arm_dofs + self.left_arm_dofs
        self.gripper_dofs = self.right_gripper_dofs + self.left_gripper_dofs
        self.all_dofs = list(range(18))

        # Flag to track if DOF indices have been initialized from joint names
        self._dofs_initialized = False

        # Joint limits from URDF
        # Right arm: Joint1-7_R limits
        self.right_arm_lower = torch.tensor(
            [-3.1067, -1.8, -3.1067, -2.5307, -3.1067, -1.047, -1.047],
            device=device,
        )
        self.right_arm_upper = torch.tensor(
            [3.1067, 2.0944, 3.1067, -0.08727, 3.1067, 1.047, 1.047],
            device=device,
        )

        # Left arm: Same limits as right
        self.left_arm_lower = self.right_arm_lower.clone()
        self.left_arm_upper = self.right_arm_upper.clone()

        # Gripper limits: 0 to 0.05 (prismatic)
        self.gripper_lower = torch.tensor([0.0, 0.0], device=device)
        self.gripper_upper = torch.tensor([0.05, 0.05], device=device)

        # Initialize reset positions
        # fmt: off
        right_arm_init = [-1.5708, -1.3089969, 1.5708, -1.5708, -1.3089969, 0.0, -0.349]
        right_gripper_init = [0.045, 0.045]
        left_arm_init = [1.5708, -1.3089969, -1.5708, -1.5708, 1.3089969, 0.0, 0.349]
        left_gripper_init = [0.045, 0.045]
        # fmt: on

        self.reset_joint_positions = torch.tensor(
            right_arm_init + right_gripper_init + left_arm_init + left_gripper_init,
            device=device,
        ).repeat(n_envs, 1)

        # Control parameters for EE delta mode
        self.pos_scale = 0.05  # Position delta scale (meters)
        self.rot_scale = 0.1  # Rotation delta scale (radians)

        # Mapping from sequential position (0-17) to DOF index
        # This will be updated by init_dof_indices() after scene.build()
        # Initial values assume DOF indices match sequential positions
        # NOTE: Keep on CPU for indexing operations (works with both CPU and CUDA tensors)
        self._seq_pos_to_dof = torch.arange(18, dtype=torch.long)

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
            # Query DOF indices for right arm joints
            # Note: dofs_idx_local returns a list, we take [0] for 1-DOF joints
            right_arm_dofs = []
            for joint_name in self.RIGHT_ARM_JOINTS:
                joint = self.robot_entity.get_joint(joint_name)
                if joint is not None:
                    right_arm_dofs.append(joint.dofs_idx_local[0])

            # Query DOF indices for right gripper joints
            right_gripper_dofs = []
            for joint_name in self.RIGHT_GRIPPER_JOINTS:
                joint = self.robot_entity.get_joint(joint_name)
                if joint is not None:
                    right_gripper_dofs.append(joint.dofs_idx_local[0])

            # Query DOF indices for left arm joints
            left_arm_dofs = []
            for joint_name in self.LEFT_ARM_JOINTS:
                joint = self.robot_entity.get_joint(joint_name)
                if joint is not None:
                    left_arm_dofs.append(joint.dofs_idx_local[0])

            # Query DOF indices for left gripper joints
            left_gripper_dofs = []
            for joint_name in self.LEFT_GRIPPER_JOINTS:
                joint = self.robot_entity.get_joint(joint_name)
                if joint is not None:
                    left_gripper_dofs.append(joint.dofs_idx_local[0])

            # Only update if we got all expected joints
            if (
                len(right_arm_dofs) == 7
                and len(right_gripper_dofs) == 2
                and len(left_arm_dofs) == 7
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
                # Sequential order: [R1-R7, RG1, RG2, L1-L7, LG1, LG2]
                seq_pos_to_dof = (
                    self.right_arm_dofs  # positions 0-6 -> right arm DOFs
                    + self.right_gripper_dofs  # positions 7-8 -> right gripper DOFs
                    + self.left_arm_dofs  # positions 9-15 -> left arm DOFs
                    + self.left_gripper_dofs  # positions 16-17 -> left gripper DOFs
                )
                # Keep on CPU for indexing operations (works with both CPU and CUDA tensors)
                self._seq_pos_to_dof = torch.tensor(seq_pos_to_dof, dtype=torch.long)

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
                kp_dof = np.empty(18)
                kv_dof = np.empty(18)
                fr_dof = np.empty(18)
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
        sequential position order [R1-R7, RG1, RG2, L1-L7, LG1, LG2].

        Args:
            dof_data: (n_envs, 18) data indexed by DOF

        Returns:
            (n_envs, 18) data in sequential position order
        """
        # _seq_pos_to_dof[seq_pos] = dof_idx
        # We want the reverse: for each sequential position, get the value at that DOF
        return dof_data[:, self._seq_pos_to_dof]

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset robot state for given environment indices.

        Args:
            envs_idx: Tensor of environment indices to reset
        """
        n_reset = len(envs_idx)
        # Reset joint velocities first
        zero_vel = torch.zeros(n_reset, 18, device=self.device)

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

    def hold_reset_position(self) -> None:
        """Apply PD control targeting the reset joint positions (used during settle steps)."""
        dof_indices = self._seq_pos_to_dof.tolist()
        self.robot_entity.control_dofs_position(
            self.reset_joint_positions[0],
            dofs_idx_local=dof_indices,
        )

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action based on current control mode.

        Args:
            action: Action tensor. Shape depends on control mode:
                - JOINT: (n_envs, 16) [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
                - JOINT_DELTA: (n_envs, 16) same format, added to current positions
                - EE_ABS: (n_envs, 14) [R_pos(3), L_pos(3), R_axis_angle(3), L_axis_angle(3), R_grip(1), L_grip(1)]
                - EE_DELTA: (n_envs, 14) [R_delta_pos(3), R_delta_euler(3), R_grip(1),
                                          L_delta_pos(3), L_delta_euler(3), L_grip(1)]
        """
        if self.control_mode == "JOINT_ABS":
            self._apply_joint_pos(action)
        elif self.control_mode == "JOINT_DELTA":
            self._apply_joint_delta(action)
        elif self.control_mode == "EE_ABS":
            self._apply_ee_abs(action)
        elif self.control_mode == "EE_DELTA":
            self._apply_ee_delta(action)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def _apply_joint_pos(self, action: torch.Tensor) -> None:
        """Apply direct joint position control.

        Args:
            action: (n_envs, 16) joint position targets
                [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
                Gripper width is expanded to both finger joints internally.
        """
        # Expand 16D to 18D (duplicate gripper width to both fingers)
        action_18d = self._expand_16d_to_18d(action)

        # Reorder from sequential positions to DOF indices
        dof_indices = self._seq_pos_to_dof.tolist()
        self.robot_entity.control_dofs_position(
            action_18d,
            dofs_idx_local=dof_indices,
        )

    def _apply_joint_delta(self, action: torch.Tensor) -> None:
        """Apply delta joint position control.

        Args:
            action: (n_envs, 16) joint position deltas
                [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
                Deltas are added to current positions.
        """
        # Expand 16D delta to 18D
        delta_18d = self._expand_16d_to_18d(action)

        # Get current joint positions in sequential order (18D)
        current_pos = self._get_joint_positions_18d()

        # Add delta to current positions
        target_pos = current_pos + delta_18d

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

        # Bimanual IK (solves both arms simultaneously)
        right_joints, left_joints = self.run_bimanual_ik(right_pos, right_quat, left_pos, left_quat)

        # Apply arm joint targets
        self.robot_entity.control_dofs_position(right_joints, dofs_idx_local=self.right_arm_dofs)
        self.robot_entity.control_dofs_position(left_joints, dofs_idx_local=self.left_arm_dofs)

        # Apply gripper (width -> per-finger position)
        right_finger = (right_grip / 2.0).clamp(0.0, 0.05)
        left_finger = (left_grip / 2.0).clamp(0.0, 0.05)
        self.robot_entity.control_dofs_position(
            torch.cat([right_finger, right_finger], dim=-1), dofs_idx_local=self.right_gripper_dofs
        )
        self.robot_entity.control_dofs_position(
            torch.cat([left_finger, left_finger], dim=-1), dofs_idx_local=self.left_gripper_dofs
        )

    def _expand_16d_to_18d(self, action: torch.Tensor) -> torch.Tensor:
        """Expand 16D action to 18D by duplicating gripper width to both fingers.

        Args:
            action: (n_envs, 16) [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]

        Returns:
            (n_envs, 18) [R_arm(7), R_grip(2), L_arm(7), L_grip(2)]
        """
        right_arm = action[:, :7]
        right_gripper = action[:, 7:8]
        left_arm = action[:, 8:15]
        left_gripper = action[:, 15:16]

        # Each finger gets half the gripper width
        right_finger = right_gripper / 2
        left_finger = left_gripper / 2

        return torch.cat(
            [
                right_arm,
                right_finger,
                right_finger,
                left_arm,
                left_finger,
                left_finger,
            ],
            dim=-1,
        )

    def _get_joint_positions_18d(self) -> torch.Tensor:
        """Get current joint positions in 18D sequential order."""
        dofs_pos_raw = self.robot_entity.get_dofs_position()
        return self._dof_to_sequential(dofs_pos_raw)

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
        import numpy as np

        # Parse action for right arm
        right_delta_pos = action[:, 0:3] * self.pos_scale
        right_delta_euler = action[:, 3:6] * self.rot_scale
        right_gripper = action[:, 6:7]

        # Parse action for left arm
        left_delta_pos = action[:, 7:10] * self.pos_scale
        left_delta_euler = action[:, 10:13] * self.rot_scale
        left_gripper = action[:, 13:14]

        # Get current Gripper_Tip poses via FK (same frames targeted by IK)
        # This avoids the Link7_R vs Gripper_Tip_R offset error
        solver = self._get_ik_solver()
        all_joints = self.robot_entity.get_dofs_position()[0].cpu().numpy()
        right_arm = all_joints[self.right_arm_dofs]
        right_gripper_joints = all_joints[self.right_gripper_dofs]
        left_arm = all_joints[self.left_arm_dofs]
        left_gripper_joints = all_joints[self.left_gripper_dofs]
        current_q = np.concatenate([left_arm, left_gripper_joints, right_arm, right_gripper_joints])
        solver.update_configuration(current_q)

        right_tip_pos_np, right_tip_rot_np = solver.forward_kinematics("Gripper_Tip_R")
        left_tip_pos_np, left_tip_rot_np = solver.forward_kinematics("Gripper_Tip_L")

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
            gripper_action: (n_envs, 1) gripper total width in meters [0, 0.1]
            gripper_dofs: List of gripper DOF indices
        """
        # Total width -> per-finger position (each finger gets half)
        gripper_width = (gripper_action / 2.0).clamp(min=0.0, max=0.05)
        gripper_pos = torch.cat([gripper_width, gripper_width], dim=-1)
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

    def _get_ik_solver(self):
        """Get or create the IK solver (lazy initialization)."""
        if self._ik_solver is None:
            from gs_gym.solvers.kinematics import KinematicsSolver

            # Get current joint positions for initial configuration
            # Pink expects: [left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]
            all_joints = self.robot_entity.get_dofs_position()[0].cpu().numpy()

            # Convert from Genesis order (right-first) to Pink order (left-first)
            right_arm = all_joints[self.right_arm_dofs]
            right_gripper = all_joints[self.right_gripper_dofs]
            left_arm = all_joints[self.left_arm_dofs]
            left_gripper = all_joints[self.left_gripper_dofs]

            import numpy as np

            initial_q = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])

            self._ik_solver = KinematicsSolver(
                self._urdf_path,
                initial_q=initial_q,
                position_cost=10.0,
                rotation_cost=1.0,
            )

        return self._ik_solver

    def _quat_to_rotation_matrix(self, quat: torch.Tensor):
        """Convert quaternion (wxyz) to rotation matrix."""
        import numpy as np

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
        import numpy as np

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
            (n_envs, 7) joint position targets
        """
        import numpy as np

        if self.n_envs > 1:
            raise ValueError("Pink IK solver only supports n_envs=1")

        solver = self._get_ik_solver()

        # Get current joint configuration and update solver
        all_joints = self.robot_entity.get_dofs_position()[0].cpu().numpy()
        right_arm = all_joints[self.right_arm_dofs]
        right_gripper = all_joints[self.right_gripper_dofs]
        left_arm = all_joints[self.left_arm_dofs]
        left_gripper = all_joints[self.left_gripper_dofs]

        current_q = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
        solver.update_configuration(current_q)

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

        # Extract the arm joints from result (Pink order: left first)
        # Pink qpos: [left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]
        target_joints = result_q[9:16] if is_right_arm else result_q[0:7]
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
            Tuple of (right_joints, left_joints), each (n_envs, 7)
        """
        if self.n_envs > 1:
            raise ValueError("Pink IK solver only supports n_envs=1")

        solver = self._get_ik_solver()

        # Get current joint configuration and update solver
        # Pink uses LEFT-first order: [left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]
        all_joints = self.robot_entity.get_dofs_position()[0].cpu().numpy()
        right_arm = all_joints[self.right_arm_dofs]
        right_gripper = all_joints[self.right_gripper_dofs]
        left_arm = all_joints[self.left_arm_dofs]
        left_gripper = all_joints[self.left_gripper_dofs]

        current_q = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
        solver.update_configuration(current_q)

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

            # Extract arm joints from result (Pink order: left first)
            # Pink qpos: [left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]
            left_joints = torch.from_numpy(result_q[0:7]).float().to(self.device).unsqueeze(0)
            right_joints = torch.from_numpy(result_q[9:16]).float().to(self.device).unsqueeze(0)

        except Exception:
            raise ValueError("IK failed") from None
            # Fallback: return current joints if IK fails
            right_joints = torch.tensor(all_joints[self.right_arm_dofs], device=self.device).unsqueeze(0)
            left_joints = torch.tensor(all_joints[self.left_arm_dofs], device=self.device).unsqueeze(0)

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
            (n_envs, 50) tensor containing:
            - joint positions (18) in sequential order [R1-R7, RG1, RG2, L1-L7, LG1, LG2]
            - joint velocities (18) in sequential order
            - right EE position (3)
            - right EE orientation quaternion (4)
            - left EE position (3)
            - left EE orientation quaternion (4)
        """
        # Get DOF-indexed data from Genesis
        dofs_pos_raw = self.robot_entity.get_dofs_position()
        dofs_vel_raw = self.robot_entity.get_dofs_velocity()

        # Convert from DOF indices to sequential position order
        dofs_pos = self._dof_to_sequential(dofs_pos_raw)
        dofs_vel = self._dof_to_sequential(dofs_vel_raw)

        right_ee_pos = self.ee_link_right.get_pos()
        right_ee_quat = self.ee_link_right.get_quat()
        left_ee_pos = self.ee_link_left.get_pos()
        left_ee_quat = self.ee_link_left.get_quat()

        return torch.cat(
            [
                dofs_pos,  # All 18 joint positions
                dofs_vel,  # All 18 joint velocities
                right_ee_pos,  # Right EE position (3)
                right_ee_quat,  # Right EE quaternion (4)
                left_ee_pos,  # Left EE position (3)
                left_ee_quat,  # Left EE quaternion (4)
            ],
            dim=-1,
        )

    @property
    def joint_positions(self) -> torch.Tensor:
        """Get current joint positions in sequential order (18D)."""
        dofs_pos_raw = self.robot_entity.get_dofs_position()
        return self._dof_to_sequential(dofs_pos_raw)

    @property
    def right_ee_pose(self) -> torch.Tensor:
        """Right end-effector pose (position + quaternion).

        Returns:
            (n_envs, 7) tensor with [x, y, z, qw, qx, qy, qz]
        """
        pos = self.ee_link_right.get_pos()
        quat = self.ee_link_right.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_ee_pose(self) -> torch.Tensor:
        """Left end-effector pose (position + quaternion).

        Returns:
            (n_envs, 7) tensor with [x, y, z, qw, qx, qy, qz]
        """
        pos = self.ee_link_left.get_pos()
        quat = self.ee_link_left.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def action_space(self):
        """Action space of the robot."""
        import numpy as np

        if self.control_mode in ("JOINT_ABS", "JOINT_DELTA"):
            # 16D: [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
            # Gripper is total width (0 to 0.1m); each finger gets half internally
            gripper_low = np.array([0.0])
            gripper_high = np.array([0.1])
            low = np.concatenate(
                [
                    self.right_arm_lower.cpu().numpy(),
                    gripper_low,
                    self.left_arm_lower.cpu().numpy(),
                    gripper_low,
                ]
            )
            high = np.concatenate(
                [
                    self.right_arm_upper.cpu().numpy(),
                    gripper_high,
                    self.left_arm_upper.cpu().numpy(),
                    gripper_high,
                ]
            )
            return spaces.Box(low=low, high=high, shape=(16,), dtype=np.float32)
        else:  # EE_DELTA or EE_ABS
            # 14-DOF: [right: 3 pos + 3 rot + 1 gripper, left: 3 pos + 3 rot + 1 gripper]
            return spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        """Dimension of action space."""
        if self.control_mode in ("EE_DELTA", "EE_ABS"):
            return 14  # 7+7 (pos+rot+gripper per arm)
        return 16  # 7+1+7+1 (arm + single gripper width per arm)
