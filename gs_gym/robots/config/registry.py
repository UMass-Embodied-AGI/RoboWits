from gs_gym.robots.config.schema import ManipulatorRobotArgs

# Registry of predefined robot configurations
ROBOT_CONFIGS = {
    "franka_default": ManipulatorRobotArgs(
        type="franka",
        position=(0.0, 0.0, 0.5),
        euler=(0.0, 0.0, 0.0),
        joint_damping=0.1,
        joint_armature=0.1,
        action_scale=1.0,
        max_joint_velocity=2.0,
        gripper_action_idx=6,
    )
}
