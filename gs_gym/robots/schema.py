from typing import Literal

from pydantic import BaseModel

from gs_gym.schemas.base_types import genesis_pydantic_config


class ManipulatorRobotArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    # Robot type
    type: Literal["franka", "bimanual_marvin"] = "franka"

    # Position and orientation
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Joint limits and damping
    joint_damping: float = 0.1
    joint_armature: float = 0.1

    # Control parameters
    action_scale: float = 1.0
    max_joint_velocity: float = 2.0

    # End effector configuration (Franka-specific, ignored for bimanual_marvin)
    gripper_action_idx: int = 6
