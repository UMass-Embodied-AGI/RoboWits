"""Bimanual Marvin robot implementation for Genesis environments.

A dual-arm robot with 7 DOF per arm and 2-finger gripper per arm.
"""

from typing import ClassVar

import numpy as np

from gs_gym.robots.bimanual import BBox, BimanualRobot
from gs_gym.robots.registry import register_robot


@register_robot("bimanual_marvin")
class BimanualMarvinRobot(BimanualRobot):
    """Bimanual Marvin robot with Pika grippers.

    - 2 arms: 7 DOF each (Joint1-7_R and Joint1-7_L)
    - 2 grippers: 2 prismatic joints each (Joint8-9_R and Joint8-9_L)
    - Total: 18 physical DOFs (but user-facing API uses 16D with single gripper width)
    """

    #: Joint names in order (right arm, then left arm)
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

    #: Wrist camera mount: Pika gripper + D405 RealSense
    WRIST_CAMERA_OFFSET_T: ClassVar[np.ndarray] = np.array(
        [
            [-1, 0, 0, 0.009],
            [0, 1, 0, 0.04931],
            [0, 0, -1, 0.06159 - 0.0037 + 0.002],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

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

    #: Joint limits from the URDF (Joint1-7, same for both arms); replaced by the built URDF's own
    ARM_LOWER: ClassVar[tuple[float, ...]] = (-3.1067, -1.8, -3.1067, -2.5307, -3.1067, -1.047, -1.047)
    ARM_UPPER: ClassVar[tuple[float, ...]] = (3.1067, 2.0944, 3.1067, -0.08727, 3.1067, 1.047, 1.047)
    #: Each finger travels 0 to 0.05 m, so the gripper width spans 0 to 0.1 m
    FINGER_MAX: ClassVar[float] = 0.05

    # fmt: off
    RESET_RIGHT_ARM: ClassVar[tuple[float, ...]] = (-1.5708, -1.3089969, 1.5708, -1.5708, -1.3089969, 0.0, -0.349)
    RESET_LEFT_ARM: ClassVar[tuple[float, ...]] = (1.5708, -1.3089969, -1.5708, -1.5708, 1.3089969, 0.0, 0.349)
    # fmt: on
    RESET_FINGER: ClassVar[float] = 0.045

    #: Measured from all link-center positions after reset(), base at (0, 0, 1.08); add ~0.04 m for
    #: link radius when using as collision bounds. The forearm/wrist/gripper region is the part near
    #: table height (z ≈ 0.84–0.94 m); the full extent includes the upper arm and shoulder.
    REST_FOOTPRINT: ClassVar[dict[str, BBox]] = {
        "forearm_right": {"x": (0.018, 0.622), "y": (-0.334, -0.244), "z": (0.857, 0.935)},
        "forearm_left": {"x": (0.019, 0.619), "y": (0.243, 0.332), "z": (0.842, 0.933)},
        "full_right": {"x": (0.000, 0.622), "y": (-0.334, -0.037), "z": (0.857, 1.200)},
        "full_left": {"x": (0.000, 0.619), "y": (0.037, 0.332), "z": (0.842, 1.200)},
    }

    @classmethod
    def default_urdf_path(cls) -> str:
        from gs_gym.common.utils.asset_utils import get_bimanual_marvin_urdf_path

        return get_bimanual_marvin_urdf_path()
