"""Sensor system for gs-gym environments.

Provides camera configurations and observation infrastructure for various benchmarks.
"""

from gs_gym.sensors.cameras import (
    RLBENCH_CAMERA_CONFIGS,
    CameraConfig,
    MultiCameraObserver,
)

__all__ = [
    "CameraConfig",
    "MultiCameraObserver",
    "RLBENCH_CAMERA_CONFIGS",
]
