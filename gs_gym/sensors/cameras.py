"""Camera configurations and multi-camera observation system.

Provides camera presets for various benchmarks and a unified MultiCameraObserver
for managing multiple camera views with RGB and depth modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    import genesis as gs


CameraType = Literal["rasterizer", "batch"]


@dataclass
class CameraConfig:
    """Configuration for a single camera.

    Attributes:
        name: Camera identifier
        pos: Camera position in world frame (x, y, z)
        lookat: Point camera looks at (x, y, z)
        resolution: Image resolution (width, height)
        fov: Field of view in degrees
        mount: Mount point for end-effector cameras ("ee") or None for static
        modalities: List of observation types ("rgb", "depth")
    """

    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    resolution: tuple[int, int] = (128, 128)
    fov: float = 45.0
    mount: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["rgb", "depth"])


def _normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize RGB tensor to [0, 1] float if needed."""
    if rgb.dtype == torch.uint8:
        return rgb.float() / 255.0
    return rgb


# RLBench camera presets matching the standard 5-camera setup
# Positions are calibrated to match RLBench's CoppeliaSim setup
RLBENCH_CAMERA_CONFIGS: dict[str, CameraConfig] = {
    "front": CameraConfig(
        name="front",
        pos=(1.5, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=60.0,
    ),
    "left_shoulder": CameraConfig(
        name="left_shoulder",
        pos=(0.3, -0.8, 1.2),
        lookat=(0.0, 0.0, 0.5),
        fov=60.0,
    ),
    "right_shoulder": CameraConfig(
        name="right_shoulder",
        pos=(0.3, 0.8, 1.2),
        lookat=(0.0, 0.0, 0.5),
        fov=60.0,
    ),
    "overhead": CameraConfig(
        name="overhead",
        pos=(0.0, 0.0, 2.0),
        lookat=(0.0, 0.0, 0.5),
        fov=60.0,
    ),
    "wrist": CameraConfig(
        name="wrist",
        pos=(0.0, 0.0, 0.05),
        lookat=(0.0, 0.0, 0.15),
        fov=60.0,
        mount="ee",
    ),
}


class MultiCameraObserver:
    """Manages multiple cameras for RLBench-style observations.

    Provides a unified interface for setting up and reading from multiple
    cameras with configurable modalities (RGB, depth).

    Usage:
        # In environment __init__:
        self.camera_observer = MultiCameraObserver(
            scene=self._scene.scene,
            cameras=["front", "left_shoulder", "overhead", "wrist"],
            resolution=(128, 128),
        )

        # In get_observations():
        camera_obs = self.camera_observer.get_observations()
        # Returns: {"front_rgb": tensor, "front_depth": tensor, ...}
    """

    def __init__(
        self,
        scene: gs.Scene,
        cameras: list[str] | None = None,
        configs: dict[str, CameraConfig] | None = None,
        resolution: tuple[int, int] = (128, 128),
        camera_type: CameraType = "rasterizer",
        modalities: list[str] | None = None,
        ee_link: Any | None = None,
    ) -> None:
        """Initialize the multi-camera observer.

        Args:
            scene: Genesis scene to add cameras to
            cameras: List of camera names from RLBENCH_CAMERA_CONFIGS.
                If None, uses all 5 standard cameras.
            configs: Optional custom camera configs (overrides RLBENCH_CAMERA_CONFIGS)
            resolution: Image resolution (width, height) for all cameras
            camera_type: Type of camera ("rasterizer" or "batch")
            modalities: Modalities to capture (["rgb", "depth"] by default)
            ee_link: End-effector link for wrist camera mounting
        """
        self.scene = scene
        self.resolution = resolution
        self.camera_type = camera_type
        self.modalities = modalities or ["rgb", "depth"]
        self.ee_link = ee_link

        # Use default configs if not provided
        camera_configs = configs or RLBENCH_CAMERA_CONFIGS

        # Use all cameras if not specified
        if cameras is None:
            cameras = list(camera_configs.keys())

        self._cameras: dict[str, Any] = {}
        self._camera_configs: dict[str, CameraConfig] = {}

        for name in cameras:
            if name not in camera_configs:
                available = list(camera_configs.keys())
                raise ValueError(f"Unknown camera '{name}'. Available: {available}")
            config = camera_configs[name]
            self._camera_configs[name] = config
            self._cameras[name] = self._create_camera(config)

    def _create_camera(self, config: CameraConfig) -> Any:
        """Create a Genesis camera from config.

        Args:
            config: Camera configuration

        Returns:
            Genesis camera sensor
        """
        if self.camera_type == "rasterizer":
            from genesis.options.sensors import RasterizerCameraOptions

            CameraOptionsClass = RasterizerCameraOptions
        elif self.camera_type == "batch":
            from genesis.options.sensors import BatchRendererCameraOptions

            CameraOptionsClass = BatchRendererCameraOptions
        else:
            raise ValueError(f"Unknown camera_type '{self.camera_type}'")

        # Build camera options - either attached to link or absolute position
        if config.mount == "ee" and self.ee_link is not None:
            # Attached camera: use entity_idx/link_idx_local with position offset
            camera_kwargs = {
                "entity_idx": self.ee_link.entity.idx,
                "link_idx_local": self.ee_link.idx_local,
                "pos_offset": config.pos,
                "euler_offset": (0.0, 0.0, 0.0),
                "fov": config.fov,
                "res": self.resolution,
            }
        else:
            # Static camera: use absolute position and lookat
            camera_kwargs = {
                "pos": config.pos,
                "lookat": config.lookat,
                "fov": config.fov,
                "res": self.resolution,
            }

        camera_options = CameraOptionsClass(**camera_kwargs)
        camera = self.scene.add_sensor(camera_options)

        return camera

    def get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations from all cameras.

        Returns:
            Dictionary mapping "{camera_name}_{modality}" to tensors:
            - RGB: (n_envs, H, W, 3) float32 in [0, 1]
            - Depth: (n_envs, H, W) float32 in meters (if available)

        Note:
            Genesis CameraData only returns RGB by default. Depth requires
            a separate DepthCameraSensor. If depth is requested but not
            available, it will be silently skipped.
        """
        observations = {}

        for name, camera in self._cameras.items():
            data = camera.read()

            if "rgb" in self.modalities:
                observations[f"{name}_rgb"] = _normalize_rgb(data.rgb)

            if "depth" in self.modalities and hasattr(data, "depth") and data.depth is not None:
                observations[f"{name}_depth"] = data.depth

        return observations

    def get_rgb(self, camera_name: str) -> torch.Tensor:
        """Get RGB image from a specific camera.

        Args:
            camera_name: Name of the camera

        Returns:
            RGB tensor of shape (n_envs, H, W, 3)
        """
        if camera_name not in self._cameras:
            raise KeyError(f"Camera '{camera_name}' not found")
        return _normalize_rgb(self._cameras[camera_name].read().rgb)

    def get_depth(self, camera_name: str) -> torch.Tensor:
        """Get depth image from a specific camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Depth tensor of shape (n_envs, H, W)
        """
        if camera_name not in self._cameras:
            raise KeyError(f"Camera '{camera_name}' not found")
        data = self._cameras[camera_name].read()
        return data.depth

    @property
    def camera_names(self) -> list[str]:
        """Get list of camera names."""
        return list(self._cameras.keys())

    @property
    def observation_keys(self) -> list[str]:
        """Get list of observation keys that will be returned."""
        keys = []
        for name in self._cameras:
            if "rgb" in self.modalities:
                keys.append(f"{name}_rgb")
            if "depth" in self.modalities:
                keys.append(f"{name}_depth")
        return keys
