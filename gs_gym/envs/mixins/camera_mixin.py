"""Camera mixin for visual observations in Genesis environments.

Provides shared camera infrastructure that can be mixed into environment classes
to add visual observation support.
"""

from typing import Any, ClassVar, Literal

import torch

CameraType = Literal["rasterizer", "batch"]


class CameraMixin:
    """Mixin class providing camera infrastructure for visual observations.

    This mixin adds camera setup and pixel retrieval functionality to environments.
    It supports multiple predefined camera positions commonly used in robotics
    benchmarks like Metaworld.

    Camera Types:
        - "rasterizer": Uses RasterizerCameraOptions, works with both CUDA and Vulkan backends
        - "batch": Uses BatchRendererCameraOptions, requires CUDA backend (faster for parallel envs)

    Usage:
        class MyEnv(GenesisGymEnv, CameraMixin):
            def __init__(self, obs_type="pixels", camera_type="rasterizer", ...):
                super().__init__(...)
                if obs_type in ("pixels", "pixels_state"):
                    self._setup_cameras(["corner2"], 480, 480, camera_type=camera_type)

            def get_observations(self):
                obs = {"state": self._get_state_obs()}
                if self._obs_type in ("pixels", "pixels_state"):
                    obs["pixels"] = self._get_pixels()
                return obs
    """

    # Predefined camera configurations for common robotics benchmarks
    # Positions are approximate matches to Metaworld camera views
    CAMERA_CONFIGS: ClassVar[dict[str, dict[str, Any]]] = {
        "corner2": {
            "pos": (1.2, 0.8, 1.2),
            "lookat": (0.4, 0.0, 0.4),
            "fov": 45,
        },
        "topview": {
            "pos": (0.5, 0.0, 2.0),
            "lookat": (0.5, 0.0, 0.4),
            "fov": 45,
        },
        "frontview": {
            "pos": (1.5, 0.0, 0.8),
            "lookat": (0.4, 0.0, 0.4),
            "fov": 45,
        },
        "sideview": {
            "pos": (0.5, 1.2, 0.8),
            "lookat": (0.4, 0.0, 0.4),
            "fov": 45,
        },
    }

    # Camera storage (populated by _setup_cameras)
    _cameras: dict[str, Any]
    _camera_names: list[str]
    _camera_width: int
    _camera_height: int
    _camera_type: CameraType

    def _setup_cameras(
        self,
        camera_names: list[str],
        width: int = 480,
        height: int = 480,
        camera_type: CameraType = "rasterizer",
    ) -> None:
        """Setup camera sensors for visual observations.

        Args:
            camera_names: List of camera names to create (must be in CAMERA_CONFIGS)
            width: Image width in pixels
            height: Image height in pixels
            camera_type: Type of camera to use:
                - "rasterizer": Works with both CUDA and Vulkan backends (default)
                - "batch": Requires CUDA backend, faster for parallel environments

        Raises:
            ValueError: If camera_name is not in CAMERA_CONFIGS or camera_type is invalid
            AttributeError: If _scene is not available (must be called after scene creation)
        """
        if camera_type == "rasterizer":
            from genesis.options.sensors import RasterizerCameraOptions

            CameraOptionsClass = RasterizerCameraOptions
        elif camera_type == "batch":
            from genesis.options.sensors import BatchRendererCameraOptions

            CameraOptionsClass = BatchRendererCameraOptions
        else:
            raise ValueError(f"Unknown camera_type '{camera_type}'. Available types: 'rasterizer', 'batch'")

        self._cameras = {}
        self._camera_names = camera_names
        self._camera_width = width
        self._camera_height = height
        self._camera_type = camera_type

        for name in camera_names:
            config = self.CAMERA_CONFIGS.get(name)
            if config is None:
                available = list(self.CAMERA_CONFIGS.keys())
                raise ValueError(f"Unknown camera '{name}'. Available cameras: {available}")

            camera_options = CameraOptionsClass(
                pos=config["pos"],
                lookat=config["lookat"],
                fov=config["fov"],
                res=(width, height),
            )

            # Add sensor to scene (requires _scene attribute from environment)
            camera = self._scene.scene.add_sensor(camera_options)
            self._cameras[name] = camera

    def _get_pixels(self) -> dict[str, torch.Tensor]:
        """Get RGB images from all cameras.

        Returns:
            Dictionary mapping camera names to RGB tensors of shape (n_envs, H, W, 3)
        """
        pixels = {}
        for name, camera in self._cameras.items():
            data = camera.read()
            # Genesis CameraData is a NamedTuple with 'rgb' attribute
            pixels[name] = data.rgb
        return pixels

    def _get_single_camera_pixels(self, camera_name: str) -> torch.Tensor:
        """Get RGB image from a single camera.

        Args:
            camera_name: Name of the camera to read from

        Returns:
            RGB tensor of shape (n_envs, H, W, 3)

        Raises:
            KeyError: If camera_name is not in _cameras
        """
        if camera_name not in self._cameras:
            raise KeyError(f"Camera '{camera_name}' not found. Available cameras: {list(self._cameras.keys())}")
        data = self._cameras[camera_name].read()
        return data.rgb
