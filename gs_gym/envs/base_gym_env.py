import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
import yaml

logging.getLogger("genesis").propagate = False


class GenesisGymEnv(gym.Env, ABC):
    """Base gym environment wrapper for Genesis-based environments.

    This is now a standard gym.Env (single environment, no batch dimension).
    For vectorized environments, use gs_gym.make(task, n_envs>1) which wraps
    multiple instances with gym.vector.VectorEnv.
    """

    def __init__(self, config_name: str = "default", device: str = "auto", genesis_backend: str = "cpu", **kwargs):
        """Initialize Genesis gym environment.

        Args:
            config_name: Name of configuration file to load
            device: PyTorch device for robot controller tensors ("auto", "cpu", "cuda", "mps")
                    Defaults to "auto" which uses CUDA if available for performance
            genesis_backend: Genesis physics backend ("cpu", "cuda", "gpu", "vulkan")
                             Defaults to "cpu" for stability and LeRobot compatibility
            **kwargs: Additional arguments passed to subclasses
        """
        # Single environment only (n_envs=1 internally with Genesis)
        # Remove n_envs from kwargs to avoid passing it to subclasses
        kwargs.pop("n_envs", None)

        # Initialize parent class first
        super().__init__()

        # Set num_envs for compatibility with gymnasium VectorEnv interface
        n_envs = getattr(self, "_n_envs", 1)
        self.num_envs = n_envs

        # Set metadata required by LeRobot and other frameworks
        self.metadata = {
            "render_fps": 30,  # Default FPS for video rendering
            "render_modes": ["rgb_array"],
        }

        # Store Genesis backend preference (before _resolve_device which reads it)
        self._genesis_backend = genesis_backend

        # Resolve device for robot controller computations
        self._device = self._resolve_device(device)

        # Load configuration
        self._config = self._load_config(config_name)

        # Get space information directly from config (unbatched)
        self.single_action_space, self.single_observation_space = self._get_spaces_from_config(self._config)

        # Create batched spaces if n_envs > 1 (set by subclass before calling __init__)
        n_envs = getattr(self, "_n_envs", 1)
        if n_envs > 1:
            self.action_space = self._batch_space(self.single_action_space, n_envs)
            self.observation_space = self._batch_space(self.single_observation_space, n_envs)
        else:
            self.action_space = self.single_action_space
            self.observation_space = self.single_observation_space

        # Initialize Genesis if not already done
        try:
            is_initialized = gs._initialized
        except AttributeError:
            # Newer versions of genesis may not have _initialized attribute
            is_initialized = False
        if not is_initialized:
            gs_backend = self._resolve_genesis_backend(genesis_backend)
            gs.init(backend=gs_backend)

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device for robot controller computations.

        Args:
            device: Device string ("auto", "cpu", "cuda", "mps", etc.)

        Returns:
            torch.device instance

        Note:
            For compatibility with LeRobot when using CPU Genesis backend,
            robot controller should also use CPU to avoid device mismatch issues.
            When Genesis backend is CPU, observations come as CPU tensors and
            mixing devices can cause issues in the data pipeline.
        """
        if device == "auto":
            # Match the Genesis backend device
            if self._genesis_backend in ("cuda", "gpu"):
                return torch.device("cuda")
            return torch.device("cpu")
        else:
            return torch.device(device)

    def _batch_space(self, space: gym.Space, n_envs: int) -> gym.Space:
        """Create a batched version of a space for vectorized envs.

        Args:
            space: The unbatched space
            n_envs: Number of environments

        Returns:
            Batched space with n_envs as first dimension
        """
        if isinstance(space, gym.spaces.Box):
            # Stack bounds along new first axis for batch dimension
            batched_low = np.stack([space.low] * n_envs, axis=0)
            batched_high = np.stack([space.high] * n_envs, axis=0)
            return gym.spaces.Box(low=batched_low, high=batched_high, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({key: self._batch_space(subspace, n_envs) for key, subspace in space.spaces.items()})
        else:
            return space

    def _resolve_genesis_backend(self, genesis_backend: str):
        """Resolve Genesis backend string to Genesis backend constant.

        Args:
            genesis_backend: Backend type string:
                - "auto": Auto-detect best available backend (defaults to CPU for stability)
                - "cuda": Use CUDA backend (requires NVIDIA GPU with CUDA)
                - "gpu": Use GPU backend (auto-selects CUDA or Vulkan)
                - "vulkan": Use Vulkan backend
                - "cpu": Use CPU backend

        Returns:
            Genesis backend constant (gs.cuda, gs.gpu, gs.vulkan, or gs.cpu)
        """
        if genesis_backend == "auto":
            # Default to CPU for stability and compatibility with LeRobot
            # (avoids torch.set_default_device('cuda') conflicts)
            # Use genesis_backend="cuda" or genesis_backend="gpu" explicitly for GPU acceleration
            return gs.cpu
        elif genesis_backend == "cuda":
            return gs.cuda
        elif genesis_backend == "gpu":
            return gs.gpu
        elif genesis_backend == "vulkan":
            return gs.vulkan
        elif genesis_backend == "cpu":
            return gs.cpu
        else:
            raise ValueError(
                f"Unknown genesis_backend '{genesis_backend}'. Available backends: 'auto', 'cuda', 'gpu', 'vulkan', 'cpu'"
            )

    def _load_config(self, config_name: str) -> dict:
        """Load configuration from embedded YAML files."""
        import gs_gym

        # Get the package directory
        package_dir = os.path.dirname(gs_gym.__file__)
        config_path = os.path.join(package_dir, "configs", "envs", f"{config_name}.yaml")

        if not os.path.exists(config_path):
            raise ValueError(f"Configuration '{config_name}' not found at {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config

    @abstractmethod
    def _get_spaces_from_config(self, config: dict):
        """Get space information directly from config. Must be implemented by subclasses."""
        pass

    def reset(
        self, *, seed: int | list | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray | dict, dict[str, Any]]:
        """Reset the environment."""
        # gymnasium expects a single int seed; vectorized callers may pass a list
        if isinstance(seed, list | tuple):
            seed = seed[0] if seed else None
        super().reset(seed=seed, options=options)

        # Reset the environment
        self.reset_env()

        # Get initial observation
        obs = self.get_observations()
        info = self.get_extra_infos()
        # Convert torch tensors to numpy arrays for gym
        obs = self._convert_obs_to_numpy(obs)
        if isinstance(info, dict):
            info = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in info.items()}

        return obs, info

    @property
    def envs(self) -> list:
        """VectorEnv-compatible: return self as the list of sub-environments."""
        return [self]

    @property
    def unwrapped(self):
        return self

    def call(self, name: str, *args, **kwargs) -> list:
        """VectorEnv-compatible call: get attribute or call method, return as list per env."""
        attr = getattr(self, name)
        result = attr(*args, **kwargs) if callable(attr) else attr
        # Return as a list with one entry per env
        n = getattr(self, "num_envs", 1)
        return [result] * n

    def step(self, action: np.ndarray) -> tuple[np.ndarray | dict, float, bool, bool, dict[str, Any]]:
        """Step the environment with a single action.

        Args:
            action: Single action (no batch dimension)

        Returns:
            obs: Observation (no batch dimension)
            reward: Float reward value
            terminated: Boolean termination flag
            truncated: Boolean truncation flag
            info: Dictionary of additional information
        """
        # Convert numpy action to torch tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._device)

        # Add batch dimension if needed (action_space.sample() returns 1D)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Apply action to environment
        self.apply_action(action)

        # Get observation, reward, and termination info
        obs = self.get_observations()
        reward_info = self.get_reward()
        reward, reward_dict = reward_info
        terminated = self.get_terminated()
        truncated = self.get_truncated()

        # Convert tensors to numpy/scalars
        obs = self._convert_obs_to_numpy(obs)

        # Convert reward/terminated/truncated based on n_envs
        if isinstance(reward, torch.Tensor):
            reward = float(reward.item()) if reward.numel() == 1 else reward.detach().cpu().numpy()
        else:
            reward = float(reward)

        if isinstance(terminated, torch.Tensor):
            terminated = bool(terminated.item()) if terminated.numel() == 1 else terminated.detach().cpu().numpy()
        else:
            terminated = bool(terminated)

        if isinstance(truncated, torch.Tensor):
            truncated = bool(truncated.item()) if truncated.numel() == 1 else truncated.detach().cpu().numpy()
        else:
            truncated = bool(truncated)

        # Get extra info (includes is_success, step_count, etc.)
        info = self.get_extra_infos()

        # Merge with reward dict
        if isinstance(info, dict):
            info.update(reward_dict)
            # Convert any tensor values to scalars/numpy
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    # Convert tensor info to scalar or numpy
                    if v.numel() == 1:
                        info[k] = v.item()
                    else:
                        info[k] = v.detach().cpu().numpy()

        return obs, reward, terminated, truncated, info

    def _convert_obs_to_numpy(self, obs: torch.Tensor | dict) -> np.ndarray | dict:
        """Convert observation (tensor or nested dict) to numpy.

        When n_envs=1, squeezes the batch dimension for compatibility with SyncVectorEnv.
        """
        n_envs = getattr(self, "_n_envs", 1)
        if isinstance(obs, torch.Tensor):
            arr = obs.detach().cpu().numpy()
            # Squeeze batch dimension when n_envs=1 for SyncVectorEnv compatibility
            if n_envs == 1 and arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            return arr
        elif isinstance(obs, dict):
            return {k: self._convert_obs_to_numpy(v) for k, v in obs.items()}
        elif isinstance(obs, np.ndarray):
            # Also handle numpy arrays that may have been converted elsewhere
            if n_envs == 1 and obs.ndim > 1 and obs.shape[0] == 1:
                return obs.squeeze(0)
            return obs
        else:
            return obs

    def close(self):
        """Close the environment and release Genesis/EGL resources."""
        try:
            is_initialized = gs._initialized
        except AttributeError:
            is_initialized = False
        if is_initialized:
            gs.destroy()

    def render(self, mode: str = "human"):
        """Render the environment."""
        # For now, return None (no rendering support)
        return None

    @property
    def device(self):
        """Device property."""
        return self._device

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Return the task description string for this environment.

        Required by LeRobot policies that use language-conditioned learning.
        Subclasses MUST implement this property.

        Returns:
            str: Human-readable task description (e.g., "Pick up the red cube and place it on the target.")
        """
        pass

    @property
    def task(self) -> str:
        """Return the task name/identifier string for this environment.

        Required by LeRobot policies. Returns task_description by default.
        Override in subclasses if task name should differ from description.

        Returns:
            str: Task name or identifier
        """
        return self.task_description

    # Abstract methods that subclasses must implement
    @abstractmethod
    def reset_env(self) -> None:
        """Reset the environment (single environment)."""
        pass

    @abstractmethod
    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to the environment (single action, no batch dimension)."""
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor | dict[str, Any]:
        """Get current observations (single observation, no batch dimension).

        Returns:
            torch.Tensor or dict: Observations in environment-specific format.
            - For state-only envs: torch.Tensor of shape (obs_dim,)
            - For pixel envs: dict with "pixels" and/or "agent_pos" keys
        """
        pass

    @abstractmethod
    def get_reward(self) -> tuple[torch.Tensor | float, dict[str, Any]]:
        """Get reward and reward components (scalar reward)."""
        pass

    @abstractmethod
    def get_terminated(self) -> torch.Tensor | bool:
        """Check if episode is terminated (scalar bool)."""
        pass

    @abstractmethod
    def get_truncated(self) -> torch.Tensor | bool:
        """Check if episode is truncated (scalar bool)."""
        pass

    @abstractmethod
    def get_extra_infos(self) -> dict[str, Any]:
        """Get extra information."""
        pass
