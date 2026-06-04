"""Task registry for gs-gym environments.

Provides a simple decorator-based registration system for task classes,
with factory functions and task listing capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from gs_gym.envs.base_gym_env import GenesisGymEnv

# Global task registry
TASK_REGISTRY: dict[str, type[GenesisGymEnv]] = {}


def register_task(name: str) -> Callable[[type], type]:
    """Decorator to register a task class.

    Args:
        name: Task identifier in "benchmark/task-name" format
              (e.g., "robowits/01-align-blocks-v0", "gs-gym/goal-reaching-v0")

    Returns:
        Decorator function that registers the class

    Example:
        @register_task("robowits/01-align-blocks-v0")
        class AlignBlocksEnv(GenesisGymEnv):
            ...
    """

    def decorator(cls: type) -> type:
        if name in TASK_REGISTRY:
            raise ValueError(
                f"Task '{name}' is already registered to {TASK_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__}."
            )
        TASK_REGISTRY[name] = cls
        # Store the registered name on the class for introspection
        cls._registered_name = name
        return cls

    return decorator


def make(name: str, n_envs: int = 1, **kwargs) -> gym.vector.VectorEnv:
    """Factory function to create vectorized task instances.

    Args:
        name: Registered task name (e.g., "robowits/01-align-blocks-v0")
        n_envs: Number of parallel environments (default: 1)
        **kwargs: Task-specific arguments (robot, control_mode, etc.)

    Returns:
        gym.vector.VectorEnv wrapping n_envs environment instances

    Raises:
        ValueError: If task name is not registered

    Example:
        # Single environment (still returns VectorEnv)
        env = gs_gym.make("robowits/01-align-blocks-v0")

        # Multiple environments
        envs = gs_gym.make("robowits/01-align-blocks-v0", n_envs=4)
    """
    if name not in TASK_REGISTRY:
        available = list_tasks()
        raise ValueError(
            f"Unknown task: '{name}'. Available tasks: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    env_class = TASK_REGISTRY[name]

    def _make_env():
        return env_class(**kwargs)

    return gym.vector.SyncVectorEnv(
        [_make_env for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )


def list_tasks(benchmark: str | None = None) -> list[str]:
    """List registered tasks, optionally filtered by benchmark.

    Args:
        benchmark: Optional benchmark prefix filter (e.g., "robowits")

    Returns:
        Sorted list of task names

    Example:
        gs_gym.list_tasks()                       # All tasks
        gs_gym.list_tasks(benchmark="robowits")  # Only robowits tasks
    """
    tasks = list(TASK_REGISTRY.keys())
    if benchmark:
        tasks = [t for t in tasks if t.startswith(f"{benchmark}/")]
    return sorted(tasks)


def list_benchmarks() -> list[str]:
    """List all benchmarks with registered tasks.

    Returns:
        Sorted list of benchmark names

    Example:
        gs_gym.list_benchmarks()  # ["robowits", ...]
    """
    benchmarks = set()
    for task_name in TASK_REGISTRY:
        if "/" in task_name:
            benchmarks.add(task_name.split("/")[0])
    return sorted(benchmarks)


def get_task_class(name: str) -> type[GenesisGymEnv]:
    """Get task class without instantiation.

    Args:
        name: Registered task name

    Returns:
        The registered task class

    Raises:
        ValueError: If task name is not registered
    """
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: '{name}'")
    return TASK_REGISTRY[name]


def is_registered(name: str) -> bool:
    """Check if a task name is registered.

    Args:
        name: Task name to check

    Returns:
        True if registered, False otherwise
    """
    return name in TASK_REGISTRY


def unregister_task(name: str) -> None:
    """Unregister a task (primarily for testing).

    Args:
        name: Task name to unregister

    Raises:
        ValueError: If task name is not registered
    """
    if name not in TASK_REGISTRY:
        raise ValueError(f"Cannot unregister unknown task: '{name}'")
    del TASK_REGISTRY[name]


def clear_registry() -> None:
    """Clear all registered tasks (primarily for testing)."""
    TASK_REGISTRY.clear()
