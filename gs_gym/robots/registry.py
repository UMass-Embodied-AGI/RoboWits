"""Robot registry for gs-gym environments.

Provides a simple decorator-based registration system for robot classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Global robot registry
ROBOT_REGISTRY: dict[str, type] = {}


def register_robot(name: str) -> Callable[[type], type]:
    """Decorator to register a robot class.

    Args:
        name: Robot identifier (e.g., "franka", "sawyer", "ur5")

    Returns:
        Decorator function that registers the class

    Example:
        @register_robot("franka")
        class FrankaRobot:
            ...
    """

    def decorator(cls: type) -> type:
        if name in ROBOT_REGISTRY:
            raise ValueError(
                f"Robot '{name}' is already registered to {ROBOT_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__}."
            )
        ROBOT_REGISTRY[name] = cls
        cls._registered_name = name
        return cls

    return decorator


def create_robot(name: str, **kwargs) -> Any:
    """Factory function to create robot instances.

    Args:
        name: Registered robot name
        **kwargs: Robot-specific arguments (n_envs, scene, device, control_mode, etc.)

    Returns:
        Instantiated robot

    Raises:
        ValueError: If robot name is not registered
    """
    if name not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown robot: '{name}'. Available: {list_robots()}")
    return ROBOT_REGISTRY[name](**kwargs)


def list_robots() -> list[str]:
    """List all registered robots.

    Returns:
        Sorted list of robot names
    """
    return sorted(ROBOT_REGISTRY.keys())


def get_robot_class(name: str) -> type:
    """Get robot class without instantiation.

    Args:
        name: Registered robot name

    Returns:
        The registered robot class

    Raises:
        ValueError: If robot name is not registered
    """
    if name not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown robot: '{name}'")
    return ROBOT_REGISTRY[name]


def is_robot_registered(name: str) -> bool:
    """Check if a robot name is registered.

    Args:
        name: Robot name to check

    Returns:
        True if registered, False otherwise
    """
    return name in ROBOT_REGISTRY


# Auto-register built-in robots when this module is imported
def _register_builtin_robots() -> None:
    """Register built-in robot classes."""
    from gs_gym.robots.franka import FrankaRobot

    if "franka" not in ROBOT_REGISTRY:
        ROBOT_REGISTRY["franka"] = FrankaRobot

    # Try to import and register Sawyer if available
    try:
        from gs_gym.robots.sawyer import SawyerRobot

        if "sawyer" not in ROBOT_REGISTRY:
            ROBOT_REGISTRY["sawyer"] = SawyerRobot
    except ImportError:
        pass  # Sawyer not yet implemented

    # Try to import and register BimanualMarvin if available
    try:
        from gs_gym.robots.bimanual_marvin import BimanualMarvinRobot

        if "bimanual_marvin" not in ROBOT_REGISTRY:
            ROBOT_REGISTRY["bimanual_marvin"] = BimanualMarvinRobot
    except ImportError:
        pass  # BimanualMarvin not yet implemented


# Don't auto-register on import to avoid circular imports
# Users should call _register_builtin_robots() or import robots directly
