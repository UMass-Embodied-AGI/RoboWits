"""GS-Gym: Genesis-based robotics environments for OpenAI Gym.

This package provides a unified interface for robotic manipulation tasks
using the Genesis simulator. It supports multiple benchmarks and provides
a consistent API for training and evaluation.

Example:
    import gs_gym

    # Create an environment
    env = gs_gym.make("robowits/01-align-blocks-v0", n_envs=4)

    # List available tasks
    print(gs_gym.list_tasks())

    # List tasks by benchmark
    print(gs_gym.list_tasks(benchmark="robowits"))
"""

# Import registry functions first (before importing envs)
from gs_gym.envs.registry import (
    TASK_REGISTRY,
    get_task_class,
    is_registered,
    list_benchmarks,
    list_tasks,
    make,
    register_task,
)

# Import RoboWits environments (registers tasks via decorators)
from gs_gym.envs.robowits import AlignBlocksEnv, RoboWitsEnv  # noqa: F401

# Import robot classes (imports trigger @register_robot decorators)
from gs_gym.robots.bimanual_marvin import BimanualMarvinRobot

# Import robot registry
from gs_gym.robots.registry import (
    ROBOT_REGISTRY,
    create_robot,
    get_robot_class,
    is_robot_registered,
    list_robots,
    register_robot,
)

__version__ = "0.1.0"
__all__ = [
    # Version
    "__version__",
    # Task registry
    "make",
    "register_task",
    "list_tasks",
    "list_benchmarks",
    "get_task_class",
    "is_registered",
    "TASK_REGISTRY",
    # Robot registry
    "create_robot",
    "register_robot",
    "list_robots",
    "get_robot_class",
    "is_robot_registered",
    "ROBOT_REGISTRY",
    # Environment classes
    "RoboWitsEnv",
    # Robot classes
    "BimanualMarvinRobot",
]
