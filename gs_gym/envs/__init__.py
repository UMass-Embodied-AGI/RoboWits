# Import RoboWits environments (registers tasks via decorators)
from gs_gym.envs import robowits

from .base_gym_env import GenesisGymEnv

__all__ = [
    "GenesisGymEnv",
    "robowits",
]
