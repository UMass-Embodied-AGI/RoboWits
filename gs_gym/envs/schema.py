from pydantic import BaseModel

from gs_gym.robots.schema import ManipulatorRobotArgs
from gs_gym.scenes.schema import SceneArgs
from gs_gym.schemas.base_types import genesis_pydantic_config


class EnvArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    scene_args: SceneArgs
    robot_args: ManipulatorRobotArgs
