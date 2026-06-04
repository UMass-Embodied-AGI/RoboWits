from typing import Literal

from pydantic import BaseModel

from gs_gym.schemas.base_types import genesis_pydantic_config


class SPHOptionsArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    particle_size: float = 0.015


class MPMOptionsArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    grid_density: int = 64
    particle_size: float = 0.01
    enable_CPIC: bool = False


class SceneArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    # Scene type
    type: Literal["flat"] = "flat"

    # Scene dimensions
    size: tuple[float, float, float] = (1.0, 1.0, 0.1)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Physics parameters
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float
    substeps: int

    # Particle solvers (None → FlatScene defaults)
    sph_options: SPHOptionsArgs | None = None
    mpm_options: MPMOptionsArgs | None = None

    # Visualization
    show_viewer: bool = False
    show_fps: bool = False

    # Ground plane properties
    ground_friction: float = 1.0
    ground_restitution: float = 0.0
