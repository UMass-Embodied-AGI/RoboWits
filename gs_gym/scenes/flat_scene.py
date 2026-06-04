import genesis as gs
import torch

from gs_gym.scenes.schema import MPMOptionsArgs, SPHOptionsArgs

# Table footprint: TABLE_POS=(0.597, 0, 0.38), TABLE_SIZE=(0.85, 1.5, 0.76)
_WORK_AREA_BOUND = ((0.172, 1.022), (-0.75, 0.75), 0.73)  # ((x_min, x_max), (y_min, y_max), z_surface)

_BOUND_MARGIN = 0.05
_SOLVER_LOWER = (
    _WORK_AREA_BOUND[0][0] - _BOUND_MARGIN,
    _WORK_AREA_BOUND[1][0] - _BOUND_MARGIN,
    _WORK_AREA_BOUND[2] - _BOUND_MARGIN,
)
_SOLVER_UPPER = (
    _WORK_AREA_BOUND[0][1] + _BOUND_MARGIN,
    _WORK_AREA_BOUND[1][1] + _BOUND_MARGIN,
    _WORK_AREA_BOUND[2] + 1.5,
)

_DEFAULT_SPH_OPTIONS = SPHOptionsArgs()
_DEFAULT_MPM_OPTIONS = MPMOptionsArgs()


def _sph_options_from_args(sph_options: SPHOptionsArgs | None) -> gs.options.SPHOptions:
    cfg = sph_options if sph_options is not None else _DEFAULT_SPH_OPTIONS
    return gs.options.SPHOptions(
        particle_size=cfg.particle_size,
        lower_bound=_SOLVER_LOWER,
        upper_bound=_SOLVER_UPPER,
    )


def _mpm_options_from_args(mpm_options: MPMOptionsArgs | None) -> gs.options.MPMOptions:
    cfg = mpm_options if mpm_options is not None else _DEFAULT_MPM_OPTIONS
    return gs.options.MPMOptions(
        grid_density=cfg.grid_density,
        particle_size=cfg.particle_size,
        enable_CPIC=cfg.enable_CPIC,
        lower_bound=_SOLVER_LOWER,
        upper_bound=_SOLVER_UPPER,
    )


class FlatScene:
    """Flat scene implementation for Genesis environments."""

    def __init__(
        self,
        n_envs: int,
        args,  # SceneArgs
        show_viewer: bool = False,
        img_resolution: tuple[int, int] | None = None,
        enable_collision: bool = False,
        add_ground_plane: bool = False,
    ) -> None:
        self._n_envs = n_envs

        # Create Genesis scene
        substeps = getattr(args, "substeps", 2)
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=args.dt,
                gravity=args.gravity,
                substeps=substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=enable_collision,
                tolerance=1e-6,
                noslip_iterations=10,
            ),
            sph_options=_sph_options_from_args(getattr(args, "sph_options", None)),
            mpm_options=_mpm_options_from_args(getattr(args, "mpm_options", None)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane if requested
        self._plane = None
        if add_ground_plane:
            self._plane = self._scene.add_entity(gs.morphs.Plane())

    def build(self) -> None:
        """Build the scene."""
        # Use constants for env_spacing and n_envs_per_row as specified
        env_spacing = (1.0, 1.0)  # Constant spacing between environments
        n_envs_per_row = None  # Let Genesis determine optimal layout

        self._scene.build(
            n_envs=self._n_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
        )

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset the scene for the given environments."""
        self._scene.reset(envs_idx=envs_idx)

    @property
    def scene(self):
        """The underlying Genesis scene object."""
        return self._scene

    @property
    def n_envs(self) -> int:
        """Number of environments in this scene."""
        return self._n_envs
