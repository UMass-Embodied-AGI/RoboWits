from importlib import import_module as _import

import genesis as gs

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups
from gs_gym.scenes.schema import MPMOptionsArgs

SealColanderEnv = _import("gs_gym.envs.robowits.21_seal_colander").SealColanderEnv


@register_task("robowits/21_03-v0")
class SealColanderMut3Env(SealColanderEnv):
    #: Dough radius (m): pressed into a ~4 cm disc about 1.1 cm thick it covers the 1.6 cm hole and fits the 6 cm bowl.
    DOUGH_RADIUS = 0.015
    #: MPM couples to rigid bodies through its grid (7.8 mm cells here): the dough fell straight through the default
    #: 4 mm floor, so this ladle's floor is thicker than a cell. The hole becomes a 1.2 cm deep channel.
    LADLE_FLOOR = 0.012
    #: The pressed-dough seal was verified on a 1.6 cm hole; the dome variants' 2.2 cm leaves a 4 cm disc little margin.
    LADLE_HOLE_SIDE = 0.016

    def _config_to_env_args(self, config):
        """Resolve the dough finely enough to seal the hole.

        At the default MPM resolution (1 cm particles, 64-cell grid) the dough was 23 particles about 1 cm apart:
        it could not seal a 1.6 cm hole against the water, and its reconstructed surface rendered as nothing.
        4 mm particles on a 128 grid give ~200. E is 1e5 (was 5e5) so its wave speed (10 m/s against a 7.8 mm
        cell) stays inside the 100-substep step; 5e5 would need a substep under 3.5e-4 s, against 3.3e-4.
        """
        args = super()._config_to_env_args(config)
        scene_args = args.scene_args.model_copy(
            update={"mpm_options": MPMOptionsArgs(grid_density=128, particle_size=0.004)}
        )
        return args.model_copy(update={"scene_args": scene_args})

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Randomize the container, holder and clutter -- never the pitcher or its water.

        The water is authored inside the pitcher, so grouping it with the perforated
        container (24 cm away) dragged the liquid out of the pitcher on every reset.
        21_05 documents hitting exactly that and blowing up the solver.
        """
        return ("perforated_container", "dough")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""

        # Code Block: perforated container (the box ladle, see 21_seal_colander.py)
        self._add_ladle((0.68, 0.00), yaw_deg=180.0)
        # Code Block: pitcher with its water (task 25's, see 21_seal_colander.py)
        self._add_pitcher_with_water(0.45, -0.1)
        # Code Block: dough
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.66, 0.2, 0.76 + self.DOUGH_RADIUS),
                radius=self.DOUGH_RADIUS,
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.MPM.ElastoPlastic(
                E=1e5, nu=0.3, rho=1000.0, sampler="pbs", von_mises_yield_stress=8000.0
            ),
            surface=gs.surfaces.Rough(color=(0.86, 0.7, 0.45), double_sided=True, vis_mode="recon"),  # tan dough
        )
        self._entities["dough"] = {"entity": _e}
