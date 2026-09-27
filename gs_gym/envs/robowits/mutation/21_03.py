from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

SealColanderEnv = _import("gs_gym.envs.robowits.21_seal_colander").SealColanderEnv


@register_task("robowits/21_03-v0")
class SealColanderMut3Env(SealColanderEnv):
    def _config_to_env_args(self, config):
        """Raise substeps for this variant's MPM dough.

        The seed task's 30 substeps are sized for its SPH water. This variant also adds
        MPM.ElastoPlastic dough at E=5e5, whose wave speed (22.4 m/s against a 0.0156 m
        grid) needs a substep under 7e-4 s; 30 substeps gives 1.1e-3 and the solver goes
        non-finite on the first reset. Task 08's dough runs at 100 for the same reason.
        """
        args = super()._config_to_env_args(config)
        return args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})

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
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: perforated container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("toolbench/slotted_spoon.glb", pattern_is_dir=False),
                scale=0.8847,
                pos=(0.68, 0.00, 0.779),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.6),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["perforated_container"] = {"entity": _e}
        # Code Block: pitcher
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb", pattern_is_dir=False),
                scale=1.2476,
                pos=(0.45, -0.10, 0.76 + 0.0721 * 1.2476),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(rho=200.0, sdf_max_res=256),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["pitcher"] = {"entity": _e}
        self._add_pitcher_blockers(0.45, -0.1)
        # Code Block: dough
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(pos=(0.66, 0.2, 0.76 + 0.018), radius=0.018, euler=(0, 0, 0), fixed=False, collision=True),
            material=gs.materials.MPM.ElastoPlastic(
                E=5e5, nu=0.3, rho=1000.0, sampler="pbs", von_mises_yield_stress=8000.0
            ),
            surface=gs.surfaces.Rough(double_sided=True, vis_mode="recon"),
        )
        self._entities["dough"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.45, -0.10, 0.86), euler=(0, 0, 0), radius=0.045, height=0.13, fixed=False, collision=True
            ),
            material=gs.materials.SPH.Liquid(
                rho=500.0, stiffness=80000.0, exponent=7.0, mu=0.005, gamma=0.01, sampler="pbs"
            ),
            surface=gs.surfaces.Default(color=(0.6, 0.7, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
