from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

RetrieveCubeEnv = _import("gs_gym.envs.robowits.02_retrieve_cube").RetrieveCubeEnv


@register_task("robowits/02_05-v0")
class RetrieveCubeMut5Env(RetrieveCubeEnv):
    # SPH liquid added to a rigid seed task: far faster on the GPU backend.
    preferred_genesis_backend = "cuda"

    def _config_to_env_args(self, config):
        """Raise substeps for this variant's SPH water.

        The seed task is rigid-only and runs at the default 2 substeps, i.e. a 1.7e-2 s
        step. Adding SPH water at that step size makes the solver diverge -- 02_08's water
        reaches 6.7 m/s within 120 steps. The water-centric seeds (25, 30) run SPH at 100.
        """
        args = super()._config_to_env_args(config)
        return args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 100})})

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (("deep_narrow_container", "cube"), "target_area", ("cup", "water"), "teaspoon")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: deep narrow container
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/ddfb928e-c26d-4805-b41a-ab1545269e99/obj.glb", pattern_is_dir=False),
                scale=0.6,
                pos=(0.56, 0.00, 0.76 + 0.1332 * 0.6),
                euler=(0, 0, 0),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.9, 0.9, 1.0), double_sided=True),
        )
        self._entities["deep_narrow_container"] = {"entity": _e}
        # Code Block: cup
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.47, -0.15, 0.76 + (0.0918 * 1.1) / 2.0),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["cup"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, -0.08, 0.761), euler=(0, 0, 0), size=(0.12, 0.12, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        )
        self._entities["target_area"] = {"entity": _e}
        # Code Block: teaspoon
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/bec2248d-0f97-4683-89af-ecbb50934af2/obj.glb", pattern_is_dir=False),
                scale=1.4,
                pos=(0.56, 0.12, 0.7702),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=500.0, friction=0.8),
            surface=gs.surfaces.Metal(color=(0.8, 0.8, 0.85), double_sided=True, metal_type="aluminium"),
        )
        self._entities["teaspoon"] = {"entity": _e}
        # Code Block: cube
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.56, 0.00, 0.768), euler=(0, 0, 0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=150.0, friction=0.6, coup_restitution=0.05),
            surface=gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        )
        self._entities["cube"] = {"entity": _e}
        # Code Block: water
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.47, -0.15, 0.76 + 0.055 / 2.0 + 0.02),
                euler=(0, 0, 0),
                radius=0.018,
                height=0.055,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.SPH.Liquid(rho=1000.0, stiffness=60000.0, exponent=7.0, mu=0.01, gamma=0.02),
            surface=gs.surfaces.Default(color=(0.7, 0.85, 1.0), double_sided=True, vis_mode="recon"),
        )
        self._entities["water"] = {"entity": _e}
