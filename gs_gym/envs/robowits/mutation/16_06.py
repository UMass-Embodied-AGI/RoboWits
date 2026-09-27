from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

StandBulbEnv = _import("gs_gym.envs.robowits.16_stand_bulb").StandBulbEnv


@register_task("robowits/16_06-v0")
class StandBulbMut6Env(StandBulbEnv):
    def _config_to_env_args(self, config):
        # The thin rubber band (72 x 9 x 19 mm) makes contacts stiff enough that the
        # solver hits "Invalid constraint forces causing 'nan'" for some layouts at
        # the default 4 substeps; 10 still fails, 20 is stable (40 also works).
        args = super()._config_to_env_args(config)
        return args.model_copy(update={"scene_args": args.scene_args.model_copy(update={"substeps": 20})})

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return ("target_patch", "stabilizing_ring", "tall_object_with_curved_bottom", "rubber_band")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: target patch
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.50, 0.00, 0.761), euler=(0, 0, 0), radius=0.09, height=0.002, fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6, ior=1.5),
        )
        self._entities["target_patch"] = {"entity": _e}
        # Code Block: stabilizing ring (a plastic lid takes the ring's stabilizing role here)
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb", pattern_is_dir=False),
                scale=0.78,
                pos=(0.62, -0.08, 0.76 + 0.5 * (0.048 * 1.2)),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["stabilizing_ring"] = {"entity": _e}
        # Code Block: tall object with curved bottom
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4c5e9552-87c0-46e3-8e40-8c736fee4ff1/obj.glb", pattern_is_dir=False),
                scale=2.0,
                pos=(0.60, 0.2, 0.738),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                decimate=True,
                decimate_face_num=100,
                group_by_material=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(double_sided=True),
        )
        self._entities["tall_object_with_curved_bottom"] = {"entity": _e}
        # Code Block: rubber band
        # scale y 0.5 -> 1.0: the anisotropic squash left this ring 23.5mm thick across y
        # while COACD split it into wedges around the hole, so the collision pieces were
        # thin slivers -- the same instability class as 16_02's gold_ring (see the note
        # there), which cost 1/20 episodes here. Un-squashing y thickens every wedge and
        # leaves the z extent, and therefore the resting height, unchanged.
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/d853e8c3-9281-4c79-a0e5-fdaefca47dfa/obj.glb", pattern_is_dir=False),
                scale=(4.0, 1.0, 4.0),
                pos=(0.41, -0.18, 0.76115),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.8),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["rubber_band"] = {"entity": _e}
