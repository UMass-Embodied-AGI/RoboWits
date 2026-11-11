from importlib import import_module as _import

import genesis as gs

from gs_gym.common.utils import get_asset_path
from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

BallIntoBottleEnv = _import("gs_gym.envs.robowits.12_ball_into_bottle").BallIntoBottleEnv


# The seed's bottle, funnel, container and ball keep the seed's core region (x 0.25-0.65, y +-0.4) to themselves,
# as in the seed; added objects go in strips around it, still on the table (x 0.21-1.00, y +-0.69). At the seed's
# sizes the core region has no room left for extra objects: placement failed every attempt with them allowed in it.
LEFT_STRIP = {"x_min": 0.30, "x_max": 0.70, "y_min": 0.40, "y_max": 0.68}
RIGHT_STRIP = {"x_min": 0.30, "x_max": 0.70, "y_min": -0.68, "y_max": -0.40}
FAR_STRIP = {"x_min": 0.66, "x_max": 0.99, "y_min": -0.45, "y_max": 0.45}


@register_task("robowits/12_06-v0")
class BallIntoBottleMut6Env(BallIntoBottleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_reachable_areas.update(
            {
                "coffee_filter": LEFT_STRIP,
            }
        )

    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization (the seed's order: the wide fixed bottle first)."""
        return ("red_bottle", "funnel", ("ball_container", "small_ball"), "coffee_filter")

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        coacd_options = gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True
        )

        # Code Block: red bottle (the seed's size, fixed)
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=gs.options.CoacdOptions(
                    threshold=0.01, preprocess_resolution=150, max_convex_hull=50, decimate=True
                ),
                file=get_asset_path("blender_kit/ffb3fbe7-1355-465f-8750-475210d8c949/obj.glb", pattern_is_dir=False),
                scale=(1.4, 1.4, 1.0),
                pos=(0.48, 0.0, 0.88026),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(double_sided=True, color=(0.8, 0.2, 0.2)),
        )
        self._entities["red_bottle"] = {"entity": _e}
        # Code Block: funnel (the seed's size, lying on its side)
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("hf_assets/funnel.glb", pattern_is_dir=False),
                scale=(0.7, 0.7, 1.1),
                pos=(0.58, 0.0, 0.825),
                euler=(0.0, -90.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["funnel"] = {"entity": _e}
        # Code Block: coffee filter
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/4b5c4da2-2e9c-40aa-9542-711c163d9d80/obj.glb", pattern_is_dir=False),
                scale=1,
                pos=(0.72, -0.15, 0.76 + (0.0459 - (-0.0459)) * 1.3 / 2),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Rough(double_sided=True),
        )
        self._entities["coffee_filter"] = {"entity": _e}
        # Code Block: ball container (the seed's size)
        _e = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                coacd_options=coacd_options,
                file=get_asset_path("blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb", pattern_is_dir=False),
                scale=1.1,
                pos=(0.48, 0.15, 0.76 + 0.0459 * 1.1),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Glass(color=(0.8, 0.9, 1.0), opacity=0.3, double_sided=True),
        )
        self._entities["ball_container"] = {"entity": _e}
        # Code Block: small ball (the seed's size, dropped into the container)
        _e = self._scene.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.48, 0.15, 0.85),
                radius=0.005,
                fixed=False,
                collision=True,
            ),
            material=gs.materials.Rigid(rho=200.0),
            surface=gs.surfaces.Smooth(color=(0.2, 0.6, 0.9), double_sided=True),
        )
        self._entities["small_ball"] = {"entity": _e}
