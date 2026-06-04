from importlib import import_module as _import

import genesis as gs

from gs_gym.envs.registry import register_task
from gs_gym.envs.robowits.robowits import PlacementGroups

DominosEnv = _import("gs_gym.envs.robowits.06_dominos").DominosEnv


@register_task("robowits/06_05-v0")
class DominosMut5Env(DominosEnv):
    @property
    def placement_groups(self) -> PlacementGroups | None:
        """Define placement groups for random initialization."""
        return (
            "white block 4",
            ("white block 3", "white block 2", "white block 1", "red block", "target area"),
            "plastic straw",
            "kitchen sponge",
        )

    def _add_custom_entities(self) -> None:
        """Add custom entities using genesis APIs."""
        # Code Block: white block 4
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.68, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 4"] = {"entity": _e}
        # Code Block: white block 3
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.735, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 3"] = {"entity": _e}
        # Code Block: white block 2
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.795, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 2"] = {"entity": _e}
        # Code Block: white block 1
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.855, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        )
        self._entities["white block 1"] = {"entity": _e}
        # Code Block: red block
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.915, 0.00, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=500.0, friction=1.0),
            surface=gs.surfaces.Smooth(color=(0.9, 0.1, 0.1), double_sided=False),
        )
        self._entities["red block"] = {"entity": _e}
        # Code Block: target area
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.960, 0.00, 0.761), euler=(0, 0, 0), size=(0.06, 0.10, 0.002), fixed=True, collision=False
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.5),
            surface=gs.surfaces.Smooth(color=(0.1, 0.7, 0.2), double_sided=True),
        )
        self._entities["target area"] = {"entity": _e}
        # Code Block: plastic straw
        _e = self._scene.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.66, 0.07, 0.763), euler=(90, 0, 0), radius=0.003, height=0.20, fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=200.0, friction=0.4),
            surface=gs.surfaces.Smooth(color=(0.9, 0.9, 0.9), double_sided=False),
        )
        self._entities["plastic straw"] = {"entity": _e}
        # Code Block: kitchen sponge
        _e = self._scene.scene.add_entity(
            gs.morphs.Box(
                pos=(0.50, 0.12, 0.775), euler=(0, 0, 0), size=(0.11, 0.07, 0.03), fixed=False, collision=True
            ),
            material=gs.materials.Rigid(rho=120.0, friction=0.6),
            surface=gs.surfaces.Smooth(color=(0.95, 0.85, 0.1), double_sided=False),
        )
        self._entities["kitchen sponge"] = {"entity": _e}
