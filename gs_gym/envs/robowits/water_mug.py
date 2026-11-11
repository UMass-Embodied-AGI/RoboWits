"""The shared water-filled mug.

Every task that holds water in a mug for pouring or spooning (02_04, 02_05, 02_08, 02_09, 30, 30_01, 30_05) builds
this one vessel with this one amount of liquid, authored at the same place inside it, so one settled particle cloud
(``dataset/robowits/particles/water_mug_particles.npz``) serves them all. Tasks keep their own mug pose, mass,
friction and fixedness; only what shapes the settled water is shared.
"""

from __future__ import annotations

import math

import genesis as gs

from gs_gym.common.utils import get_asset_path

MUG_ASSET = "blender_kit/3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb"
MUG_SCALE = 1.2
# Measured on the mesh at MUG_SCALE, in Genesis's z-up frame. The mesh origin is the centre of the box around body
# and handle, so the body's axis sits 19 mm from it, away from the handle. Inner floor 7.9 mm above the bottom,
# inner radius 36.5-40 mm, 110 mm tall (504 mL to the rim).
MUG_HALF_HEIGHT = 0.055
BODY_AXIS_XY = (-0.0001, -0.019)
INNER_FLOOR = 0.0079

# 400 mL, settling to ~65 mm deep (~65% of the inner depth, ~3.5 cm below the rim). Authored as a column on the
# body axis, 3 mm above the floor: radius 35 mm, so the outermost particles (sampled at cell centres, 7.5 mm in from
# the column's surface) end 1.5 mm inside the narrowest inner wall; 104 mm tall, so its top starts 5 mm above the
# rim and drops in as it settles. The mug holds 504 mL to the rim; a 30 mm column could not fit 400 mL inside it.
WATER_ML = 400.0
WATER_RADIUS = 0.035
WATER_HEIGHT = WATER_ML * 1e-6 / (math.pi * WATER_RADIUS**2)
WATER_FLOOR_GAP = 0.003
# Density 500 floats every cube the tasks float (150-200) and sinks task 30's metal cube (800). Stiffness 2e4 keeps
# the weakly compressible solver stable at 100 substeps (sound speed ~15 m/s, a third of a particle per substep);
# task 30's former 8e5 crossed 2.3 particles per substep and blew its water apart, and 8e4 leaks half of it out.
LIQUID = {"rho": 500.0, "stiffness": 2e4, "exponent": 7.0, "mu": 0.005, "gamma": 0.01, "sampler": "pbs"}
# Softer than the default 0.002: at the default, a loose mug and its water keep exciting each other on the table
# (v95 0.25-0.6 m/s after 5 s); at 0.01 the water rests as in a fixed mug (v95 0.01 m/s).
MUG_COUP_SOFTNESS = 0.01
WATER_COLOR = (0.55, 0.75, 1.0)

PARTICLES_FILE = "water_mug_particles.npz"


def water_center(mug_pos, yaw_deg: float) -> tuple[float, float, float]:
    """Where the water column is authored for a mug at ``mug_pos`` turned ``yaw_deg`` about z."""
    yaw = math.radians(yaw_deg)
    ax, ay = BODY_AXIS_XY
    dx = ax * math.cos(yaw) - ay * math.sin(yaw)
    dy = ax * math.sin(yaw) + ay * math.cos(yaw)
    bottom = float(mug_pos[2]) - MUG_HALF_HEIGHT
    return (float(mug_pos[0]) + dx, float(mug_pos[1]) + dy, bottom + INNER_FLOOR + WATER_FLOOR_GAP + WATER_HEIGHT / 2)


def add_water_mug(
    env,
    xy,
    yaw_deg: float = 0.0,
    *,
    mug_name: str = "water_mug",
    water_name: str = "water",
    table_z: float = 0.76,
    fixed: bool = False,
    rho: float = 200.0,
    friction: float | None = None,
    surface=None,
    coacd_options=None,
) -> None:
    """Add the mug standing on the table at ``xy`` with its water, as ``env._entities[mug_name]`` and ``[water_name]``.

    The mug's SDF runs at 128-256 like task 25's pitcher: the default resolution is too coarse for its 3-4 mm walls
    to hold SPH particles.
    """
    mug_pos = (float(xy[0]), float(xy[1]), table_z + MUG_HALF_HEIGHT)
    material_kwargs = {"rho": rho, "sdf_min_res": 128, "sdf_max_res": 256, "coup_softness": MUG_COUP_SOFTNESS}
    if friction is not None:
        material_kwargs["friction"] = friction
    mug = env._scene.scene.add_entity(
        gs.morphs.Mesh(
            coacd_options=coacd_options,
            file=get_asset_path(MUG_ASSET, pattern_is_dir=False),
            scale=MUG_SCALE,
            pos=mug_pos,
            euler=(0.0, 0.0, yaw_deg),
            fixed=fixed,
            collision=True,
            # Keep the real concave mesh. Genesis otherwise convexifies a movable mesh, and the convex pieces fill
            # part of the mug's hollow: the water never settles (5 s in, v95 0.3-1.1 m/s, up to 16% escaping),
            # where the same water in a fixed (never convexified) mug is at rest within a second (v95 0.01 m/s).
            convexify=False,
        ),
        material=gs.materials.Rigid(**material_kwargs),
        surface=surface if surface is not None else gs.surfaces.Smooth(double_sided=True),
    )
    env._entities[mug_name] = {"entity": mug}

    water = env._scene.scene.add_entity(
        gs.morphs.Cylinder(
            pos=water_center(mug_pos, yaw_deg),
            euler=(0.0, 0.0, 0.0),
            radius=WATER_RADIUS,
            height=WATER_HEIGHT,
            fixed=False,
            collision=True,
        ),
        material=gs.materials.SPH.Liquid(**LIQUID),
        surface=gs.surfaces.Default(color=WATER_COLOR, double_sided=True, vis_mode="recon"),
    )
    env._entities[water_name] = {"entity": water}


def mug_resting_quat(yaw_deg: float) -> list[float]:
    """The wxyz quaternion of the mug standing upright, turned ``yaw_deg`` about z."""
    h = math.radians(yaw_deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


__all__ = [
    "LIQUID",
    "MUG_HALF_HEIGHT",
    "PARTICLES_FILE",
    "WATER_ML",
    "add_water_mug",
    "mug_resting_quat",
    "water_center",
]
