"""Bimanual AgileX Piper (model read from ``assets/hf_assets/piper_description``).

Two 6-DOF arms side by side on one base, each with its stock two-finger gripper -- the real rigs' setup
(PIPER arm + PIPER gripper). The
single-arm URDF is composed into one bimanual URDF (joint and link names suffixed ``_R``/``_L``) so the pair
is one Genesis entity and one IK model, like Marvin.

- ``bimanual_piper``: ``piper_description.urdf``, gripper width 0-0.07 m
"""

from __future__ import annotations

import copy
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import numpy as np

from gs_gym.robots.bimanual import BBox, BimanualRobot
from gs_gym.robots.registry import register_robot

#: Private HF dataset holding the Piper description, pinned to one revision
PIPER_ASSETS_REPO = "Genesis-Intelligence/internal_assets"
PIPER_ASSETS_REVISION = "7fc5e31bd7f8667662617a117c9ad244aead00ee"
#: Lateral distance of each arm base from the robot base (the real bimanual Piper layout)
PIPER_ARM_SPACING_Y = 0.23
#: Height of the arm bases above the floor when they stand on a column (draws a visual-only mount);
#: 0 for bases bolted to the table top, the RoboWits default and the real rigs' setup
PIPER_MOUNT_HEIGHT = 0.0


def build_bimanual_piper_urdf(
    arm_urdf: Path,
    out_path: Path,
    spacing_y: float = PIPER_ARM_SPACING_Y,
    mount_height: float = PIPER_MOUNT_HEIGHT,
    convert_dae: bool = False,
    inertial_copies: dict[str, tuple[str, tuple[float, float, float]]] | None = None,
    keep_urdf_colors: bool = True,
) -> Path:
    """Compose two copies of a single-arm Piper URDF into one bimanual URDF.

    A root link ``piper_base`` carries the right arm at -spacing_y and the left at +spacing_y (the robot
    faces +x). Every link and joint name gets a ``_R``/``_L`` suffix, and mesh paths become absolute so
    the output can live anywhere. With ``mount_height`` > 0, ``piper_base`` also carries a visual-only mount -- a
    plate under both arm bases and a column down to the floor ``mount_height`` below -- so elevated arms do not
    float in renders; it has no collision geometry, so the physics is that of the bare arms. With ``convert_dae``,
    every Collada mesh is converted to OBJ next to the output: Genesis fails on some .dae files
    ("unknown or unsupported mesh file") and then loads that link with no geometry at all; the conversion keeps each
    mesh's per-part materials. Without ``keep_urdf_colors``, the arm links' flat URDF colors are dropped so the
    meshes' own materials show (the robot is loaded with prioritize_urdf_material=True). ``inertial_copies``
    maps a link to (another link, axis signs): it takes that link's ``<inertial>`` mirrored by the signs -- the
    centre of mass flipped per axis and the products of inertia with it -- for a link the arm URDF gives no usable
    mass but whose mirror twin it has.
    """
    arm = ET.parse(arm_urdf).getroot()
    arm_links = {link.get("name"): link for link in arm.findall("link")}
    for target, (source, signs) in (inertial_copies or {}).items():
        for old in arm_links[target].findall("inertial"):
            arm_links[target].remove(old)
        inertial = copy.deepcopy(arm_links[source].find("inertial"))
        origin = inertial.find("origin")
        if origin is not None:
            xyz = [float(v) * sign for v, sign in zip(origin.get("xyz", "0 0 0").split(), signs, strict=True)]
            origin.set("xyz", " ".join(f"{v:g}" for v in xyz))
        inertia = inertial.find("inertia")
        sx, sy, sz = signs
        for key, sign in (("ixy", sx * sy), ("ixz", sx * sz), ("iyz", sy * sz)):
            if inertia.get(key) is not None:
                inertia.set(key, f"{float(inertia.get(key)) * sign:g}")
        arm_links[target].insert(0, inertial)
    mesh_dir = arm_urdf.parent
    converted_dir = out_path.parent / f"{out_path.stem}_meshes"
    converted: dict[str, str] = {}

    def mesh_path(filename: str) -> str:
        path = os.path.normpath(mesh_dir / filename)  # normpath, not resolve(): HF cache symlinks are extension-less
        if not (convert_dae and path.lower().endswith(".dae")):
            return path
        if path not in converted:
            import trimesh

            converted_dir.mkdir(parents=True, exist_ok=True)
            obj = converted_dir / f"{len(converted):02d}_{Path(path).stem}.obj"
            # As a scene, so the parts keep their materials; each mesh gets its own .mtl
            trimesh.load(path).export(obj, file_type="obj", mtl_name=f"{obj.stem}.mtl")
            converted[path] = str(obj)
        return converted[path]

    root = ET.Element("robot", {"name": "bimanual_piper"})
    for material in arm.findall("material"):
        root.append(copy.deepcopy(material))
    base = ET.SubElement(root, "link", {"name": "piper_base"})
    plate_t = 0.02
    if mount_height > plate_t:
        gray = ET.SubElement(root, "material", {"name": "piper_mount_gray"})
        ET.SubElement(gray, "color", {"rgba": "0.28 0.29 0.31 1"})
        for size, xyz in (
            ((0.14, 2 * spacing_y + 0.14, plate_t), (0.0, 0.0, -plate_t / 2)),  # plate under both arm bases
            ((0.08, 0.08, mount_height - plate_t), (0.0, 0.0, -plate_t - (mount_height - plate_t) / 2)),  # column
        ):
            visual = ET.SubElement(base, "visual")
            ET.SubElement(visual, "origin", {"xyz": " ".join(f"{v:g}" for v in xyz), "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(visual, "geometry"), "box", {"size": " ".join(f"{v:g}" for v in size)})
            ET.SubElement(visual, "material", {"name": "piper_mount_gray"})

    arm_root = {j.find("parent").get("link") for j in arm.findall("joint")} - {
        j.find("child").get("link") for j in arm.findall("joint")
    }
    if len(arm_root) != 1:
        raise ValueError(f"{arm_urdf}: expected one root link, got {sorted(arm_root)}")
    (arm_root,) = arm_root

    for side, y in (("R", -spacing_y), ("L", spacing_y)):
        for link in arm.findall("link"):
            link = copy.deepcopy(link)
            link.set("name", f"{link.get('name')}_{side}")
            for mesh in link.iter("mesh"):
                mesh.set("filename", mesh_path(mesh.get("filename")))
            if not keep_urdf_colors:
                for visual in link.findall("visual"):
                    for material in visual.findall("material"):
                        visual.remove(material)
            root.append(link)
        for joint in arm.findall("joint"):
            joint = copy.deepcopy(joint)
            joint.set("name", f"{joint.get('name')}_{side}")
            for tag in ("parent", "child"):
                el = joint.find(tag)
                el.set("link", f"{el.get('link')}_{side}")
            mimic = joint.find("mimic")
            if mimic is not None:
                mimic.set("joint", f"{mimic.get('joint')}_{side}")
            root.append(joint)
        mount = ET.SubElement(root, "joint", {"name": f"piper_base_to_{arm_root}_{side}", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"0 {y} 0", "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "piper_base"})
        ET.SubElement(mount, "child", {"link": f"{arm_root}_{side}"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so parallel envs never read a half-written file
    with tempfile.NamedTemporaryFile("wb", dir=out_path.parent, suffix=".urdf", delete=False) as f:
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=True)
    os.replace(f.name, out_path)
    return out_path


#: The real robot's startup pose for both variants: each gripper
#: tip 0.25 m in front of its arm base and 0.05 m above it, pointing straight down
#: On the table-top mount that rests each tip 5 cm above the table inside the object area. Placement keeps movable
#: objects that reach the gripper clear of it (REST_FOOTPRINT), but fixed entities get no robot clearance, so a
#: task can drop one under a resting gripper (task 13's pot does): such episodes start with the arm pushed.
PIPER_STARTUP_TIP = (0.25, 0.0, 0.05)


class _BimanualPiperBase(BimanualRobot):
    """Piper's joint names, gripper frames, controller gains, the wrist camera."""

    #: Description directory in internal_assets, and the arm URDF inside it
    DESCRIPTION: ClassVar[str]
    ARM_URDF: ClassVar[str]
    #: Convert the arm's Collada meshes to OBJ when composing (see build_bimanual_piper_urdf)
    CONVERT_DAE: ClassVar[bool] = False
    #: False: drop the arm's flat URDF colors so the meshes' own materials show (see build_bimanual_piper_urdf)
    KEEP_URDF_COLORS: ClassVar[bool] = True
    #: Links that take another link's <inertial>, mirrored by axis signs, when composing (see build_bimanual_piper_urdf)
    INERTIAL_COPIES: ClassVar[dict[str, tuple[str, tuple[float, float, float]]]] = {}

    EE_LINK_RIGHT: ClassVar[str] = "gripper_tip_R"
    EE_LINK_LEFT: ClassVar[str] = "gripper_tip_L"
    GRIPPER_BASE_LINK_RIGHT: ClassVar[str] = "gripper_base_R"
    GRIPPER_BASE_LINK_LEFT: ClassVar[str] = "gripper_base_L"

    #: The Piper wrist-camera mount is unmeasured, so this reuses the Pika + D405 offset measured on Marvin,
    #: attached to the gripper base. Both Piper gripper_base frames point +z at the
    #: tip like Pika's, so the camera looks along the fingers; the real rigs' wrist mounts are unmeasured.
    WRIST_CAMERA_OFFSET_T: ClassVar[np.ndarray] = np.array(
        [
            [-1, 0, 0, 0.009],
            [0, 1, 0, 0.04931],
            [0, 0, -1, 0.06159 - 0.0037 + 0.002],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    #: The real robot's 6-DOF controller defaults
    KP: ClassVar[tuple[float, ...]] = (800, 800, 800, 800, 800, 800, 200, 200)
    KV: ClassVar[tuple[float, ...]] = (80, 80, 80, 80, 80, 80, 20, 20)
    FORCE_RANGE: ClassVar[tuple[float, ...]] = (87, 87, 87, 87, 12, 12, 100, 100)

    #: joint7 opens over [0, FINGER_MAX], joint8 over [-FINGER_MAX, 0]
    FINGER_SIGNS: ClassVar[tuple[float, float]] = (1.0, -1.0)
    #: Genesis collides each robot link as one convex hull. The L-shaped fingers' hulls collide with each other and
    #: with the wrist (Piper's gripper stopped at 2.7 cm),
    #: so decompose non-convex links at Genesis's threshold for ordinary objects.
    URDF_MORPH_OPTIONS: ClassVar[dict] = {"decompose_robot_error_threshold": 0.15}

    @classmethod
    def default_urdf_path(cls) -> str:
        """Compose the bimanual URDF in the cache from the single-arm description in ``assets/hf_assets/``."""
        from gs_gym.common.utils.asset_utils import get_asset_path

        local = Path(get_asset_path(f"hf_assets/{cls.DESCRIPTION}")).parent
        # Named after the arm URDF it is built from, so a different model can never reuse a stale file
        stem = Path(cls.ARM_URDF).stem
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gs_gym" / "bimanual_piper"
        out = (
            cache
            / f"bimanual_{stem}_{PIPER_ASSETS_REVISION[:8]}_y{PIPER_ARM_SPACING_Y:g}_mount{PIPER_MOUNT_HEIGHT:g}.urdf"
        )
        if not out.exists():
            build_bimanual_piper_urdf(
                local / cls.DESCRIPTION / "urdf" / cls.ARM_URDF,
                out,
                convert_dae=cls.CONVERT_DAE,
                inertial_copies=cls.INERTIAL_COPIES,
                keep_urdf_colors=cls.KEEP_URDF_COLORS,
            )
        return str(out)


@register_robot("bimanual_piper")
class BimanualPiperRobot(_BimanualPiperBase):
    """Bimanual Piper with its stock gripper. JOINT_DIM = 14."""

    DESCRIPTION: ClassVar[str] = "piper_description"
    ARM_URDF: ClassVar[str] = "piper_description.urdf"

    RIGHT_ARM_JOINTS: ClassVar[list[str]] = [f"joint{i}_R" for i in range(1, 7)]
    LEFT_ARM_JOINTS: ClassVar[list[str]] = [f"joint{i}_L" for i in range(1, 7)]
    RIGHT_GRIPPER_JOINTS: ClassVar[list[str]] = ["joint7_R", "joint8_R"]
    LEFT_GRIPPER_JOINTS: ClassVar[list[str]] = ["joint7_L", "joint8_L"]

    #: Joint limits from the URDF (joint1-6); replaced by the built URDF's own
    ARM_LOWER: ClassVar[tuple[float, ...]] = (-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944)
    ARM_UPPER: ClassVar[tuple[float, ...]] = (2.168, 3.14, 0.0, 1.745, 1.22, 2.0944)
    #: Gripper width 0-0.07 m, as the real client (PiperType.PIPER)
    FINGER_MAX: ClassVar[float] = 0.035

    #: The real client's startup pose (PIPER_STARTUP_TIP, pointing straight down), solved by IK for this arm:
    #: exact, every joint >= 0.28 rad inside its limits
    RESET_RIGHT_ARM: ClassVar[tuple[float, ...]] = (0.0, 1.5547, -0.8411, 0.0, 0.9444, 0.0)
    RESET_LEFT_ARM: ClassVar[tuple[float, ...]] = (0.0, 1.5547, -0.8411, 0.0, 0.9444, 0.0)
    RESET_FINGER: ClassVar[float] = 0.035  # fully open, as the real client starts

    #: Link centers at the reset pose (pinocchio FK) with the base at (0.20, 0, 0.76), the RoboWits overlay's
    #: table-top mount; forearm = link3 through the gripper, full = from base_link
    REST_FOOTPRINT: ClassVar[dict[str, BBox]] = {
        "forearm_right": {"x": (0.234, 0.450), "y": (-0.265, -0.195), "z": (0.810, 1.166)},
        "forearm_left": {"x": (0.234, 0.450), "y": (0.195, 0.265), "z": (0.810, 1.166)},
        "full_right": {"x": (0.200, 0.450), "y": (-0.265, -0.195), "z": (0.760, 1.166)},
        "full_left": {"x": (0.200, 0.450), "y": (0.195, 0.265), "z": (0.760, 1.166)},
    }
