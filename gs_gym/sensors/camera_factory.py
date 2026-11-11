"""Camera-sensor factory shared by every environment that adds cameras to a scene.

Genesis exposes each renderer as a distinct camera-options class passed to
``scene.add_sensor()``. They share the placement fields (``pos`` / ``lookat`` / ``fov`` /
``res`` / ``entity_idx`` / ``link_idx_local`` / ``offset_T``), so callers can stay
renderer-agnostic and let this module supply the renderer-specific extras.

    "rasterizer" - OpenGL rasterizer. Works on CUDA and Vulkan. Default everywhere:
                   it is what the training and eval observation pipelines are tuned on.
    "batch"      - Madrona batch renderer. CUDA only, fastest for many parallel envs.
    "nyx"        - Nyx GPU path tracer (``gs-nyx`` / ``gs-nyx-plugin``). Photorealistic;
                   intended for offline stills and videos, not for rollout observations.
                   Costs ~0.33 s/frame more than the rasterizer at 1696x960. That cost is
                   fixed per-frame overhead, not sample count: 8 -> 512 spp moves wall
                   clock by under 5%, so raising ``spp`` buys quality nearly for free.
"""

from __future__ import annotations

from typing import Any, Literal

CameraType = Literal["rasterizer", "batch", "nyx", "luisa"]

#: Samples per pixel for Nyx. Enough to be clean once the denoiser runs. Matches what
#: mcaps uses for its Nyx cameras (spp=32, mode="pt_fast", denoise=True).
NYX_DEFAULT_SPP = 32

#: Samples per pixel for LuisaRender (denoised).
LUISA_DEFAULT_SPP = 64

#: Nyx render mode. FastPathTracer is the usable-latency path tracer; RefPathTracer is
#: the ground-truth reference.
NYX_DEFAULT_RENDER_MODE = "FastPathTracer"

# ── Presentation look ─────────────────────────────────────────────────────────────────────
# Camera placement, env-map brightness and light rig are ported from the Genesis IPC demo
# (Genesis_IPC_demo, examples/IPC_Solver/Robowits/replay_robowits.py), which is the reference
# for how RoboWits scenes should look. Keep these in sync with that script.

#: Three-quarter view from front-right and above. Reads better than a straight-on shot.
GALLERY_CAM_POS = (1.5122, -0.767, 1.8931)
GALLERY_CAM_LOOKAT = (0.838, -0.3497, 1.2837)
GALLERY_CAM_FOV = 40


#: Per-task env map, so each scene gets a distinct backdrop. Tasks not listed fall back to
#: DEFAULT_ENV_MAP.
TASK_ENVMAP: dict[str, str] = {
    "02": "brown_photostudio_07_4k.exr",
    "03": "machine_shop_01_4k.exr",
    "04": "poly_haven_studio_4k.exr",
    "06": "art_studio_4k.exr",
    "09": "lythwood_lounge_4k.exr",
    "11": "empty_warehouse_01_4k.exr",
    "13": "brown_photostudio_05_4k.exr",
    "14": "blue_photo_studio_4k.exr",
    "16": "gear_store_4k.exr",
    "17": "small_hangar_01_4k.exr",
    "18": "abandoned_factory_canteen_02_4k.exr",
    "19": "ballroom_4k.exr",
    "24": "machine_shop_03_4k.exr",
    "27": "abandoned_hall_01_4k.exr",
    "28": "marry_hall_4k.exr",
    "29": "brown_photostudio_06_4k.exr",
}

DEFAULT_ENV_MAP = "brown_photostudio_07_4k.exr"

#: Env-map brightness. The light rig below is balanced against this value, so changing one
#: without the other will over- or under-expose the scene.
DEFAULT_ENV_MAP_MULTIPLIER = 0.5

#: Sphere/point light rig, in Nyx intensity units (the demo's Luisa values already scaled by
#: its env-light damping and Luisa->Nyx conversion).
NYX_LIGHTS: tuple[dict[str, Any], ...] = (
    # Key: above-left, warm, casts the main shadows across the table
    {"type": "point", "pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 1.5},
    # Fill: right side, cooler, large and soft
    {"type": "point", "pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 0.03},
    # Rim: behind the scene, cool and hard, separates the robot from the background
    {"type": "point", "pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 4.5},
)


def env_map_for_task(task_id: str | None) -> str:
    """Env map filename for a task id like "02" (falls back to DEFAULT_ENV_MAP)."""
    return TASK_ENVMAP.get((task_id or "").zfill(2), DEFAULT_ENV_MAP)


def env_map_path(texture: str | None = None) -> str:
    """Local path of an env-map texture: an absolute path, a bare .exr name resolved against
    ``assets/hf_assets/envmap/``, or a path inside ``assets/``. Defaults to DEFAULT_ENV_MAP."""
    import os

    from gs_gym.common.utils.asset_utils import get_asset_path

    path = texture or DEFAULT_ENV_MAP
    if os.path.isabs(path) or os.path.exists(path):
        return path
    if "/" in path:
        return get_asset_path(path)
    return get_asset_path(f"hf_assets/envmap/{path}")


def make_luisa_renderer(env_map: str | None = None, env_multiplier: float | None = None, lights: tuple = ()):
    """The scene-level LuisaRender renderer that ``luisa`` cameras need, lit by the same panorama as Nyx.

    Genesis imports LuisaRenderPy from genesis/ext/LuisaRender/build/bin, which the plain submodule checkout does not
    have: link a LuisaRender build of the commit Genesis pins at third_party/genesis/genesis/ext/LuisaRender/build.

    ``lights``: Genesis SphereLights. None by default, so the panorama alone lights the scene: Genesis's own default
    (a radius-4 sphere light 10 m up at intensity 10) saturates the white table and arms however dim the sky is.
    """
    import genesis as gs

    return gs.renderers.RayTracer(
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ImageTexture(
                image_path=env_map_path(env_map),
                encoding="linear",
                image_color=DEFAULT_ENV_MAP_MULTIPLIER if env_multiplier is None else env_multiplier,
            ),
        ),
        env_radius=15.0,
        lights=list(lights),
    )


def make_env_map(
    texture: str | None = None, multiplier: float = DEFAULT_ENV_MAP_MULTIPLIER, rotation: float | None = None
):
    """Build a Nyx environment-map asset, downloading the texture if needed.

    Args:
        texture: An absolute path, a bare .exr name resolved against the env-map repo, or a
            path inside the gs-gym-assets repo. Defaults to DEFAULT_ENV_MAP.
        multiplier: Brightness scale. See DEFAULT_ENV_MAP_MULTIPLIER on keeping this balanced
            against NYX_LIGHTS.
        rotation: Optional yaw applied to the map, to steer where the light comes from.
    """
    import gs_nyx.nyx_py_sdk as nps

    path = env_map_path(texture)
    env_map = nps.EnvironmentMapAsset()
    env_map.texture = path
    env_map.layout = nps.EEnvMapLayout.LongLat
    env_map.multiplier = multiplier
    if rotation is not None:
        env_map.rotation = rotation
    return env_map


def _nyx_extras(spp: int | None, render_mode: str | None, env_maps: tuple | None) -> dict[str, Any]:
    import gs_nyx.nyx_py_renderer as npr

    mode_name = render_mode or NYX_DEFAULT_RENDER_MODE
    try:
        mode = getattr(npr.ERenderMode, mode_name)
    except AttributeError as exc:
        available = [m for m in dir(npr.ERenderMode) if not m.startswith("_")]
        raise ValueError(f"Unknown Nyx render_mode '{mode_name}'. Available: {available}") from exc

    extras: dict[str, Any] = {
        "spp": NYX_DEFAULT_SPP if spp is None else spp,
        "render_mode": mode,
        "denoise": True,
    }
    # Default to the standard environment map; pass env_maps=() to opt out and light the scene
    # from the light rig only.
    extras["env_maps"] = tuple(env_maps) if env_maps is not None else (make_env_map(),)
    extras["lights"] = list(NYX_LIGHTS)
    return extras


def make_camera_options(
    camera_type: CameraType = "rasterizer",
    *,
    spp: int | None = None,
    render_mode: str | None = None,
    env_maps: tuple | None = None,
    env_map: str | None = None,
    env_rotation: float | None = None,
    env_multiplier: float | None = None,
    **placement,
) -> Any:
    """Build camera options for ``scene.add_sensor()``.

    Args:
        camera_type: Which renderer backs the camera.
        spp: Nyx samples per pixel. Ignored by the other renderers.
        render_mode: Nyx ``ERenderMode`` name. Ignored by the other renderers.
        env_maps: Nyx environment maps (``gs_nyx.nyx_py_sdk.EnvironmentMapAsset``). Defaults to
            DEFAULT_ENV_MAP; pass an empty tuple to light the scene from its own lights only.
        env_map: Convenience alternative to ``env_maps`` — a texture path (or a path inside the
            gs-gym-assets repo) to build a single environment map from.
        env_rotation: Yaw (degrees) applied to the environment map, to swing a chosen part of
            the panorama behind the subject.
        env_multiplier: Override the env-map brightness (see DEFAULT_ENV_MAP_MULTIPLIER).
        **placement: Renderer-agnostic fields (``pos``, ``lookat``, ``fov``, ``res``,
            ``entity_idx``, ``link_idx_local``, ``offset_T``, ``up``, ...).
    """
    if camera_type == "rasterizer":
        from genesis.options.sensors import RasterizerCameraOptions

        return RasterizerCameraOptions(**placement)

    if camera_type == "batch":
        from genesis.options.sensors import BatchRendererCameraOptions

        return BatchRendererCameraOptions(**placement)

    if camera_type == "nyx":
        try:
            from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
        except ImportError as exc:  # pragma: no cover - depends on optional wheels
            raise ImportError(
                "The Nyx renderer needs 'gs-nyx' and 'gs-nyx-plugin', which ship wheels for "
                "Linux x86_64 and Windows only. Run 'uv sync' on a supported platform."
            ) from exc

        if env_maps is None and (env_map is not None or env_rotation is not None or env_multiplier is not None):
            env_maps = (
                make_env_map(
                    env_map,
                    multiplier=DEFAULT_ENV_MAP_MULTIPLIER if env_multiplier is None else env_multiplier,
                    rotation=env_rotation,
                ),
            )
        # NyxCameraOptions defaults to a 0.1 m near plane against the rasterizer's 0.01 m. The wrist cameras sit on
        # the gripper, so at 0.1 m Nyx clipped away all but the fingertips, and sliced open anything held close
        # (a grasped mug's rim showed its black inner faces). Match the rasterizer unless the caller sets one.
        placement.setdefault("near", 0.01)
        return NyxCameraOptions(**placement, **_nyx_extras(spp, render_mode, env_maps))

    if camera_type == "luisa":
        # LuisaRender path tracing. The scene must be created with make_luisa_renderer(), which carries the sky.
        from genesis.options.sensors import RaytracerCameraOptions

        return RaytracerCameraOptions(**placement, spp=spp or LUISA_DEFAULT_SPP, denoise=True)

    raise ValueError(f"Unknown camera_type '{camera_type}'. Available: 'rasterizer', 'batch', 'nyx', 'luisa'")
