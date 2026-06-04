"""Asset utilities for managing local assets.

Asset Search Order:
1. Local assets/ directory at the project root
2. GS_GYM_ASSET_PATHS environment variable (colon-separated paths)
"""

from __future__ import annotations

import os
from pathlib import Path

# Local assets directory: <project_root>/assets/
# asset_utils.py lives at gs_gym/common/utils/asset_utils.py → 4 levels up = project root
_LOCAL_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


def get_asset_path(
    asset_name: str,
    **_kwargs,
) -> str:
    """Get path to a local asset.

    Args:
        asset_name: Relative path within the assets directory (e.g., "uuid/obj.glb")

    Returns:
        Absolute path to the local asset file/directory
    """
    # 1. Check local assets/ directory at project root
    local_path = _LOCAL_ASSETS_DIR / asset_name
    if local_path.exists():
        return str(local_path)

    # 2. Check environment variable for local override
    local_assets_dir = os.environ.get("GS_GYM_ASSET_PATHS", "")
    if local_assets_dir:
        for dir_path in local_assets_dir.split(":"):
            dir_path = dir_path.strip()
            if dir_path and os.path.exists(os.path.join(dir_path, asset_name)):
                return os.path.join(dir_path, asset_name)

    raise FileNotFoundError(
        f"Asset '{asset_name}' not found in {_LOCAL_ASSETS_DIR} or GS_GYM_ASSET_PATHS."
    )


def get_table_path() -> str:
    return get_asset_path("hf_assets/work_table.glb")


def get_worktable_texture_normal_path() -> str:
    return get_asset_path("hf_assets/worktable_texture/grained black plastic_Normal.jpg")


def get_worktable_texture_roughness_path() -> str:
    return get_asset_path("hf_assets/worktable_texture/grained black plastic_Roughness.jpg")


def get_bimanual_marvin_urdf_path() -> str:
    asset_dir = get_asset_path("hf_assets/marvin_bimanual")
    return f"{asset_dir}/urdf/marvin_pika.urdf"
