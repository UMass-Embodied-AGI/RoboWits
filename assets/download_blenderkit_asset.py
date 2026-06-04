#!/usr/bin/env python3
"""
Standalone script to download and preprocess BlenderKit assets by asset base ID.

Optional:  blender on PATH (for GLB re-centering)

Usage:
    python download_blenderkit_asset.py [--free] \\
        [--api-key KEY] [--assets-dir assets/models]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from urllib.parse import urlparse

import requests

BLENDER_PREPROCESS_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_preprocess.py")


class SubscriptionRequiredError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("download_assets")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def download_file(url: str, dest: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def fetch_blenderkit_asset(asset_id: str, api_key: str = None) -> dict:
    url = f"https://www.blenderkit.com/api/v1/search/?query=asset_base_id:{asset_id}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = json.loads(requests.get(url, headers=headers).content.decode())
    if not data.get('results'):
        raise ValueError(f"No BlenderKit results for asset {asset_id}.")
    return data['results'][0]


def download_glb(asset_id: str, api_key: str, dest_dir: str,
                 resolution: str = '1k', overwrite: bool = False,
                 logger: logging.Logger = None) -> str:
    """Download asset as GLB if available, otherwise fall back to blend. Returns raw downloaded path."""
    dest_file = os.path.join(dest_dir, "obj.glb")
    if os.path.exists(dest_file) and not overwrite:
        if logger:
            logger.info(f"{asset_id}: already downloaded, skipping.")
        return dest_file

    os.makedirs(dest_dir, exist_ok=True)

    obj = fetch_blenderkit_asset(asset_id, api_key)
    if not obj.get('canDownload', False):
        raise SubscriptionRequiredError(
            f"{asset_id} ({obj.get('name', '')!r}) requires a full-plan subscription."
        )
    files = obj['files']

    gltf_file = next((f for f in files if f.get('fileType') == 'gltf'), None)
    blend_file = next((f for f in files if f.get('fileType') == 'blend'), None)
    if gltf_file is not None:
        file_url = gltf_file['downloadUrl']
    elif blend_file is not None:
        if logger:
            logger.info(f"{asset_id}: no gltf variant available, falling back to blend.")
        file_url = blend_file['downloadUrl']
    else:
        raise ValueError(f"{asset_id}: no gltf or blend file type available in API response.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    dl = json.loads(
        requests.get(f"{file_url}?scene_uuid={uuid.uuid4()}", headers=headers).content.decode()
    )
    suffix = urlparse(dl['filePath']).path.split('/')[-1].split('.')[-1]
    raw_file = os.path.join(dest_dir, f"obj.{suffix}")
    download_file(dl['filePath'], raw_file)

    if suffix != 'blend':
        with open(raw_file, 'rb') as f:
            magic = f.read(4)
        if magic != b'glTF':
            os.remove(raw_file)
            raise ValueError(
                f"{asset_id}: downloaded file is not a GLB (magic={magic!r}, suffix=.{suffix})."
            )
        if raw_file != dest_file:
            os.rename(raw_file, dest_file)
        raw_file = dest_file

    if logger:
        logger.info(f"{asset_id}: downloaded -> {raw_file}")
    return raw_file


def preprocess_glb(file_path: str, logger: logging.Logger = None):
    """Re-center mesh and re-export as GLB using Blender. Accepts .glb or .blend input."""
    try:
        if logger:
            logger.info(f"Running blender re-centering on {file_path} ...")
        subprocess.run(
            ["blender", "--background", "--python", BLENDER_PREPROCESS_SCRIPT, "--", "-f", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        if logger:
            logger.info(f"Blender preprocessing done: {file_path}")
    except FileNotFoundError:
        if logger:
            logger.warning("'blender' not found on PATH — skipping GLB re-centering.")
    except subprocess.CalledProcessError as e:
        if logger:
            logger.warning(f"Blender preprocessing failed: {e.stderr.decode()[:400]}")


# ---------------------------------------------------------------------------
# Per-asset pipeline
# ---------------------------------------------------------------------------

def process_asset(asset_id: str, api_key: str, assets_dir: str,
                  logger: logging.Logger, overwrite: bool = False) -> bool:
    try:
        logger.info(f"Fetching metadata for {asset_id} ...")
        obj = fetch_blenderkit_asset(asset_id, api_key)
        dest_dir = os.path.join(assets_dir, 'blender_kit', asset_id)
        raw_file = download_glb(asset_id, api_key, dest_dir, resolution='1k', overwrite=overwrite, logger=logger)
        if overwrite or not os.path.exists(os.path.join(dest_dir, 'obj.glb')):
            preprocess_glb(raw_file, logger)
        logger.info(f"Done: {asset_id} ({obj.get('name', '')})")
        return True

    except SubscriptionRequiredError as e:
        logger.warning(f"Skipping {asset_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to process {asset_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ASSET_METADATA = os.path.join(os.path.dirname(__file__), "metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess BlenderKit assets.")
    parser.add_argument("--api-key", required=True,
                        help="BlenderKit API key.")
    parser.add_argument("--assets-dir", default="assets/models",
                        help="Root directory to store downloaded assets (default: assets/models).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download and reprocess assets that already exist.")
    args = parser.parse_args()

    with open(ASSET_METADATA) as f:
        asset_metadata = json.load(f)
    asset_ids = asset_metadata["free"] + asset_metadata["full_plan"]

    api_key = args.api_key

    os.makedirs(args.assets_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.assets_dir, "download_assets.log"))

    failed = []
    skipped = []
    for asset_id in asset_ids:
        logger.info(f"\n{'='*60}\nProcessing {asset_id}\n{'='*60}")
        result = process_asset(asset_id, api_key, args.assets_dir, logger, overwrite=args.overwrite)
        if result is None:
            skipped.append(asset_id)
        elif not result:
            failed.append(asset_id)

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} asset(s) that require a full-plan subscription: {skipped}\n"
            "Re-run with --free to suppress these warnings."
        )
    if failed:
        logger.error(f"Failed ({len(failed)}): {failed}")
        sys.exit(1)
    else:
        logger.info(f"Done: {len(asset_ids) - len(skipped)} asset(s) processed, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()
