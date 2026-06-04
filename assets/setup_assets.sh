#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_REPO="XHRlyb2001/RoboWits_assets"
HF_ASSETS_DIR="$SCRIPT_DIR/hf_assets"
ASSETS_DIR="$SCRIPT_DIR"

usage() {
    echo "Usage: $0 --api-key <BLENDERKIT_KEY> [--skip-hf] [--skip-blenderkit]"
    exit 1
}

API_KEY=""
SKIP_HF=0
SKIP_BLENDERKIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)         API_KEY="$2";    shift 2 ;;
        --skip-hf)         SKIP_HF=1;       shift ;;
        --skip-blenderkit) SKIP_BLENDERKIT=1; shift ;;
        *) usage ;;
    esac
done

[[ -z "$API_KEY" && "$SKIP_BLENDERKIT" -eq 0 ]] && { echo "Error: --api-key is required"; usage; }

if [[ "$SKIP_HF" -eq 0 ]]; then
    echo "=== Downloading HuggingFace assets ==="
    mkdir -p "$HF_ASSETS_DIR"
    hf download "$HF_REPO" \
        --repo-type dataset \
        --local-dir "$HF_ASSETS_DIR"
fi

if [[ "$SKIP_BLENDERKIT" -eq 0 ]]; then
    echo "=== Downloading BlenderKit assets ==="
    python "$SCRIPT_DIR/download_blenderkit_asset.py" \
        --api-key "$API_KEY" \
        --assets-dir "$ASSETS_DIR"
fi

echo "All assets ready."
