#!/usr/bin/env bash
# End-to-end check that the release branch is self-contained: install assets from the
# manifest, then build every task the published benchmark tables reference.
#
# Needs a BlenderKit API key, since release downloads its assets rather than pulling a
# prebuilt bundle. Run it on a GPU node -- the SPH/MPM variants route themselves to CUDA.
#
#   BLENDERKIT_API_KEY=... bash verify_release.sh            # assets + envs
#   BLENDERKIT_API_KEY=... SKIP_ASSETS=1 bash verify_release.sh   # envs only
set -euo pipefail

SKIP_ASSETS=${SKIP_ASSETS:-0}
TASKS=${TASKS:-""}   # optional space-separated subset, e.g. TASKS="robowits/02_08-v0"

if [ "${SKIP_ASSETS}" = "0" ]; then
    : "${BLENDERKIT_API_KEY:?set BLENDERKIT_API_KEY, or SKIP_ASSETS=1 if assets are already installed}"
    bash assets/setup_assets.sh --api-key "${BLENDERKIT_API_KEY}"
fi

TASKS="${TASKS}" python scripts/release_check/build_every_task.py
