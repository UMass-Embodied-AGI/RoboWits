#!/bin/bash
# Path-trace every task's first recording (mode pkl: robot and objects posed from the mcap).
# Needs a GPU: Nyx is CUDA only.
set -e

PKL_DIR=${PKL_DIR:-/mnt/data/hongxin/RoboWitsDataset/pkl_converted_0301}
TASKS=${TASKS:-$(ls -d "${PKL_DIR}"/*_SUC_raw 2>/dev/null | sed 's|.*/||;s|_SUC_raw||')}
OUT_DIR=${OUT_DIR:-tmp/replay_nyx_demos}
DEVICE=${DEVICE:-cuda}  # Nyx is CUDA only
SPP=${SPP:-}
MAX_STEPS=${MAX_STEPS:-}
GALLERY=${GALLERY:-}
RENDER_MODE=${RENDER_MODE:-}
# Drop the ground plane by default: it is the checkerboard that hides the environment map, and
# the env map is the whole point of a path-traced demo. GROUND=1 puts it back.
GROUND=${GROUND:-0}
FORCE=${FORCE:-0}

mkdir -p "${OUT_DIR}"

for TASK in ${TASKS}; do
    PKL=$(ls "${PKL_DIR}/${TASK}_SUC_raw"/*.pkl 2>/dev/null | head -1)
    if [ -z "${PKL}" ]; then
        echo "Skipping ${TASK}: no recording in ${PKL_DIR}/${TASK}_SUC_raw"
        continue
    fi
    if [ -f "${OUT_DIR}/${TASK}.mp4" ] && [ "${FORCE}" = "0" ]; then
        echo "Skipping ${TASK}: ${OUT_DIR}/${TASK}.mp4 exists (use FORCE=1 to redo)"
        continue
    fi
    echo "Rendering ${TASK}: $(basename "${PKL}")"
    python scripts/robowits/data/replay_lerobot.py "${PKL}" \
        --task "${TASK}" --mode pkl --camera_type nyx --device "${DEVICE}" --no_csv \
        ${SPP:+--spp "${SPP}"} ${MAX_STEPS:+--max_steps "${MAX_STEPS}"} \
        ${GALLERY:+--gallery} ${RENDER_MODE:+--render_mode "${RENDER_MODE}"} \
        $([ "${GROUND}" = "0" ] && echo --no_ground) || continue
    # replay_lerobot.py names its own output; collect it under one flat directory.
    mv "tmp/replay_demo/${TASK}/0_pkl_nyx${GALLERY:+_gallery}_clean.mp4" "${OUT_DIR}/${TASK}.mp4"
    echo "  -> ${OUT_DIR}/${TASK}.mp4"
done
