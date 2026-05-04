#!/bin/bash
# Downloads only the files needed for pose estimation training:
#   - *.mp4          (RGB video frames, ~3.3GB)
#   - *_info.mat     (joints2D labels + metadata, ~3.8GB)
#
# Usage (run inside an srun session or via download_job.sh):
#   bash download_surreal.sh [output_dir] [username] [password]
#
# Credentials fall back to $SURREAL_USER / $SURREAL_PASS env vars.
# Set those in your ~/.bashrc on OSCAR to avoid passing them on the command line.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR="${1:-${HOME}/scratch/virtual-tryon/surreal_data}"
USERNAME="${2:-${SURREAL_USER:-}}"
PASSWORD="${3:-${SURREAL_PASS:-}}"

if [[ -z "$USERNAME" || -z "$PASSWORD" ]]; then
    echo "Error: SURREAL credentials not set."
    echo "Either pass them as arguments or export SURREAL_USER and SURREAL_PASS in ~/.bashrc"
    exit 1
fi

mkdir -p "$OUTDIR"
echo "Downloading SURREAL data to: $OUTDIR"
echo "Modalities: .mp4 and _info.mat only (skipping depth/segm/flow)"

for dataset in 'cmu'; do
    for setname in 'train' 'val'; do
        for modality in '.mp4' '_info.mat'; do # download only RGB input frames and info.mat and gender/pose
            FILE_LIST="${SCRIPT_DIR}/files/files_${dataset}_${setname}${modality}.txt"

            if [[ ! -f "$FILE_LIST" ]]; then
                echo "Warning: file list not found: $FILE_LIST — skipping"
                continue
            fi

            echo "  -> ${dataset}/${setname}${modality}  ($(wc -l < "$FILE_LIST") files)"
            wget \
                --user="${USERNAME}" \
                --password="${PASSWORD}" \
                --continue \
                --mirror \
                --quiet \
                --show-progress \
                -i "$FILE_LIST" \
                --no-host-directories \
                -P "$OUTDIR"
        done
    done
done

echo ""
echo "Download complete."
du -sh "$OUTDIR"
