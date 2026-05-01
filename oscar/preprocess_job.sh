#!/bin/bash
#SBATCH -J surreal_preprocess
#SBATCH -o surreal_preprocess_%j.out
#SBATCH -e surreal_preprocess_%j.err
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p batch

# Submit from the repo root:
#   sbatch oscar/preprocess_job.sh
#
# Monitor:
#   squeue -u $(whoami)
#   tail -f surreal_preprocess_<jobid>.out

set -euo pipefail

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

DATA_ROOT="${HOME}/scratch/virtual-tryon/surreal_data"
OUT_DIR="${HOME}/scratch/virtual-tryon/surreal_preprocessed"

module load python/3.11.11-5e66
source "${HOME}/scratch/venv-tryon/bin/activate"

python "${SLURM_SUBMIT_DIR}/train/preprocess.py" \
    "$DATA_ROOT" \
    "$OUT_DIR"   \
    --splits train val \
    --workers 8

echo "Job finished at $(date)"
du -sh "${OUT_DIR}"
