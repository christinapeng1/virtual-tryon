#!/bin/bash
#SBATCH -J posenet_sub
#SBATCH -o posenet_sub_%j.out
#SBATCH -e posenet_sub_%j.err
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Submit from the repo root (uses 8 CPUs, leaving 4 free for a parallel job):
#   sbatch oscar/train_job_subsample.sh
#
# To resume after a timeout, resubmit with the --resume flag:
#   sbatch oscar/train_job_subsample.sh --resume
#
# Monitor:
#   squeue -u $(whoami)
#   tail -f posenet_sub_<jobid>.out

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load python/3.11.11-5e66
source "${HOME}/scratch/venv-tryon/bin/activate"

# Sanity-check CUDA before launching training
python - <<'EOF'
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
else:
    import sys; sys.exit("CUDA not available — aborting. Check PyTorch CUDA version.")
EOF

PREPROCESSED="${HOME}/scratch/virtual-tryon/surreal_preprocessed"
CHECKPOINTS="${HOME}/scratch/virtual-tryon/posenet_checkpoints_sub"
mkdir -p "$CHECKPOINTS"
echo "Checkpoints: $CHECKPOINTS"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" && -f "${CHECKPOINTS}/last.pt" ]]; then
    RESUME_FLAG="--resume ${CHECKPOINTS}/last.pt"
    echo "Resuming from ${CHECKPOINTS}/last.pt"
fi

python "${SLURM_SUBMIT_DIR}/train/train_subsample.py" \
    --train_csv "${PREPROCESSED}/index_train.csv" \
    --val_csv   "${PREPROCESSED}/index_val.csv"   \
    --out_dir   "${CHECKPOINTS}"                  \
    --subsample 0.20                              \
    --backbone  mobilenet_v3_small                \
    --epochs    30                                \
    --batch     128                               \
    --workers   8                                 \
    $RESUME_FLAG

echo "Job finished at $(date)"
echo "Best checkpoint: ${CHECKPOINTS}/best.pt"
