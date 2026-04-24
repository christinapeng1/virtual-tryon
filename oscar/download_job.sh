#!/bin/bash
#SBATCH --job-name=surreal_download
#SBATCH --output=%x_%j.out       # output file: surreal_download_<jobid>.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00           # 4 hours should be more than enough for ~7GB
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch         # use 'batch' (no GPU needed for download)

# Submit with:
#   sbatch download_job.sh
#
# Monitor with:
#   squeue -u $(whoami)
#   tail -f surreal_download_<jobid>.out

set -euo pipefail

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/download_surreal.sh"

echo "Job finished at $(date)"
