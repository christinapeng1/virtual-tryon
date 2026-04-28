#!/bin/bash
#SBATCH --job-name=surreal_download
#SBATCH --output=%x_%j.out       # output file: surreal_download_<jobid>.out
#SBATCH --error=%x_%j.err
#SBATCH --time=8:00:00           # 8 hours to safely download ~7GB across 110k files
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

# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was run.
# Use it instead of BASH_SOURCE, which resolves to the spool directory on compute nodes.
bash "${SLURM_SUBMIT_DIR}/oscar/download_surreal.sh"

echo "Job finished at $(date)"
