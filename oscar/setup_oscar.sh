#!/bin/bash
# Run this ONCE after cloning the repo on OSCAR.
# Sets up your Python environment and scratch directories.
#
# Usage:
#   bash oscar/setup_oscar.sh

set -euo pipefail

echo "=== Setting up virtual-tryon on OSCAR ==="

# --- Directories ---
SCRATCH_DIR="${HOME}/scratch/virtual-tryon"
DATA_DIR="${SCRATCH_DIR}/surreal_data"
OUTPUT_DIR="${SCRATCH_DIR}/output"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
echo "Created scratch directories under: $SCRATCH_DIR"


# --- Python environment (manual setup required) ---
# Do NOT run this automatically — module availability varies by OSCAR node.
# Follow these steps manually in the terminal after running this script:
#
#   1. Check available Python modules:
#        module avail python
#
#   2. Load a Python 3.11+ module, e.g.:
#        module load python/3.11.11-5e66
#
#   3. Create the virtual environment in scratch (not home — scratch has more space).
#      Use a separate directory from ~/scratch/virtual-tryon/ (which holds data/outputs):
#        python3 -m venv ~/scratch/venv-tryon
#
#   4. Activate it:
#        source ~/scratch/venv-tryon/bin/activate
#
#   5. Install dependencies:
#        pip install --upgrade pip
#        pip install -r ~/Desktop/virtual-tryon/requirements_oscar.txt
#
#   6. To activate the venv in future sessions, add this to ~/.bashrc:
#        source ~/scratch/venv-tryon/bin/activate

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Get SURREAL credentials by accepting the license at:"
echo "       https://www.di.ens.fr/willow/research/surreal/data/"
echo "  2. When ready to download, export credentials in your session and submit:"
echo "       export SURREAL_USER='yourusername'"
echo "       export SURREAL_PASS='yourpassword'"
echo "       sbatch oscar/download_job.sh"
echo "     sbatch inherits your current environment, so the job will pick up the credentials."
echo "     Do NOT store them in ~/.bashrc or any file tracked by git."
echo ""
echo "  Your data will go to: $DATA_DIR"
echo "  Trained models will go to: $OUTPUT_DIR/checkpoints"
