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

# --- Python environment ---
# OSCAR uses modules; load a recent Python
module load python/3.11.0 2>/dev/null || module load python 2>/dev/null || true

VENV_PATH="${HOME}/scratch/venv-tryon"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating Python virtual environment at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi

source "${VENV_PATH}/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$(dirname "$0")/../requirements_oscar.txt" --quiet
echo "Python environment ready: $VENV_PATH"

# --- .bashrc additions ---
BASHRC="${HOME}/.bashrc"
if ! grep -q "SURREAL_USER" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << 'EOF'

# --- virtual-tryon OSCAR config ---
# Fill in your SURREAL credentials after accepting the license:
# https://www.di.ens.fr/willow/research/surreal/data/
export SURREAL_USER=""
export SURREAL_PASS=""

# Activate pose model venv by default (comment out if unwanted)
# source ~/scratch/venv-tryon/bin/activate
EOF
    echo "Added credential placeholders to ~/.bashrc"
    echo ">>> Edit ~/.bashrc and fill in SURREAL_USER and SURREAL_PASS <<<"
else
    echo "~/.bashrc already has SURREAL_USER set"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit ~/.bashrc and fill in SURREAL_USER / SURREAL_PASS"
echo "  2. Run:  source ~/.bashrc"
echo "  3. To download SURREAL data, submit the SLURM job:"
echo "       sbatch oscar/download_job.sh"
echo "  4. Or run interactively:"
echo "       srun --mem=4G --time=04:00:00 --pty bash"
echo "       bash oscar/download_surreal.sh"
echo ""
echo "  Your data will go to: $DATA_DIR"
echo "  Trained models will go to: $OUTPUT_DIR/checkpoints"
