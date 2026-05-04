#!/bin/bash
#SBATCH --job-name=gesture
#SBATCH --output=gesture.out
#SBATCH --error=gesture.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load python/3.10
module load cuda/11.8

source gesture-env/bin/activate

cd /gesture-swipe

python train.py