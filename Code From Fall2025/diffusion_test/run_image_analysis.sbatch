#!/bin/bash
#SBATCH --job-name=img_check
#SBATCH --output=img_check_%j.out
#SBATCH --error=img_check_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=coc-cpu

# No conda, no activation, no deactivation.
# Just call the python interpreter inside your env.

cd "$SLURM_SUBMIT_DIR"

/home/hice1/kpanchal30/scratch/animal_env/bin/python analyze_all_sm_images.py