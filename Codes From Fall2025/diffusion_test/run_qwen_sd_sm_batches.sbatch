#!/bin/bash
#SBATCH -J qwen_sr_sample
#SBATCH -A coc
#SBATCH -q coc-gpu          # (keep coc-gpu + QOS=coc-ice if that worked)
#SBATCH --qos=coc-ice
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o qwen_sr_sample.%j.out
#SBATCH -e qwen_sr_sample.%j.err

echo "Starting job on $(hostname) at $(date)"

module load anaconda3

# >>> Proper conda init for non-interactive shell <<<
# Try the conda.sh first; if that path doesn't exist on ICE, use the eval hook.
if [ -f /apps/pace/conda/etc/profile.d/conda.sh ]; then
    source /apps/pace/conda/etc/profile.d/conda.sh
else
    eval "$(/apps/pace/conda/bin/conda shell.bash hook)"
fi

# Activate your prefix env in scratch
conda activate /home/hice1/kpanchal30/scratch/animal_env

# HF cache on scratch
export HF_HOME=/home/hice1/$USER/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/hice1/kpanchal30/code/HAAG_Stone_Mountain_Species_Detection_Project/diffusion_test

python sample_qwen_sd_sm_batches.py

echo "Finished at $(date)"