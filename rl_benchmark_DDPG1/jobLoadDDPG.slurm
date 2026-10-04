#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=A4500
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=48:00:00               #limit your job by time
#SBATCH --nodes=1                  #push to specific node
#SBATCH --mem=32G
#SBATCH --output=gpu_output_%j.txt
#SBATCH --error=error_%j.log

python evaluate_ddpg.py \
    --checkpoint ddpg_test_25217/best.pt \
    --split validation \
    --seeds 1001 1002 1003 \
    --output_dir "ddpg_validation_${SLURM_JOB_ID}" \
    