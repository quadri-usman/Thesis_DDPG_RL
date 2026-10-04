#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=48:00:00               #limit your job by time
#SBATCH --nodes=1                  #push to specific node
#SBATCH --mem=32G
#SBATCH --output=gpu_output_%j.txt
#SBATCH --error=error_%j.log

python train_ddpg_split.py \
    --episodes 100 \
    --eval_every 10 \
    --checkpoint_every 10 \
    --output_dir "ddpg_test_${SLURM_JOB_ID}" \