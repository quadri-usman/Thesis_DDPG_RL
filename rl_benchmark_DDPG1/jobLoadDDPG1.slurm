#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=A4500
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=48:00:00               #limit your job by time
#SBATCH --nodes=1                  #push to specific node
#SBATCH --mem=32G
#SBATCH --output=gpu_output_%j.txt
#SBATCH --error=error_%j.log

python evaluate_ddpg_final.py \
    --checkpoint ddpg_test_25217/best.pt \
    --mode all \
    --split test \
    --seeds 2001 2002 2003 2004 2005 \
    --matlab_frequency_csv Matlab_frequency_scenario.csv \
    --output_dir "stage1_final_evaluation_${SLURM_JOB_ID}" \