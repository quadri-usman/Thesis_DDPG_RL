#!/bin/bash
#SBATCH --job-name=ddpg_sens
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=0-26
#SBATCH --output=ddpg_sens_%A_%a.out
#SBATCH --error=ddpg_sens_%A_%a.err

set -euo pipefail

# ============================================================
# DDPG reward sensitivity study
# 9 reward cases x 3 independent training seeds = 27 jobs
#
# Array mapping:
#   0-2   -> R0
#   3-5   -> R1
#   ...
#   24-26 -> R8
# ============================================================

REWARD_CASES=(R0 R1 R2 R3 R4 R5 R6 R7 R8)
TRAIN_SEEDS=(0 100 200)

NSEEDS=${#TRAIN_SEEDS[@]}

CASE_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NSEEDS))

REWARD_CASE=${REWARD_CASES[$CASE_INDEX]}
TRAIN_SEED=${TRAIN_SEEDS[$SEED_INDEX]}

OUTPUT_DIR="ddpg_reward_sensitivity/${REWARD_CASE}/seed_${TRAIN_SEED}"

echo "============================================================"
echo "DDPG REWARD SENSITIVITY ARRAY JOB"
echo "============================================================"
echo "SLURM job ID       : ${SLURM_JOB_ID}"
echo "SLURM array job ID : ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Reward case        : ${REWARD_CASE}"
echo "Training seed      : ${TRAIN_SEED}"
echo "Output directory   : ${OUTPUT_DIR}"
echo "Host               : $(hostname)"
echo "Start time         : $(date)"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

python train_ddpg_reward_sensitivity.py \
    --reward_case "${REWARD_CASE}" \
    --train_seed "${TRAIN_SEED}" \
    --episodes 100 \
    --eval_every 10 \
    --checkpoint_every 10 \
    --eval_seeds 1001 1002 1003 \
    --output_dir "${OUTPUT_DIR}"

echo "============================================================"
echo "COMPLETED"
echo "Reward case   : ${REWARD_CASE}"
echo "Training seed : ${TRAIN_SEED}"
echo "End time      : $(date)"
echo "Results       : ${OUTPUT_DIR}"
echo "============================================================"