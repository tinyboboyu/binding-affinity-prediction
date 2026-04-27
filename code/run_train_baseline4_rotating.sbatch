#!/bin/bash
# LUNARC GPU training script for one formal Baseline 4 rotating split.
#
# Defaults follow the current Baseline 4 experimental choices:
# - rotating_train_val_test
# - 500 epochs with 100 warm-up epochs
# - label normalization enabled
# - results written into the repository results directory
#
# Example:
# sbatch --export=ALL,SPLIT_ROUND=1 run_train_baseline4_rotating.sbatch

#SBATCH -A lu2026-7-28
#SBATCH -J b4_rotate
#SBATCH -p gpua40i
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11800
#SBATCH -t 06:00:00
#SBATCH -o /lunarc/nobackup/projects/snic2019-35-66/Xiaofan/binding_affinity_prediction/logs/slurm_b4_rotate_%A_%a.out
#SBATCH -e /lunarc/nobackup/projects/snic2019-35-66/Xiaofan/binding_affinity_prediction/logs/slurm_b4_rotate_%A_%a.err

set -euo pipefail

module load Anaconda3/2024.06-1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

ENV_PYTHON=${ENV_PYTHON:-/lunarc/nobackup/projects/snic2019-35-66/Xiaofan/pl_gnn_v1/bin/python}
export LD_LIBRARY_PATH=/lunarc/nobackup/projects/snic2019-35-66/Xiaofan/pl_gnn_v1/lib:${LD_LIBRARY_PATH:-}

SPLIT_ROUND=${SPLIT_ROUND:-1}
EPOCHS=${EPOCHS:-500}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-1e-3}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NUM_LAYERS=${NUM_LAYERS:-2}
DROPOUT=${DROPOUT:-0.0}
LAMBDA_AVG=${LAMBDA_AVG:-0.02}
LAMBDA_FRAME=${LAMBDA_FRAME:-0.01}
LAMBDA_DISTILL=${LAMBDA_DISTILL:-0.01}
SEED=${SEED:-42}
PRINT_EVERY=${PRINT_EVERY:-10}
NORMALIZE_LABELS=${NORMALIZE_LABELS:-true}
DEBUG=${DEBUG:-false}

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/training_runs"
cd "${SCRIPT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "ENV_PYTHON: ${ENV_PYTHON}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SPLIT_ROUND: ${SPLIT_ROUND}"
echo "EPOCHS: ${EPOCHS}"
echo "WARMUP_EPOCHS: ${WARMUP_EPOCHS}"
echo "LAMBDA_AVG: ${LAMBDA_AVG}"
echo "LAMBDA_FRAME: ${LAMBDA_FRAME}"
echo "LAMBDA_DISTILL: ${LAMBDA_DISTILL}"
echo "NORMALIZE_LABELS: ${NORMALIZE_LABELS}"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -V
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -c "import torch; import torch_geometric; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" train_baseline4.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --raw_root_dir ../data/MMPBSA \
  --frame_root_dir ../data/md_frame_exports \
  --split_mode rotating_train_val_test \
  --split_round "${SPLIT_ROUND}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lambda_avg "${LAMBDA_AVG}" \
  --lambda_frame "${LAMBDA_FRAME}" \
  --lambda_distill "${LAMBDA_DISTILL}" \
  --normalize_labels "${NORMALIZE_LABELS}" \
  --seed "${SEED}" \
  --print_every "${PRINT_EVERY}" \
  --debug "${DEBUG}"
