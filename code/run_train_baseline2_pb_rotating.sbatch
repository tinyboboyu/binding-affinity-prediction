#!/bin/bash
# LUNARC GPU training script for one formal Baseline 2-PB rotating split.
#
# Defaults follow the Baseline 2-PB crystal + average PB auxiliary setup:
# - rotating_train_val_test
# - 500 epochs
# - train-set-only experimental Delta G and average PB normalization enabled
# - results written into the repository results directory
#
# Example:
# sbatch --export=ALL,SPLIT_ROUND=1 run_train_baseline2_pb_rotating.sbatch

#SBATCH -A lu2026-7-28
#SBATCH -J b2pb_rot
#SBATCH -p gpua40i
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11800
#SBATCH -t 03:00:00
#SBATCH -o /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_b2pb_rotate_%A_%a.out
#SBATCH -e /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_b2pb_rotate_%A_%a.err

set -euo pipefail

module load Anaconda3/2024.06-1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

ENV_PYTHON=${ENV_PYTHON:-/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python}
export LD_LIBRARY_PATH=/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/lib:${LD_LIBRARY_PATH:-}

SPLIT_ROUND=${SPLIT_ROUND:-1}
EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-1e-3}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NUM_LAYERS=${NUM_LAYERS:-2}
DROPOUT=${DROPOUT:-0.0}
LAMBDA_AVG=${LAMBDA_AVG:-0.1}
SEED=${SEED:-42}
PRINT_EVERY=${PRINT_EVERY:-10}
NORMALIZE_LABELS=${NORMALIZE_LABELS:-true}
GRAPH_DIR=${GRAPH_DIR:-../data/MMPBSA/processed/graphs}
MMPBSA_ROOT=${MMPBSA_ROOT:-../data/MMPBSA}

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
echo "LAMBDA_AVG: ${LAMBDA_AVG}"
echo "NORMALIZE_LABELS: ${NORMALIZE_LABELS}"
echo "GRAPH_DIR: ${GRAPH_DIR}"
echo "MMPBSA_ROOT: ${MMPBSA_ROOT}"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -V
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -c "import torch; import torch_geometric; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" train_baseline2_pb.py \
  --graph_dir "${GRAPH_DIR}" \
  --mmpbsa_root "${MMPBSA_ROOT}" \
  --split_mode rotating_train_val_test \
  --split_round "${SPLIT_ROUND}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lambda_avg "${LAMBDA_AVG}" \
  --normalize_labels "${NORMALIZE_LABELS}" \
  --seed "${SEED}" \
  --print_every "${PRINT_EVERY}"
