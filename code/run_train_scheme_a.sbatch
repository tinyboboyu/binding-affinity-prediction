#!/bin/bash
# LUNARC GPU training script for Scheme A:
# use one validation sample from the non-test complexes to select the best epoch,
# then evaluate the held-out test sample only once with the best checkpoint.
#
# Sample selection can be overridden at submission time, for example:
# sbatch --export=ALL,TEST_SAMPLE_ID=6QLO,VAL_SAMPLE_ID=6QLP run_train_scheme_a.sbatch

#SBATCH -A lu2026-7-28
#SBATCH -J pl_gnn_a
#SBATCH -p gpua40i
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11800
#SBATCH -t 02:00:00
#SBATCH -o /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_scheme_a_%j.out
#SBATCH -e /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_scheme_a_%j.err

set -euo pipefail

module load Anaconda3/2024.06-1

ENV_PYTHON=/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python
export LD_LIBRARY_PATH=/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/lib:$LD_LIBRARY_PATH

TEST_SAMPLE_ID=${TEST_SAMPLE_ID:-6QLN}
VAL_SAMPLE_ID=${VAL_SAMPLE_ID:-6QLO}

cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "ENV_PYTHON: ${ENV_PYTHON}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "TEST_SAMPLE_ID: ${TEST_SAMPLE_ID}"
echo "VAL_SAMPLE_ID: ${VAL_SAMPLE_ID}"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -V
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -c "import torch; import torch_geometric; from rdkit import Chem; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

# Scheme A for one leave-one-out run.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode leave_one_out \
  --selection_mode val \
  --test_sample_id "${TEST_SAMPLE_ID}" \
  --val_sample_id "${VAL_SAMPLE_ID}" \
  --target_mode multi_gb \
  --epochs 500 \
  --batch_size 1 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --num_layers 2 \
  --lambda_aux 1.0 \
  --seed 42 \
  --print_every 20
