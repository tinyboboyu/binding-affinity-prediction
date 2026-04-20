#!/bin/bash
# LUNARC GPU training script for the first stable graph baseline.
#
# Official LUNARC references used when choosing resources:
# - GPU access and partitions:
#   https://lunarc-documentation.readthedocs.io/en/latest/manual/submitting_jobs/manual_specifying_requirements/
# - Basic SLURM job scripts:
#   https://lunarc-documentation.readthedocs.io/en/latest/manual/submitting_jobs/manual_basic_job/
# - Job environment notes:
#   https://lunarc-documentation.readthedocs.io/en/latest/manual/submitting_jobs/manual_env/
#
# Recommended default for this lightweight baseline:
# - gpua40i
# - 1 node
# - 16 CPU tasks
# - 1 GPU

#SBATCH -A lu2026-7-28
#SBATCH -J pl_gnn_train
#SBATCH -p gpua40i
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11800
#SBATCH -t 02:00:00
#SBATCH -o /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_train_%j.out
#SBATCH -e /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/slurm_train_%j.err

set -euo pipefail

module load Anaconda3/2024.06-1

ENV_PYTHON=/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python
export LD_LIBRARY_PATH=/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/lib:$LD_LIBRARY_PATH

cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "ENV_PYTHON: ${ENV_PYTHON}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -V
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -c "import torch; import torch_geometric; from rdkit import Chem; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

# Default training command:
# - structure-only graph regression
# - leave_one_out is the most useful evaluation mode for 5 samples
# - override any of these by editing the command below
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode leave_one_out \
  --test_sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 500 \
  --batch_size 1 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --num_layers 2 \
  --lambda_aux 1.0 \
  --seed 42 \
  --print_every 20
