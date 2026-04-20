#!/bin/bash
#SBATCH -A <your_project>
#SBATCH -J pl_gnn_prep
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/prep_%j.out
#SBATCH -e /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs/prep_%j.err

module load Anaconda3/2024.06-1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1

cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code

python -m binding_graph_preprocessing.cli \
  --root-dir ../data/MMPBSA \
  --output-dir ../data/MMPBSA/processed \
  --ligand-resname J5W \
  --ligand-resid 139
