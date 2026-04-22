# Binding Affinity Prediction with Physics-Guided GNNs

This project develops a graph neural network pipeline for protein-ligand binding affinity prediction using structure-based graph representations and physics-derived auxiliary supervision.

The main input is a protein-ligand complex structure. The primary target is the experimental binding free energy, while MM/GBSA or MM/PBSA-derived quantities are used as training-time auxiliary labels.

A key principle of the project is:

> Physics-derived quantities are used during training, but are not required during inference.

The goal is therefore to build a practical crystal-structure-only predictor at inference time, while still learning from richer physics-based supervision during training.
<img width="1916" height="821" alt="ChatGPT Image 2026年4月22日 14_53_45" src="https://github.com/user-attachments/assets/78f3fc39-3296-4164-a728-2e10cb84934c" />

## Data

For each protein-ligand complex, the project may use:

- crystal complex structure: `complex.pdb`
- experimental binding free energy, `ΔG`
- average MM/GBSA or MM/PBSA results from `mmpbsa.out`
- MD snapshot structures:
  - `frame_200.pdb`
  - `frame_250.pdb`
  - `frame_300.pdb`
  - `frame_350.pdb`
  - `frame_400.pdb`
- frame-level MM/PBSA terms from `snapshot_energy_summary.csv`

The current working set contains five valid complexes:

```text
6QLN
6QLO
6QLP
6QLR
6QLT
```

Each structure is converted into a local ligand-pocket graph containing ligand atoms, nearby protein pocket atoms, and retained pocket-region metals when present.

## Model Variants

### Baseline 1: Crystal Only

Baseline 1 uses only the crystal complex graph to predict experimental binding free energy.

```text
complex.pdb
  -> ligand-pocket graph
  -> GNN encoder
  -> experimental ΔG
```

Inputs:

- `complex.pdb`

Targets:

- experimental `ΔG`

### Baseline 2: Crystal + Average MM/PBSA Auxiliary Supervision

Baseline 2 adds trajectory-averaged MM/PBSA-style terms as auxiliary training labels while keeping crystal-only inference.

```text
complex.pdb
  -> ligand-pocket graph
  -> GNN encoder
  |- experimental ΔG
  -> average MM/PBSA terms
```

Inputs:

- `complex.pdb`

Targets:

- experimental `ΔG`
- average MM/GBSA or MM/PBSA terms from `mmpbsa.out`

### Baseline 3: Crystal Prediction + MD Frame MM/PBSA Auxiliary Supervision

Baseline 3 uses MD frame structures and frame-level MM/PBSA terms during training, but final prediction still depends only on the crystal structure branch.

```text
crystal graph
  -> shared GNN encoder
  -> crystal embedding
  |- experimental ΔG
  -> average PB/MM-PBSA terms

MD frame graphs
  -> same shared GNN encoder
  -> frame embeddings
  -> frame-level PB/MM-PBSA terms
```

The PB/MM-PBSA target vector is:

```text
[vdw, elec, polar_solv, nonpolar_solv, dispersion, total]
```

Training inputs:

- `complex.pdb`
- `frame_200.pdb`
- `frame_250.pdb`
- `frame_300.pdb`
- `frame_350.pdb`
- `frame_400.pdb`

Targets:

- experimental `ΔG`
- average PB/MM-PBSA terms
- frame-level PB/MM-PBSA terms

Inference:

```text
complex.pdb
  -> ligand-pocket graph
  -> shared GNN encoder
  |- experimental ΔG
  -> average PB/MM-PBSA terms
```

No MD frames are required during inference.

### Baseline 4: Baseline 3 + Representation Distillation

Baseline 4 is the next planned variant. It adds representation distillation from MD frame embeddings to the crystal embedding.

MD frame embeddings are mean-pooled into a teacher representation:

```text
h_teacher = mean(h_200, h_250, h_300, h_350, h_400)
```

The crystal embedding is projected and trained to match this teacher representation:

```text
z_crystal = Projector(h_crystal)
L_distill = MSE(z_crystal, stop_gradient(h_teacher))
```

Purpose:

- test whether a crystal-only model can learn additional conformational information from MD frame representations during training

Inference:

- same as Baseline 3: only `complex.pdb` is required

## Baseline Summary

| Model | Crystal Input | Average MM/PBSA | MD Frames | Frame-Level MM/PBSA | Distillation | MD Required at Inference |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 1 | Yes | No | No | No | No | No |
| Baseline 2 | Yes | Yes | No | No | No | No |
| Baseline 3 | Yes | Yes | Training only | Training only | No | No |
| Baseline 4 | Yes | Yes | Training only | Training only | Yes | No |

## Current Repository Status

- `baseline1`
  Conceptually defined, but not exposed as a separate standalone training script in the current repository.
- `baseline2`
  Implemented and runnable. Supports `overfit_one`, `overfit_all`, `leave_one_out`, and validation-based `Scheme A`.
- `baseline3`
  Implemented and tested. The repository includes the rotating 5-run training workflow, Slurm submission scripts, and an evaluation summary script.
- `baseline4`
  Not implemented yet as a separate training pipeline. The shared MD frame export step already exists for future experiments.

## 目录结构 | Layout

- `code/`
  项目代码、训练脚本、预处理脚本，以及更详细的说明文档。
  Source code, training scripts, preprocessing scripts, and the detailed project README.
- `data/`
  原始数据与处理后的图数据。
  Raw dataset inputs and processed graph files.
- `results/`
  训练输出、模型权重、预测结果和评估汇总。
  Training outputs, saved checkpoints, predictions, and evaluation summaries.
- `logs/`
  Slurm 任务日志与预处理日志。
  Slurm stdout/stderr logs and preprocessing job logs.

## 主要入口 | Main Entry Points

- 代码说明 / Code overview:
  `code/README.md`
- 训练脚本 / Training script:
  `code/train_baseline.py`
- 默认 Slurm 脚本 / Main Slurm script:
  `code/run_train_baseline.sbatch`
- 预处理入口 / Preprocessing CLI:
  `code/binding_graph_preprocessing/cli.py`

## 环境 | Environment

当前 conda 环境路径：  
Current conda environment path:

```text
/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1
```

Slurm 脚本当前使用的 Python：  
Python executable currently used by the Slurm scripts:

```text
/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python
```

## 常用流程 | Typical Workflow

进入代码目录：  
Move into the code directory:

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
```

本地做一个快速 sanity check：  
Run a quick local training sanity check:

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_one \
  --sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 50 \
  --print_every 10
```

提交默认训练任务：  
Submit the default Slurm training job:

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
sbatch run_train_baseline.sbatch
```

查看日志：  
Inspect generated logs:

```bash
ls /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/logs
```

查看训练结果：  
Inspect saved results:

```bash
ls /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/results/training_runs
```

## 说明 | Notes

- 现在默认推荐从 `code/` 目录运行脚本。
  Scripts are now organized around being run from `code/`.
- 处理后的图数据默认位于 `data/MMPBSA/processed/graphs`。
  Processed graphs are expected under `data/MMPBSA/processed/graphs`.
- 默认训练输出位于 `results/training_runs/`。
  Default training outputs are written under `results/training_runs/`.
- Slurm 日志统一写到 `logs/`。
  Slurm logs are written into `logs/`.
