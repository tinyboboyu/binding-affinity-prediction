# Binding Affinity Prediction

这个目录是整理后的 protein-ligand binding affinity prediction 项目根目录。  
This directory is the reorganized project root for protein-ligand binding affinity prediction.

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
