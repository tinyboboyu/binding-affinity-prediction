# Protein-Ligand Graph Preprocessing

This package preprocesses the MMPBSA dataset under `../data/MMPBSA/` into one merged PyTorch Geometric graph per complex, with experimental `ΔG` as the main label and GB energy components as auxiliary labels.

## Minimal dependencies

This v1 pipeline follows the minimal-dependency route and only needs:

```text
python
torch
torch-geometric
rdkit
```

Recommended installation order:

1. Install `torch`
2. Install `torch-geometric`
3. Install `rdkit`

`rdkit` is strongly recommended because it improves ligand bond typing, aromaticity, hybridization, donor/acceptor flags, and hydrogen counts. If `rdkit` is unavailable, the pipeline still runs and falls back in this fixed order:

```text
RDKit -> CONECT -> geometry
```

Without `rdkit`, ligand chemistry features become more heuristic, but the dataset preprocessing workflow still works.

## Dataset output

```text
../data/MMPBSA/processed/
  graphs/
    6QLN.pt
    6QLO.pt
    6QLP.pt
    6QLR.pt
    6QLT.pt
  metadata.csv
  failures.json
```

## Example usage

```bash
python -m binding_graph_preprocessing.cli
```

```bash
python -m binding_graph_preprocessing.cli \
  --root-dir /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/data/MMPBSA \
  --output-dir /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/data/MMPBSA/processed \
  --ligand-resname J5W \
  --ligand-resid 139
```

## Notes

- The default sample set is `6QLN`, `6QLO`, `6QLP`, `6QLR`, `6QLT`.
- `6QLQ` and `6QLU` are skipped by design.
- Waters are removed, hydrogens are retained, and active-site metals are kept only when they fall inside the pocket range.
- Protein internal edges are spatial only.
- Ligand internal edges are covalent only.
- Ligand-protein edges are spatial only.
- Experimental values in `bd` are converted from kJ/mol to kcal/mol.
- GB terms from `mmpbsa.out` are read directly in kcal/mol.

## Training baseline

The first stable training baseline is built around the already processed graph files in:

```text
../data/MMPBSA/processed/graphs
```

It uses the existing saved PyTorch Geometric `Data` objects directly and does not redesign the graph format.

### Training files

- `dataset.py`
  Loads the saved `.pt` graph files in a deterministic order.
- `model.py`
  Defines `MultiTaskComplexGNN`, a simple shared GIN encoder with two regression heads.
- `train_baseline.py`
  Main training script for overfit and leave-one-out experiments.
- `inspect_all_graphs.py`
  Utility script that summarizes all saved graphs and writes `graph_summary.csv`.
- `run_train_baseline.sbatch`
  Example Slurm submission script for LUNARC GPU training.

### Model behavior

This baseline is a structure-only multi-task regression model.

- Inputs used by the model:
  - `x`
  - `edge_index`
  - `batch`
- Inputs intentionally ignored in v1:
  - `pos`
  - `edge_attr`
- Targets:
  - main target: `y_exp`
  - auxiliary target: `y_aux = [y_vdw, y_elec, y_polar, y_nonpolar]`

So MM/GBSA terms are supervision targets, not model inputs.

### Valid sample IDs

Only these five samples are valid and used anywhere in training:

```text
6QLN
6QLO
6QLP
6QLR
6QLT
```

`6QLQ` and `6QLU` are excluded and must not be used.

## Local training

When using your conda environment on LUNARC, it is recommended to run with:

```bash
PYTHONNOUSERSITE=1 python ...
```

This avoids user-site package pollution from `~/.local`.

### Quick sanity check: overfit one sample

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_one \
  --sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 50 \
  --print_every 10
```

This is the best first test to confirm:

- graphs load correctly
- the model runs forward and backward
- losses decrease
- outputs are saved correctly

### Overfit all samples

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode multi_gb \
  --epochs 500 \
  --print_every 20
```

### Leave-one-out

Example with `6QLN` as the held-out test sample:

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode leave_one_out \
  --test_sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 500 \
  --print_every 20
```

This means:

- train: `6QLO`, `6QLP`, `6QLR`, `6QLT`
- test: `6QLN`

### Target modes

`train_baseline.py` supports three target modes:

- `exp`
  Predict experimental `ΔG` only.
- `aux_gb`
  Predict the 4D GB decomposition target only:
  `[y_vdw, y_elec, y_polar, y_nonpolar]`
- `multi_gb`
  Predict both `y_exp` and `y_aux`

Examples:

Experimental target only:

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode exp
```

Auxiliary GB targets only:

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode aux_gb
```

Recommended multi-task mode:

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode multi_gb
```

### Important CLI arguments

Main training arguments:

- `--graph_dir`
- `--save_dir`
- `--split_mode`
- `--sample_id`
- `--test_sample_id`
- `--target_mode`
- `--epochs`
- `--batch_size`
- `--lr`
- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--lambda_aux`
- `--seed`
- `--print_every`
- `--device`

Default values:

```text
split_mode = overfit_one
sample_id = 6QLN
target_mode = multi_gb
epochs = 500
batch_size = 1
lr = 1e-3
hidden_dim = 64
num_layers = 2
lambda_aux = 1.0
seed = 42
print_every = 20
```

## LUNARC Slurm workflow

An example GPU submission script is provided in:

```text
run_train_baseline.sbatch
```

Before submitting, update:

```bash
#SBATCH -A <YOUR_PROJECT>
```

to your real LUNARC project account.

Then submit:

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
sbatch run_train_baseline.sbatch
```

Check job queue:

```bash
squeue -u $USER
```

Inspect logs:

```bash
cat ../logs/slurm_train_<jobid>.out
cat ../logs/slurm_train_<jobid>.err
```

The provided Slurm script currently runs a leave-one-out experiment by default. You can edit the final Python command inside the script to switch to:

- `overfit_one`
- `overfit_all`
- `leave_one_out`

## Training outputs

Each run writes results into a run-specific directory under:

```text
../results/training_runs/
```

For example:

```text
../results/training_runs/overfit_one_multi_gb_6QLN/
../results/training_runs/overfit_all_multi_gb_all/
../results/training_runs/leave_one_out_multi_gb_loo_6QLN/
```

Each run saves:

- `best_model.pt`
  Best checkpoint selected by the lowest evaluation `loss_total`.
- `train_log.csv`
  Per-epoch training and evaluation losses.
- `best_predictions.csv`
  Predictions from the best checkpoint on the evaluation set.

### What is inside `best_model.pt`

The file stores a dictionary, not just a raw state dict. It contains:

- `model_state_dict`
- `model_config`
- `split_info`
- `target_mode`
- `best_epoch`
- `best_eval_loss`

### What is inside `train_log.csv`

Columns:

- `epoch`
- `train_loss_total`
- `train_loss_exp`
- `train_loss_aux`
- `eval_loss_total`
- `eval_loss_exp`
- `eval_loss_aux`

### What is inside `best_predictions.csv`

Always includes:

- `sample_id`
- `pred_exp`
- `true_exp`

For `aux_gb` and `multi_gb`, it also includes:

- `pred_vdw`
- `true_vdw`
- `pred_elec`
- `true_elec`
- `pred_polar`
- `true_polar`
- `pred_nonpolar`
- `true_nonpolar`

## Inspecting saved graphs

To summarize all processed graphs:

```bash
PYTHONNOUSERSITE=1 python inspect_all_graphs.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --save_csv graph_summary.csv
```

This creates a CSV with:

- graph size
- node counts by type
- edge counts by type
- labels
- NaN checks for `x`, `pos`, and `edge_attr`

## Common troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

You are likely not in the correct conda environment.

Check:

```bash
which python
python -V
```

Expected Python should point to your conda environment, for example:

```text
/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python
```

### `ModuleNotFoundError: No module named 'torch_geometric'`

Install PyG inside the active conda environment:

```bash
PYTHONNOUSERSITE=1 python -m pip install torch-geometric
```

### `torch.load` fails with `weights_only` errors

PyTorch 2.6+ changed the default `torch.load` behavior.
All project code uses:

```python
torch.load(path, weights_only=False)
```

If you inspect graph files manually, use the same pattern.

### `pip` imports the wrong packages from `~/.local`

Use:

```bash
PYTHONNOUSERSITE=1 python -m pip ...
PYTHONNOUSERSITE=1 python ...
```

instead of plain `pip`.

### Slurm job fails immediately because of project/account errors

Check the `#SBATCH -A` line in `run_train_baseline.sbatch`.
LUNARC requires the actual Slurm account string, not a human-readable project title.

### Training seems to run but loss does not change

Start with:

- `split_mode=overfit_one`
- `sample_id=6QLN`
- `target_mode=multi_gb`

If even the single-sample overfit run does not improve, inspect:

- graph loading
- label values
- optimizer settings
- whether the correct environment is active

## 中文训练说明

下面是上面训练部分的中文说明，便于直接查阅和在 LUNARC 上使用。

### 训练相关文件

- `dataset.py`
  负责按固定顺序加载已经生成好的 `.pt` 图文件。
- `model.py`
  定义 `MultiTaskComplexGNN`，是一个共享 GIN 编码器加两个回归头的基线模型。
- `train_baseline.py`
  主训练脚本，负责数据划分、训练、评估、保存模型和日志。
- `inspect_all_graphs.py`
  用来批量检查所有 `.pt` 图文件，并导出 `graph_summary.csv`。
- `run_train_baseline.sbatch`
  用于在 LUNARC 上提交 GPU 训练任务的 Slurm 示例脚本。

### 训练输入和目标

这个第一版模型是一个“只看结构”的多任务回归模型。

- 模型真正使用的输入：
  - `x`
  - `edge_index`
  - `batch`
- 当前版本故意不使用：
  - `pos`
  - `edge_attr`
- 训练目标：
  - 主目标：`y_exp`
  - 辅助目标：`y_aux = [y_vdw, y_elec, y_polar, y_nonpolar]`

也就是说，MM/GBSA 分解项是监督信号，不是模型输入。

### 有效样本

训练中只允许使用以下 5 个样本：

```text
6QLN
6QLO
6QLP
6QLR
6QLT
```

`6QLQ` 和 `6QLU` 不参与训练，也不参与任何评估。

## 本地训练命令

在 LUNARC 的 conda 环境里，推荐统一使用：

```bash
PYTHONNOUSERSITE=1 python ...
```

这样可以避免 `~/.local` 里的用户级 Python 包污染当前 conda 环境。

### 1. 单样本过拟合：最小 sanity check

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_one \
  --sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 50 \
  --print_every 10
```

这一步最适合先检查：

- 图文件能不能正常加载
- 模型能不能正常 forward / backward
- loss 会不会下降
- 输出文件能不能正常保存

### 2. 全部样本过拟合

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode multi_gb \
  --epochs 500 \
  --print_every 20
```

### 3. 留一验证 leave-one-out

例如把 `6QLN` 作为测试集：

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode leave_one_out \
  --test_sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 500 \
  --print_every 20
```

这时数据划分是：

- train: `6QLO`, `6QLP`, `6QLR`, `6QLT`
- test: `6QLN`

## 三种 split_mode 的含义

### `overfit_one`

- 训练集只包含一个样本
- 默认样本是 `6QLN`
- 主要用途是检查代码和模型是否正常工作

### `overfit_all`

- 训练集包含全部 5 个样本
- 评估也在这 5 个样本上完成
- 主要用途是检查模型是否有能力拟合整个小数据集

### `leave_one_out`

- 留出 1 个样本作为测试集
- 其余 4 个样本作为训练集
- 这是当前最接近“泛化测试”的 baseline 模式

## target_mode 的含义

### `exp`

只预测实验结合自由能 `y_exp`

示例：

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode exp
```

### `aux_gb`

只预测辅助 GB 分解目标：

```text
[y_vdw, y_elec, y_polar, y_nonpolar]
```

示例：

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode aux_gb
```

### `multi_gb`

同时预测：

- `y_exp`
- `y_aux`

这是当前推荐模式。

示例：

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_all \
  --target_mode multi_gb
```

## 主要训练参数

常用参数包括：

- `--graph_dir`
- `--save_dir`
- `--split_mode`
- `--sample_id`
- `--test_sample_id`
- `--target_mode`
- `--epochs`
- `--batch_size`
- `--lr`
- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--lambda_aux`
- `--seed`
- `--print_every`
- `--device`

默认值：

```text
split_mode = overfit_one
sample_id = 6QLN
target_mode = multi_gb
epochs = 500
batch_size = 1
lr = 1e-3
hidden_dim = 64
num_layers = 2
lambda_aux = 1.0
seed = 42
print_every = 20
```

## LUNARC 上的 Slurm 提交流程

项目中已经提供了一个示例脚本：

```text
run_train_baseline.sbatch
```

提交之前，先把脚本里的这一行：

```bash
#SBATCH -A <YOUR_PROJECT>
```

改成你自己在 LUNARC 上可用的实际项目账号。

然后提交：

```bash
cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code
sbatch run_train_baseline.sbatch
```

查看任务队列：

```bash
squeue -u $USER
```

查看日志：

```bash
cat ../logs/slurm_train_<jobid>.out
cat ../logs/slurm_train_<jobid>.err
```

当前 `run_train_baseline.sbatch` 默认跑的是一个 leave-one-out 任务。  
如果你想切换成 `overfit_one` 或 `overfit_all`，只需要修改脚本最后一条 `python train_baseline.py ...` 命令。

## 训练输出文件解释

每次训练都会把结果写入：

```text
../results/training_runs/
```

下面的一个独立运行目录中，例如：

```text
../results/training_runs/overfit_one_multi_gb_6QLN/
../results/training_runs/overfit_all_multi_gb_all/
../results/training_runs/leave_one_out_multi_gb_loo_6QLN/
```

每个目录里至少包含：

- `best_model.pt`
- `train_log.csv`
- `best_predictions.csv`

### `best_model.pt`

它不是单纯的权重文件，而是一个字典，里面包括：

- `model_state_dict`
- `model_config`
- `split_info`
- `target_mode`
- `best_epoch`
- `best_eval_loss`

### `train_log.csv`

每个 epoch 一行，主要字段包括：

- `epoch`
- `train_loss_total`
- `train_loss_exp`
- `train_loss_aux`
- `eval_loss_total`
- `eval_loss_exp`
- `eval_loss_aux`

### `best_predictions.csv`

始终包含：

- `sample_id`
- `pred_exp`
- `true_exp`

在 `aux_gb` 或 `multi_gb` 模式下，还会额外包含：

- `pred_vdw`
- `true_vdw`
- `pred_elec`
- `true_elec`
- `pred_polar`
- `true_polar`
- `pred_nonpolar`
- `true_nonpolar`

## 图文件汇总检查

如果你想快速检查所有图文件，可以运行：

```bash
PYTHONNOUSERSITE=1 python inspect_all_graphs.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --save_csv graph_summary.csv
```

这个脚本会输出每个图的：

- 节点数
- 边数
- ligand / protein / metal 节点数
- 各种标签值
- 是否存在 `NaN`

## 常见报错排查

### 1. `ModuleNotFoundError: No module named 'torch'`

通常说明你没有进入正确的 conda 环境。

检查：

```bash
which python
python -V
```

你应该看到类似：

```text
/lunarc/nobackup/projects/teobio/Xiaofan/pl_gnn_v1/bin/python
```

### 2. `ModuleNotFoundError: No module named 'torch_geometric'`

说明当前环境里还没有正确安装 PyG。  
在激活环境后安装：

```bash
PYTHONNOUSERSITE=1 python -m pip install torch-geometric
```

### 3. `torch.load` 提示 `weights_only` 错误

这是 PyTorch 2.6 之后的默认行为变化。  
项目代码里已经统一使用：

```python
torch.load(path, weights_only=False)
```

如果你手动加载 `.pt` 文件，也要这样写。

### 4. `pip` 导入了 `~/.local` 里的错误包

请尽量使用：

```bash
PYTHONNOUSERSITE=1 python -m pip ...
PYTHONNOUSERSITE=1 python ...
```

不要直接使用裸 `pip`。

### 5. Slurm 提交后立刻失败，提示 project/account 错误

请检查 `run_train_baseline.sbatch` 里的：

```bash
#SBATCH -A ...
```

这里必须填写 LUNARC 真正认可的 Slurm account 名称，而不是人类可读的项目标题。

### 6. 训练能运行，但 loss 几乎不变

建议先从下面这个最小 sanity check 开始：

```bash
PYTHONNOUSERSITE=1 python train_baseline.py \
  --graph_dir ../data/MMPBSA/processed/graphs \
  --split_mode overfit_one \
  --sample_id 6QLN \
  --target_mode multi_gb \
  --epochs 50 \
  --print_every 10
```

如果单样本都无法明显拟合，再去检查：

- 图文件是否正确
- 标签值是否合理
- 学习率是否合适
- conda 环境是否真的激活
