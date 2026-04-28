# Physics-Guided Graph Neural Networks for Protein-Ligand Binding Affinity Prediction with Crystal-Only Inference

## 1. Title

**Physics-Guided Graph Neural Networks for Protein-Ligand Binding Affinity Prediction with Crystal-Only Inference**

## 2. Abstract

Protein-ligand binding affinity prediction is a central task in structure-based drug discovery, but accurate prediction remains challenging when only small ligand-complex datasets are available. This project implements a hierarchy of graph neural network (GNN) baselines for binding affinity prediction using protein-ligand complex structures. The central design principle is that physics-derived quantities are used during training but are not required during inference; the final inference input remains the crystal protein-ligand complex graph.

The repository implements four current baselines. Baseline 1 predicts experimental binding free energy from the crystal graph only. Baseline 2-PB augments the crystal-only prediction task with average PB/MM-PBSA auxiliary labels parsed from the POISSON BOLTZMANN Differences block of `mmpbsa.out`. Baseline 3 adds MD frame graphs and frame-level PB/MM-PBSA supervision during training while retaining crystal-only inference. Baseline 4 extends Baseline 3 with an explicit representation distillation objective that aligns the crystal embedding with the mean embedding of MD frame graphs. The implemented models use a shared GIN encoder and graph-level mean pooling.

The current saved formal results cover five rotating train/validation/test rounds for each of Baseline 1, Baseline 2-PB, Baseline 3, and Baseline 4. These results are stored under `results/training_runs/`. Based on the saved experimental Delta G summary files and merged prediction tables, Baseline 2-PB has the strongest current experimental Delta G performance among the four baselines, with test MAE 0.269 kcal/mol, RMSE 0.430 kcal/mol, and Pearson correlation 0.753 across five held-out test predictions. The dataset contains only five complexes, so these findings should be interpreted cautiously.

## 3. Introduction

Protein-ligand binding affinity prediction is important for ranking candidate compounds, prioritizing experimental assays, and understanding molecular recognition. Structure-based machine learning methods are attractive because they can represent the three-dimensional molecular environment of a ligand bound to a protein target. Graph neural networks provide a natural modeling framework: atoms can be represented as nodes, molecular or spatial relationships as edges, and learned graph-level embeddings can be used for binding affinity prediction.

Purely data-driven graph models can be limited when the available dataset is small. In this repository, the current main dataset contains five valid complexes. Under such conditions, auxiliary supervision may help constrain representation learning. MM/PBSA and related endpoint free-energy calculations provide approximate physics-derived decomposition terms, including van der Waals, electrostatic, polar solvation, nonpolar solvation, dispersion, and total binding-energy components. These quantities are not treated as perfect ground truth, but they provide physically structured training signals.

MD snapshots may provide additional conformational information beyond a single crystal structure. However, requiring MD simulation or MM/PBSA calculation at inference time would be computationally expensive and would reduce practical usability. Therefore, this project asks the following research question:

**Can MM/PBSA-derived auxiliary supervision and MD-derived conformational information improve graph-based binding affinity prediction while keeping inference crystal-structure-only?**

The implemented baseline hierarchy addresses this question by progressively adding physics-derived and MD-derived training signals while preserving crystal-only inference.

## 4. Dataset

The dataset and preprocessing paths were identified from the repository files and code. The main crystal-complex dataset is stored under `data/MMPBSA/`. The implemented default valid sample IDs are defined in `code/binding_graph_preprocessing/constants.py` as:

```text
6QLN
6QLO
6QLP
6QLR
6QLT
```

The same constants file defines `6QLQ` and `6QLU` as skipped sample IDs. The processed metadata file `data/MMPBSA/processed/metadata.csv` contains five processed graph rows, one for each valid complex. The preprocessing failure file `data/MMPBSA/processed/failures.json` contains `[]`, indicating that no preprocessing failures are recorded in the current repository.

Crystal input files present in the repository include:

```text
data/MMPBSA/6QLN/complex.pdb
data/MMPBSA/6QLO/complex.pdb
data/MMPBSA/6QLP/complex.pdb
data/MMPBSA/6QLR/complex.pdb
data/MMPBSA/6QLT/complex.pdb
```

Each of these sample directories also contains `mmpbsa.out`. Experimental Delta G values are stored in `data/MMPBSA/bd` and are parsed by `parse_experimental_bd_table` in `code/binding_graph_preprocessing/labels.py`.

Processed crystal graphs are stored under:

```text
data/MMPBSA/processed/graphs/
```

with the following files:

```text
6QLN.pt
6QLO.pt
6QLP.pt
6QLR.pt
6QLT.pt
```

The processed metadata table reports the following graph sizes:

| sample_id | ligand_resname | ligand_resid | nodes | edges | ligand atoms | protein atoms | metal atoms | y_exp (kcal/mol) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6QLN | LIG | 139 | 277 | 6548 | 52 | 225 | 0 | -6.381 |
| 6QLO | LIG | 139 | 280 | 6438 | 52 | 228 | 0 | -6.931 |
| 6QLP | LIG | 139 | 298 | 7050 | 52 | 246 | 0 | -7.242 |
| 6QLR | LIG | 139 | 277 | 6604 | 52 | 225 | 0 | -6.620 |
| 6QLT | J5W | 139 | 291 | 6770 | 52 | 239 | 0 | -5.545 |

The metadata file contains absolute paths from the environment in which preprocessing was generated. The repository-local files are present under `data/MMPBSA/` and `data/MMPBSA/processed/`.

MD frame data are stored under:

```text
data/md_frame_exports/<sample_id>/
```

For each of the five sample IDs, the repository contains:

```text
frame_200.pdb
frame_250.pdb
frame_300.pdb
frame_350.pdb
frame_400.pdb
snapshot_energy_summary.csv
snapshot_energy_summary.md
```

Thus, five MD frames per sample are available in the current repository. The frame-level label parser in `code/md_frame_labels.py` reads `snapshot_energy_summary.csv` and selects rows with `method == "PB"`. The CSV columns include delta quantities such as `delta_vdwaals`, `delta_eel`, `delta_epb`, `delta_ecavity`, `delta_edisper`, and `delta_g_total`. The file names and parser indicate that these frame-level values are binding deltas; the CSV column names and project script use delta values corresponding to complex - receptor - ligand. The exact upstream computation procedure used to create `snapshot_energy_summary.csv` is not fully documented in the current repository beyond the exported CSV/Markdown files and `code/scripts/prepare_md_frame_exports.py`.

## 5. Label Definitions

### 5.1 Experimental Binding Free Energy

Experimental binding free energy is the primary prediction target. It is loaded from `data/MMPBSA/bd` by `parse_experimental_bd_table` in `code/binding_graph_preprocessing/labels.py`. The parser reads tab-delimited rows, uses the first column as the PDB/sample ID, and converts the second-column value from kJ/mol to kcal/mol by dividing by 4.184. The processed graphs store this value as `y_exp`.

### 5.2 Average PB/MM-PBSA Labels

Average PB/MM-PBSA labels used by Baseline 2-PB, Baseline 3, and Baseline 4 are parsed by `parse_average_pb_labels` in `code/md_frame_labels.py`. The parser explicitly searches for the `POISSON BOLTZMANN` section and then the `Differences (Complex - Receptor - Ligand)` block. It does not read the Complex block for these average PB labels.

The implemented target vector is:

```text
[vdw, elec, polar_solv, nonpolar_solv, dispersion, total]
```

The implemented mapping is:

| `mmpbsa.out` key | model target |
|---|---|
| `VDWAALS` | `vdw` |
| `EEL` | `elec` |
| `EPB` | `polar_solv` |
| `ENPOLAR` | `nonpolar_solv` |
| `EDISPER` | `dispersion` |
| `TOTAL` or `DELTA TOTAL` | `total` |

The run configuration files for Baseline 2-PB also record `pb_source_section` as `POISSON BOLTZMANN` and `pb_source_block` as `Differences (Complex - Receptor - Ligand)`.

The legacy preprocessing path in `code/binding_graph_preprocessing/labels.py` also parses GB auxiliary labels from the `GENERALIZED BORN` Differences block. Those graph-level labels are stored as `y_vdw`, `y_elec`, `y_polar`, `y_nonpolar`, and `y_aux` in processed crystal graphs. This legacy GB target path is separate from the current Baseline 2-PB average PB parser.

### 5.3 Frame-Level PB/MM-PBSA Labels

Frame-level labels are parsed from `snapshot_energy_summary.csv` by `parse_frame_pb_labels` in `code/md_frame_labels.py`. The parser only uses rows where `method == "PB"`. The implemented mapping is:

| `snapshot_energy_summary.csv` field | model target |
|---|---|
| `delta_vdwaals` | `vdw` |
| `delta_eel` | `elec` |
| `delta_epb` | `polar_solv` |
| `delta_ecavity` | `nonpolar_solv` |
| `delta_edisper` | `dispersion` |
| `delta_g_total` | `total` |

The average-level parser maps `ENPOLAR` to `nonpolar_solv`, whereas the frame-level parser maps `delta_ecavity` to `nonpolar_solv`. This naming difference is explicitly described in the module docstring of `code/md_frame_labels.py`.

## 6. Graph Construction

Graph construction is implemented in `code/binding_graph_preprocessing/structure.py`, `code/binding_graph_preprocessing/graph.py`, and `code/binding_graph_preprocessing/featurizer.py`.

PDB parsing is performed by `parse_pdb_file` in `structure.py`. The parser reads both `ATOM` and `HETATM` records. It keeps alternate locations only when the `altloc` field is empty or `A`. Thus, ligand atoms do not have to be exclusively represented as `HETATM`; ligand selection is based on residue identifiers after parsing both record types.

Ligand atoms are selected by `select_ligand_atoms`. The implemented selection procedure first attempts an exact match using ligand residue name, residue ID, and optional chain. If that fails, it falls back to residue ID among non-standard-amino-acid, non-water, non-metal atoms. If that also fails, it falls back to residue name. If no ligand atoms can be identified, preprocessing raises an error.

The default preprocessing CLI in `code/binding_graph_preprocessing/cli.py` uses ligand residue name `J5W`, residue ID `139`, and no chain. The MD-frame dataset in `code/md_frame_dataset.py` implements the following ligand mapping:

| sample_id | ligand mapping |
|---|---|
| 6QLN | `(LIG, 139, None)` |
| 6QLO | `(LIG, 139, None)` |
| 6QLP | `(LIG, 139, None)` |
| 6QLR | `(LIG, 139, None)` |
| 6QLT | `(J5W, 139, None)` |

The local pocket is extracted by `select_pocket_protein_atoms`. If any atom in a protein residue is within the pocket cutoff of any ligand atom, the complete protein residue is retained. The default pocket cutoff is 5.0 Angstrom. Metals are retained by `select_pocket_metal_atoms` if they are within the same ligand-pocket cutoff, provided `keep_metals` is true. The current processed metadata reports zero retained metal atoms for all five processed crystal graphs.

Nodes represent atoms. The node order is ligand atoms, followed by protein pocket atoms, followed by retained pocket metals. The corresponding node types are:

```text
0 = ligand
1 = protein
2 = metal
```

Edges are added bidirectionally in `build_complex_graph`:

- ligand covalent edges are built using ligand chemistry resolved in the order RDKit, PDB `CONECT`, then geometric covalent inference;
- protein and metal spatial edges connect retained protein/metal nodes within the protein edge cutoff, default 4.5 Angstrom;
- ligand-protein spatial edges connect ligand nodes to retained protein/metal nodes within the ligand-protein edge cutoff, default 5.0 Angstrom.

Node features are built by `build_node_feature_vector` in `featurizer.py`. The node feature vector has 33 dimensions and contains atomic number, formal charge, aromatic flag, hybridization index, degree, ring flag, total hydrogens, ligand/protein/metal indicators, residue type index, backbone flag, donor/acceptor indicators, and a 19-category element one-hot vector. Edge features are built by `build_edge_feature_vector` and have 12 dimensions: edge type index, distance, edge type indicator flags, aromatic/conjugated bond flags, and a five-category bond type one-hot vector.

Crystal graphs are built from `complex.pdb`. MD frame graphs are built in `code/md_frame_dataset.py` using the same graph construction functions, but from the five exported frame PDB files. The frame graph construction call fills graph-level energy labels with zeros because frame-level PB targets are stored separately as `y_frame_pb` in the dataset record.

## 7. Model Architecture

The current baseline models use a shared graph architecture based on the `SharedGINEncoder` class in `code/model_baseline3.py`. The encoder uses:

- an input linear projection from node feature dimension to hidden dimension;
- a stack of `torch_geometric.nn.GINConv` layers;
- ReLU activation;
- dropout after each GIN layer;
- `global_mean_pool` to obtain a graph-level embedding.

The default saved configurations for formal runs use hidden dimension 64, two GIN layers, dropout 0.0, batch size 1, and learning rate 0.001. These values are recorded in the `run_config.json` files under `results/training_runs/`.

In this report, `G_crystal` denotes the graph constructed from the crystal complex structure. The learned graph-level embedding produced by the encoder is denoted `h_crystal`. For MD-frame baselines, `G_frame` denotes an MD snapshot graph and `h_frame` denotes its learned graph-level embedding. In Baseline 4, `h_teacher` denotes the mean of the five MD-frame embeddings for a sample, and `z_crystal` denotes the projected crystal embedding. An embedding is therefore a learned graph-level representation after GIN encoding and mean pooling.

The current GIN encoder uses `data.x`, `data.edge_index`, and `data.batch`. The saved graph files also include `pos` and `edge_attr`, but the implemented `SharedGINEncoder` does not directly use positions or edge attributes.

## 8. Baseline Models

### 8.1 Baseline 1: Crystal-Only Prediction

Baseline 1 is implemented in `code/model_baseline1.py` and `code/train_baseline1.py`. Its input is only the crystal graph `G_crystal`. The model applies the shared GIN encoder to produce `h_crystal` and uses a linear head to predict experimental Delta G. Its output is `pred_exp`. The loss is the mean squared error on experimental Delta G, denoted:

```text
L_exp = MSE(pred_exp, y_exp)
```

The implementation does not use MM/PBSA auxiliary labels or MD frames for Baseline 1.

### 8.2 Baseline 2-PB: Crystal Prediction with Average PB/MM-PBSA Auxiliary Supervision

Baseline 2-PB is implemented in `code/model_baseline2_pb.py` and `code/train_baseline2_pb.py`. It uses only the crystal graph as model input. The model predicts both experimental Delta G and the six-dimensional average PB/MM-PBSA target vector:

```text
[vdw, elec, polar_solv, nonpolar_solv, dispersion, total]
```

The training loss implemented in `compute_losses` is:

```text
L_total = L_exp + lambda_avg * L_avg_pb
```

The default `lambda_avg` is 0.1, as shown in `train_baseline2_pb.py` and the saved run configuration files. No MD frames or frame-level PB labels are used by Baseline 2-PB. Inference remains crystal-only.

The older `code/train_baseline.py` and `code/model.py` files are preserved as a legacy MM/GBSA multitask workflow. This report treats that workflow as legacy/ablation rather than one of the four current formal baselines.

### 8.3 Baseline 3: Crystal Final Prediction with MD Frame PB/MM-PBSA Auxiliary Supervision

Baseline 3 is implemented in `code/model_baseline3.py`, `code/train_baseline3.py`, and `code/md_frame_dataset.py`. During training, each sample provides one crystal graph and five MD frame graphs. The same `SharedGINEncoder` is used for both crystal and frame graphs. The crystal branch predicts experimental Delta G and average PB/MM-PBSA labels. The frame branch predicts frame-level PB/MM-PBSA labels.

The implemented training loss is:

```text
L_total = L_exp + lambda_avg * L_avg_pb + lambda_frame * L_frame_pb
```

The default saved configuration uses `lambda_avg = 0.1` and `lambda_frame = 0.03`. Inference uses only the crystal graph, the shared encoder, and the crystal prediction heads. MD frames are training-only.

### 8.4 Baseline 4: Baseline 3 with Representation Distillation

Baseline 4 is implemented in `code/model_baseline4.py` and `code/train_baseline4.py`. It reuses the Baseline 3 data and supervision structure but adds an explicit representation distillation loss. The model encodes the five MD frame graphs, reshapes their embeddings by sample, and computes:

```text
h_teacher = mean(h_200, h_250, h_300, h_350, h_400)
```

The crystal embedding is passed through a projector:

```text
z_crystal = Projector(h_crystal)
```

The distillation loss implemented in `compute_training_losses` is:

```text
L_distill = MSE(z_crystal, stop_gradient(h_teacher))
```

The code implements this as `mse(z_crystal, h_teacher.detach())`. The teacher is recomputed during each forward pass from the current frame embeddings. The frame PB loss remains active and continues to train the frame branch and shared encoder.

The implemented total loss is:

```text
L_total = L_exp + lambda_avg * L_avg_pb + lambda_frame * L_frame_pb + lambda_distill * L_distill
```

The saved Baseline 4 configurations use `warmup_epochs = 100`, `lambda_avg = 0.02`, `lambda_frame = 0.01`, and `lambda_distill = 0.01`. During the warm-up period, `distill_active` is false and the distillation loss is zero. After warm-up, the distillation term is added. Inference remains crystal-only.

Baseline 4 should be interpreted as an ablation-style extension. It tests whether explicit MD-ensemble representation alignment adds value beyond the shared-encoder auxiliary learning already present in Baseline 3.

## 9. Training Protocol

All four current baselines use the Adam optimizer, as imported from `torch.optim` in the training scripts. The formal saved runs use learning rate 0.001 and batch size 1. Baseline 1, Baseline 2-PB, and Baseline 4 use 500 epochs in the saved formal runs, while Baseline 3 uses 300 epochs. Baseline 4 uses 100 warm-up epochs before distillation becomes active.

The training scripts save outputs under run-specific directories in `results/training_runs/`. Each formal run includes `run_config.json`, `split_info.json`, `train_log.csv`, `best_model.pt`, `best_predictions.csv`, `graph_debug_summary.csv`, and `label_normalization_stats.json`. Baseline 3 and Baseline 4 also save `best_validation_predictions.csv`.

Early stopping is not implemented in the current training scripts. Training continues for the configured number of epochs. However, each script saves a best checkpoint according to a validation metric:

- Baseline 1 saves `best_model.pt` when `val_L_exp` improves.
- Baseline 2-PB saves `best_model.pt` when `val_L_exp` improves.
- Baseline 3 saves `best_model.pt` when `eval_L_total` improves.
- Baseline 4 saves `best_model.pt` when `eval_exp_rmse_kcal` improves.

The saved `best_predictions.csv` files are generated from the best checkpoint loaded after training.

## 10. Label Normalization

Label normalization is implemented in `code/normalization_baseline3.py`. The saved run configurations for all four current baselines have `normalize_labels = true`.

Baseline 1 uses `ExpLabelNormalizer`, which computes training-set mean and standard deviation for experimental Delta G only. Baseline 2-PB uses `ExpAvgPBLabelNormalizer`, which computes training-set statistics for experimental Delta G and the average PB target vector. Baseline 3 and Baseline 4 use `LabelNormalizer`, which computes training-set statistics for experimental Delta G, average PB labels, and frame-level PB labels.

Statistics are computed only from training samples and then applied to train, validation, and test batches. The helper `_safe_std` replaces zero standard deviations with one. Each run saves normalization statistics to `label_normalization_stats.json`.

Predictions are denormalized before being written to prediction CSV files. This is visible in the evaluation and prediction-saving paths of the training scripts; for example, Baseline 1 denormalizes experimental predictions before writing `best_predictions.csv`, and Baseline 2-PB denormalizes both experimental and average PB predictions.

## 11. Splitting and Evaluation Protocol

The rotating split logic is implemented in `code/splits_baseline3.py` and is reused across the current baseline scripts. The sample order is:

```text
6QLN, 6QLO, 6QLP, 6QLR, 6QLT
```

For `rotating_train_val_test`, round `r` uses sample `r` as test and the next sample in the order as validation. The remaining three samples are used for training. The five formal split definitions saved in `split_info.json` are:

| round | train samples | validation sample | test sample |
|---:|---|---|---|
| 1 | 6QLP, 6QLR, 6QLT | 6QLO | 6QLN |
| 2 | 6QLN, 6QLR, 6QLT | 6QLP | 6QLO |
| 3 | 6QLN, 6QLO, 6QLT | 6QLR | 6QLP |
| 4 | 6QLN, 6QLO, 6QLP | 6QLT | 6QLR |
| 5 | 6QLO, 6QLP, 6QLR | 6QLN | 6QLT |

Splitting is by complex/sample ID. For MD-enhanced baselines, all frames from a given complex remain in that complex's split, avoiding frame-level leakage across train, validation, and test.

The train set updates model parameters. The validation set selects the best checkpoint. The test set is used for final reporting through the saved `best_predictions.csv` files and the merged evaluation CSV files.

The evaluation scripts compute MAE, RMSE, and Pearson correlation where implemented. Baseline 1 and Baseline 2-PB summary files include MAE, RMSE, and Pearson correlation for experimental Delta G. Baseline 3 and Baseline 4 summary files include RMSE and Pearson correlation; their experimental MAE is not present in the summary CSV but can be computed from the saved merged prediction files. Parity plots are generated as PNG files for each baseline.

Because the current test summary contains only five held-out predictions, Pearson correlation is unstable and should be interpreted as descriptive rather than definitive.

## 12. Results

### Current Result Status

The repository contains completed five-round rotating split results for Baseline 1, Baseline 2-PB, Baseline 3, and Baseline 4. The formal summary and merged prediction files are located in `results/training_runs/`. No final results beyond these saved files are reported here.

The best epochs available from saved training logs are:

| Model | round 1 | round 2 | round 3 | round 4 | round 5 |
|---|---:|---:|---:|---:|---:|
| Baseline 1 | 382 | 75 | 373 | 6 | 439 |
| Baseline 2-PB | 219 | 3 | 301 | 448 | 1 |
| Baseline 3 | 59 | 23 | 299 | 48 | 47 |
| Baseline 4 | 478 | 12 | 417 | 6 | 464 |

For Baseline 4, the best epochs above are derived from the minimum `eval_exp_rmse_kcal` in each saved `train_log.csv`, matching the model selection rule in `code/train_baseline4.py`.

### Table 1: Experimental Delta G Prediction Performance

| Model | Split protocol | Test MAE | Test RMSE | Pearson r | Notes |
|---|---|---:|---:|---:|---|
| Baseline 1 | 5-round rotating train/val/test | 0.509 | 0.629 | 0.358 | Values from `baseline1_summary_metrics.csv`. |
| Baseline 2-PB | 5-round rotating train/val/test | 0.269 | 0.430 | 0.753 | Values from `baseline2_pb_summary_metrics.csv`; strongest current saved experimental Delta G performance. |
| Baseline 3 | 5-round rotating train/val/test | 0.519 | 0.571 | 0.327 | RMSE and r from `baseline3_summary_metrics.csv`; MAE computed from `baseline3_merged_predictions.csv` because MAE is not present in the summary file. |
| Baseline 4 | 5-round rotating train/val/test | 0.470 | 0.617 | 0.413 | RMSE and r from `baseline4_summary_metrics.csv`; MAE computed from `baseline4_merged_predictions.csv` because MAE is not present in the summary file. |

### Table 2: Per-Sample Test Predictions

All values are in kcal/mol. Absolute errors are shown in parentheses after each prediction.

| sample_id | true_exp | B1 pred (abs err) | B2-PB pred (abs err) | B3 pred (abs err) | B4 pred (abs err) |
|---|---:|---:|---:|---:|---:|
| 6QLN | -6.381 | -6.124 (0.257) | -6.354 (0.027) | -7.168 (0.787) | -6.229 (0.152) |
| 6QLO | -6.931 | -6.950 (0.019) | -7.011 (0.079) | -6.769 (0.163) | -7.012 (0.081) |
| 6QLP | -7.242 | -6.830 (0.412) | -7.038 (0.204) | -6.675 (0.567) | -7.021 (0.220) |
| 6QLR | -6.620 | -5.611 (1.009) | -6.518 (0.103) | -6.285 (0.335) | -5.562 (1.058) |
| 6QLT | -5.545 | -6.394 (0.850) | -6.474 (0.930) | -6.288 (0.743) | -6.385 (0.840) |

Baseline 2-PB improves over Baseline 1 in the saved experimental Delta G metrics. Baseline 3 does not improve over Baseline 2-PB in the current saved results. Baseline 4 does not improve over Baseline 3 in RMSE, although it has a higher Pearson correlation than Baseline 3 in the current five-sample summary. These comparisons are preliminary because the dataset is extremely small.

## 13. Discussion

The current saved results suggest that average PB/MM-PBSA auxiliary supervision is beneficial relative to the crystal-only Baseline 1. Baseline 2-PB has lower MAE and RMSE and higher Pearson correlation than Baseline 1 in the saved five-round test summary.

The MD frame-level supervision used in Baseline 3 does not improve experimental Delta G prediction relative to Baseline 2-PB in the current saved results. This does not rule out the value of MD-derived information in larger datasets or with different architectures. It indicates only that the current implementation and five-complex dataset do not show an improvement over the average-PB auxiliary baseline.

Baseline 4 is an ablation-style extension that tests explicit alignment between the crystal representation and the mean MD-frame representation. In the saved results, Baseline 4 does not improve RMSE relative to Baseline 3. This is still informative: Baseline 3 may already encode useful frame-derived information through the shared encoder and frame-level auxiliary loss, or the small dataset may be insufficient to identify benefits from representation distillation.

The auxiliary PB/MM-PBSA losses should be interpreted as noisy physics-derived supervision rather than exact physical truth. MM/PBSA decomposition terms may help regularize representation learning, but they may also introduce noise or objectives that are not fully aligned with experimental affinity.

## 14. Overfitting and Diagnostics

Overfitting is monitored through the saved `train_log.csv` files. These logs include training loss, validation/evaluation loss, weighted auxiliary losses, and model-selection information. Baseline 1 and Baseline 2-PB logs include `best_epoch` and `best_val_L_exp`. Baseline 3 logs include training and evaluation total losses; best epochs can be identified from the minimum `eval_L_total`. Baseline 4 logs include `eval_exp_rmse_kcal`, and model selection is based on the minimum of this value.

Overfitting would appear as decreasing training loss with increasing validation loss. The scripts reduce the impact of late-epoch overfitting by saving and reloading the best validation checkpoint before final prediction CSVs are written. However, early stopping is not implemented, so training still continues to the configured final epoch.

The dataset size makes overfitting likely. Each validation set contains one complex, so the selected epoch may also be sensitive to single-sample validation noise. Saved prediction files and previously inspected train logs indicate that several runs have very early or very late best epochs. These diagnostics should be considered when interpreting the test metrics.

## 15. Limitations

The current MD-enhanced dataset is very small. The formal rotating evaluation contains only five complexes and one held-out test sample per round. The summary correlation values are therefore unstable and should not be overinterpreted.

MM/PBSA and PB/MM-PBSA labels are approximate physics-derived quantities. They may provide useful structure, but they are not exact experimental labels. The auxiliary objectives may regularize learning, but they may also introduce bias or noise.

Baseline 4 is an ablation-style method. Its failure to improve over Baseline 3 in the current saved RMSE does not imply that representation distillation is generally ineffective; it may be redundant with Baseline 3's shared-encoder frame supervision or underpowered in a five-complex dataset.

The current GIN encoder does not directly use saved 3D coordinates or edge attributes. The graph topology encodes distance-based edges, but the model does not explicitly consume the edge distance feature. Geometric architectures or edge-aware message passing are not yet implemented in the current baseline models.

## 16. Future Work

Future work should expand the dataset size and evaluate the models on larger collections. Larger resources such as PLAS could be used for average PB/MM-PBSA auxiliary supervision if compatible labels are available or can be generated. MD frames and frame-level PB labels should be generated for more complexes before drawing strong conclusions about MD-derived supervision.

Several modeling extensions are natural next steps. These include separate teacher and student encoders, attention pooling over frame embeddings, explicit use of edge features and distances, and derived PB features such as net polar, net nonpolar, gas-phase, and solvation terms. Stronger overfitting control should also be tested, including early stopping, regularization, reduced model capacity, and repeated validation protocols. External baselines should be included when suitable benchmark data are available.

## 17. Conclusion

This project implements a hierarchy of four graph neural network baselines for protein-ligand binding affinity prediction. All four baselines preserve crystal-only inference. Physics-derived PB/MM-PBSA quantities are used as auxiliary supervision during training, and MD frame-level supervision is introduced in Baseline 3. Baseline 4 further tests explicit representation distillation from an MD-frame ensemble representation to the crystal representation.

The current saved results show that Baseline 2-PB has the strongest experimental Delta G performance among the four implemented baselines on the five-complex rotating evaluation. Baseline 3 and Baseline 4 provide implemented mechanisms for MD-derived supervision and representation distillation, but they do not improve the current saved RMSE relative to Baseline 2-PB. Because the current dataset contains only five complexes, these findings are preliminary and should be treated as evidence from the current repository rather than as general conclusions.
