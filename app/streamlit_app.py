"""Local Streamlit demo for crystal-only binding affinity prediction."""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import streamlit as st

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from predict_complex import DEFAULT_TEMPERATURE_K, PB_TARGET_KEYS, predict_from_pdb  # noqa: E402


RESULTS_ROOT = REPO_ROOT / "results" / "training_runs"


def make_preset(model_type: str, run_name: str) -> dict[str, Path | str]:
    run_dir = RESULTS_ROOT / run_name
    return {
        "model_type": model_type,
        "run_dir": run_dir,
        "checkpoint_path": run_dir / "best_model.pt",
        "normalization_stats_path": run_dir / "label_normalization_stats.json",
    }


# These demo checkpoints are chosen to exercise the local web workflow. They are
# not intended as a formal model-selection protocol for benchmark reporting.
MODEL_PRESETS = {
    "Baseline 1": make_preset(
        "baseline1",
        "baseline1_rotating_train_val_test_round_2_val_6QLP_test_6QLO",
    ),
    "Baseline 2": make_preset(
        "baseline2_pb",
        "baseline2_pb_rotating_train_val_test_round_1_val_6QLO_test_6QLN",
    ),
    "Baseline 3": make_preset(
        "baseline3",
        "baseline3_rotating_train_val_test_round_2_val_6QLP_test_6QLO",
    ),
    "Baseline 4": make_preset(
        "baseline4",
        "baseline4_rotating_train_val_test_round_2_val_6QLP_test_6QLO",
    ),
}


def path_status(path: Path, required: bool = True) -> None:
    if path.exists():
        st.sidebar.success(f"Found: `{path}`")
    elif required:
        st.sidebar.error(f"Missing: `{path}`")
    else:
        st.sidebar.warning(f"Optional file not found: `{path}`")


def graph_summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    labels = {
        "num_nodes": "nodes",
        "num_edges": "directed edges",
        "num_ligand_atoms": "ligand atoms",
        "num_protein_atoms": "protein atoms",
        "num_metal_atoms": "metal atoms",
        "node_feature_dim": "node feature dim",
        "edge_feature_dim": "edge feature dim",
    }
    return [{"item": labels.get(key, key), "value": value} for key, value in summary.items()]


def pb_rows(pred_avg_pb: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"term": key, "predicted value (kcal/mol)": f"{float(pred_avg_pb[key]):.4f}"}
        for key in PB_TARGET_KEYS
        if key in pred_avg_pb
    ]


def render_result(result: dict[str, object]) -> None:
    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Delta G (kcal/mol)", f"{float(result['pred_exp_kcal']):.3f}")
    col2.metric("Delta G (kJ/mol)", f"{float(result['pred_exp_kj']):.3f}")
    col3.metric("Estimated Kd", str(result["estimated_kd_display"]))
    st.caption(f"Estimated Kd is converted from predicted Delta G assuming T = {DEFAULT_TEMPERATURE_K} K.")

    pred_avg_pb = result.get("pred_avg_pb")
    if pred_avg_pb:
        st.subheader("Predicted Average PB/MM-PBSA Terms")
        st.table(pb_rows(pred_avg_pb))

    st.subheader("Graph Summary")
    st.table(graph_summary_rows(result["graph_summary"]))

    with st.expander("Debug details"):
        st.write(
            {
                "model_type": result["model_type"],
                "best_epoch": result["best_epoch"],
                "checkpoint_path": result["checkpoint_path"],
                "normalization_stats_path": result["normalization_stats_path"],
                "estimated_kd_molar": result["estimated_kd_molar"],
            }
        )


def main() -> None:
    st.set_page_config(page_title="Crystal-Only Binding Affinity Prediction Demo", layout="centered")
    st.title("Crystal-Only Binding Affinity Prediction Demo")
    st.write(
        "Upload one protein-ligand complex PDB containing both protein and ligand coordinates. "
        "Inference uses only the crystal complex graph."
    )

    st.sidebar.header("Model")
    model_label = st.sidebar.selectbox("Select model", list(MODEL_PRESETS))
    preset = MODEL_PRESETS[model_label]
    checkpoint_path = Path(preset["checkpoint_path"])
    normalization_stats_path = Path(preset["normalization_stats_path"])

    st.sidebar.caption(f"Internal model type: `{preset['model_type']}`")
    path_status(checkpoint_path, required=True)
    path_status(normalization_stats_path, required=True)

    with st.sidebar.expander("Run on LUNARC"):
        st.code(
            "streamlit run app/streamlit_app.py "
            "--server.address 127.0.0.1 --server.port 8501 --server.headless true",
            language="bash",
        )
        st.code("ssh -L 8501:127.0.0.1:8501 <user>@<lunarc-login-host>", language="bash")
        st.caption("If needed, install Streamlit in the project environment before launching the app.")

    uploaded_pdb = st.file_uploader("Upload complex.pdb", type=["pdb"])
    col1, col2, col3 = st.columns([1.2, 1.0, 1.0])
    ligand_resname = col1.text_input("Ligand residue name", value="LIG").strip()
    ligand_resid = int(col2.number_input("Ligand residue ID", min_value=-99999, max_value=99999, value=139, step=1))
    ligand_chain_text = col3.text_input("Ligand chain (optional)", value="").strip()
    ligand_chain = ligand_chain_text or None

    run_clicked = st.button("Run prediction", type="primary")
    if not run_clicked:
        return

    if uploaded_pdb is None:
        st.error("Please upload a single PDB file before running prediction.")
        return
    if not ligand_resname:
        st.error("Please provide the ligand residue name.")
        return
    if not checkpoint_path.exists():
        st.error(f"Preset checkpoint is missing: {checkpoint_path}")
        return
    if not normalization_stats_path.exists():
        st.error(f"Preset normalization stats file is missing: {normalization_stats_path}")
        return

    try:
        with tempfile.TemporaryDirectory(prefix="binding_affinity_demo_") as tmpdir:
            tmp_path = Path(tmpdir)
            pdb_path = tmp_path / "uploaded_complex.pdb"
            pdb_path.write_bytes(uploaded_pdb.getvalue())
            graph_path = tmp_path / "uploaded_complex.pt"
            with st.spinner("Building graph and running prediction..."):
                result = predict_from_pdb(
                    pdb_path=pdb_path,
                    ligand_resname=ligand_resname,
                    ligand_resid=ligand_resid,
                    ligand_chain=ligand_chain,
                    model_type=str(preset["model_type"]),
                    checkpoint_path=checkpoint_path,
                    normalization_stats_path=normalization_stats_path,
                    sample_id=Path(uploaded_pdb.name).stem or "uploaded_complex",
                    save_graph_path=graph_path,
                )
        render_result(result)
    except Exception as exc:  # pragma: no cover - Streamlit UI path
        st.error(f"Prediction failed: {exc}")
        with st.expander("Debug details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
