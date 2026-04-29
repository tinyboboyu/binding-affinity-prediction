"""Local Streamlit demo for crystal-only binding affinity prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from predict_complex import DEFAULT_TEMPERATURE_K, PB_TARGET_KEYS, predict_from_pdb  # noqa: E402


RESULTS_ROOT = REPO_ROOT / "results" / "training_runs"
PB_DISPLAY_NAMES = {
    "vdw": "VDWAALS",
    "elec": "EEL",
    "polar_solv": "EPB",
    "nonpolar_solv": "ENPOLAR",
    "dispersion": "EDISPER",
    "total": "DELTA TOTAL",
}
BASELINE_DESCRIPTIONS = {
    "Baseline 1": "Trained with complex structure and ΔG.",
    "Baseline 2": "Trained with complex structure, ΔG, and MMPBSA terms.",
    "Baseline 3": "Trained with complex structure, ΔG, MMPBSA terms, and MD frame-level MMPBSA terms.",
    "Baseline 4": "Based on Baseline 3 and uses representation distillation.",
}


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


def pb_rows(pred_avg_pb: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"term": PB_DISPLAY_NAMES[key], "predicted value (kcal/mol)": f"{float(pred_avg_pb[key]):.4f}"}
        for key in PB_TARGET_KEYS
        if key in pred_avg_pb
    ]


def render_baseline_descriptions() -> None:
    st.sidebar.subheader("Baseline Information")
    for label, description in BASELINE_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{label}**  \n{description}")


def render_result(result: dict[str, object]) -> None:
    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("ΔG (kcal/mol)", f"{float(result['pred_exp_kcal']):.3f}")
    col2.metric("ΔG (kJ/mol)", f"{float(result['pred_exp_kj']):.3f}")
    col3.metric("Estimated Kd", str(result["estimated_kd_display"]))
    st.caption(f"Estimated Kd is converted from predicted ΔG assuming T = {DEFAULT_TEMPERATURE_K} K.")

    pred_avg_pb = result.get("pred_avg_pb")
    if pred_avg_pb:
        st.subheader("Predicted Average PB/MM-PBSA Terms")
        st.caption("Source target: POISSON BOLTZMANN / Differences (Complex - Receptor - Ligand).")
        st.table(pb_rows(pred_avg_pb))


def main() -> None:
    st.set_page_config(page_title="Binding Affinity Prediction Demo", layout="centered")
    st.title("Binding Affinity Prediction Demo")
    st.write(
        "Upload one protein-ligand complex PDB containing both protein and ligand coordinates. "
        "Inference uses only the crystal complex graph."
    )

    st.sidebar.header("Model")
    model_label = st.sidebar.selectbox("Select model", list(MODEL_PRESETS))
    preset = MODEL_PRESETS[model_label]
    checkpoint_path = Path(preset["checkpoint_path"])
    normalization_stats_path = Path(preset["normalization_stats_path"])

    render_baseline_descriptions()

    uploaded_pdb = st.file_uploader("Upload complex.pdb", type=["pdb"])
    col1, col2, col3 = st.columns([1.2, 1.0, 1.0])
    ligand_resname = col1.text_input("Ligand residue name", value="").strip()
    ligand_resid_text = col2.text_input("Ligand residue ID", value="").strip()
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
    if not ligand_resid_text:
        st.error("Please provide the ligand residue ID.")
        return
    try:
        ligand_resid = int(ligand_resid_text)
    except ValueError:
        st.error("Ligand residue ID must be an integer.")
        return
    if not checkpoint_path.exists():
        st.error("The selected model checkpoint is unavailable.")
        return
    if not normalization_stats_path.exists():
        st.error("The selected model normalization statistics are unavailable.")
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


if __name__ == "__main__":
    main()
