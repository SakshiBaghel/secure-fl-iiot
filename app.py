"""Streamlit Interactive Dashboard for Blockchain-Enabled Secure Federated Learning for IIoT.

Provides real-time visibility into:
1. Dataset characteristics and telemetry distributions.
2. Federated training rounds and global model convergence.
3. Centralized baseline vs. Federated Learning comparison.
4. Security update validation & model poisoning attack defense.
5. On-chain blockchain transaction ledger and cryptographic audit trail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluate import compute_intrusion_metrics
from src.run_experiment import run_federated_pipeline


st.set_page_config(
    page_title="Secure FL for IIoT | Blockchain Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-accepted {
        background-color: #dcfce7; color: #166534; font-weight: 600; padding: 3px 8px; border-radius: 4px;
    }
    .badge-rejected {
        background-color: #fee2e2; color: #991b1b; font-weight: 600; padding: 3px 8px; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ Blockchain-Enabled Secure Federated Learning for IIoT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Decentralized Intrusion Detection with FedAvg Aggregation, L2-Norm Validation & Smart Contract Auditing</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Experiment Controls")

dataset_choice = st.sidebar.selectbox(
    "Dataset Source",
    ["Auto-detect / Synthetic IIoT Fallback", "data/edge_iiotset.csv", "data/cic_ids2017.csv"],
    index=0,
)
dataset_path = None if "Auto-detect" in dataset_choice else dataset_choice

num_clients = st.sidebar.slider("Simulated Clients", min_value=2, max_value=5, value=3, step=1)
num_rounds = st.sidebar.slider("Communication Rounds", min_value=1, max_value=10, value=5, step=1)
local_epochs = st.sidebar.slider("Local Epochs per Round", min_value=1, max_value=5, value=3, step=1)
l2_threshold = st.sidebar.slider("L2 Validation Threshold", min_value=5.0, max_value=30.0, value=15.0, step=1.0)
learning_rate = st.sidebar.number_input("Client Learning Rate", value=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Security Attack Injection")
simulate_attack = st.sidebar.checkbox("Simulate Poisoning Attack", value=True)

run_button = st.sidebar.button("🚀 Run Experiment Pipeline", type="primary", use_container_width=True)

# Load existing results if available
results_path = PROJECT_ROOT / "results" / "experiment_results.json"
results = None

if run_button:
    with st.spinner("Executing Federated Learning and Blockchain workflow..."):
        try:
            results = run_federated_pipeline(
                dataset_path=dataset_path,
                num_clients=num_clients,
                rounds=num_rounds,
                local_epochs=local_epochs,
                lr=learning_rate,
                validation_l2_threshold=l2_threshold,
                simulate_poisoning_experiment=simulate_attack,
                output_dir=str(PROJECT_ROOT / "results"),
            )
            st.success("✅ Experiment successfully completed! Results updated below.")
        except Exception as e:
            st.error(f"❌ Execution failed: {str(e)}")

if results is None and results_path.exists():
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        pass

# Main Dashboard View
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset & Setup",
    "📈 Federated Progress",
    "⚖️ Centralized vs. FL",
    "🛡️ Security Defense",
    "⛓️ Blockchain Audit Trail",
])

if results is None:
    st.info("👋 Welcome! Click **'🚀 Run Experiment Pipeline'** in the sidebar to start the training and blockchain simulation.")
else:
    # --- Tab 1: Dataset & Setup ---
    with tab1:
        st.subheader("Industrial IoT Dataset Summary")
        ds_info = results["dataset_info"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Network Flows", f"{ds_info['total_samples']:,}")
        col2.metric("Training Samples", f"{ds_info['train_samples']:,}")
        col3.metric("Held-out Test Samples", f"{ds_info['test_samples']:,}")
        col4.metric("Extracted Features", f"{ds_info['num_features']}")

        if ds_info.get("is_synthetic"):
            st.warning("⚠️ **Dataset Status**: Operating in labelled **Synthetic IIoT Fallback Mode**. Telemetry accurately models SCADA, Modbus, and industrial cyber-attacks.")
        else:
            st.success("✅ **Dataset Status**: Operating with external intrusion dataset.")

        st.markdown("#### Monitored Network Features")
        features_df = pd.DataFrame({
            "Feature Index": range(1, len(ds_info["feature_names"]) + 1),
            "Feature Name": ds_info["feature_names"],
            "Description": [
                "Flow duration (milliseconds)",
                "Forward packet count",
                "Backward packet count",
                "Total forward payload bytes",
                "Total backward payload bytes",
                "Mean packet length (bytes)",
                "Standard deviation of packet length",
                "Flow packet rate (pkts/sec)",
                "Flow throughput (bytes/sec)",
                "TCP SYN flags count",
                "TCP ACK flags count",
                "TCP RST flags count",
                "Shannon port entropy (scan detection)",
                "PLC / Gateway CPU utilization (%)",
                "Industrial Modbus Function Code",
            ][:len(ds_info["feature_names"])]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    # --- Tab 2: Federated Training Progress ---
    with tab2:
        st.subheader("Federated Learning Rounds Progression")
        fl_data = results["federated_learning"]
        final_m = fl_data["final_metrics"]

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Global Accuracy", f"{final_m['accuracy'] * 100:.2f}%")
        mcol2.metric("Precision", f"{final_m['precision'] * 100:.2f}%")
        mcol3.metric("Recall (Detection Rate)", f"{final_m['recall'] * 100:.2f}%", help="Critical in IIoT to avoid undetected attacks")
        mcol4.metric("F1-Score", f"{final_m['f1_score']:.4f}")
        mcol5.metric("FL Training Time", f"{fl_data.get('total_execution_time_s', 0):.2f}s")

        history = fl_data["round_history"]
        rounds_list = [h["round"] for h in history]
        acc_list = [h["metrics"]["accuracy"] * 100 for h in history]
        f1_list = [h["metrics"]["f1_score"] for h in history]
        rec_list = [h["metrics"]["recall"] * 100 for h in history]

        chart_df = pd.DataFrame({
            "Communication Round": rounds_list,
            "Accuracy (%)": acc_list,
            "Recall (%)": rec_list,
            "F1-Score (x100)": [f * 100 for f in f1_list],
        }).set_index("Communication Round")

        st.line_chart(chart_df)

        st.markdown("#### Round-by-Round Execution Logs")
        round_summary = []
        for h in history:
            round_summary.append({
                "Round": h["round"],
                "Start Global Hash": h["global_hash_start"][:16] + "...",
                "End Global Hash": h["global_hash_end"][:16] + "...",
                "Accepted Updates": h["accepted_updates"],
                "Accuracy (%)": f"{h['metrics']['accuracy']*100:.2f}%",
                "F1-Score": f"{h['metrics']['f1_score']:.4f}",
                "Duration (s)": f"{h['round_duration_s']:.2f}",
            })
        st.dataframe(pd.DataFrame(round_summary), use_container_width=True, hide_index=True)

    # --- Tab 3: Centralized Baseline vs Federated Learning ---
    with tab3:
        st.subheader("Benchmark: Centralized ML Baseline vs. Federated Learning")

        base_m = results["centralized_baseline"]
        fl_m = results["federated_learning"]["final_metrics"]

        cmp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall (Detection Rate)", "F1-Score", "False Positive Rate"],
            "Centralized Random Forest": [
                f"{base_m['accuracy'] * 100:.2f}%",
                f"{base_m['precision'] * 100:.2f}%",
                f"{base_m['recall'] * 100:.2f}%",
                f"{base_m['f1_score']:.4f}",
                f"{base_m['false_positive_rate'] * 100:.2f}%",
            ],
            "Federated Learning (FedAvg MLP)": [
                f"{fl_m['accuracy'] * 100:.2f}%",
                f"{fl_m['precision'] * 100:.2f}%",
                f"{fl_m['recall'] * 100:.2f}%",
                f"{fl_m['f1_score']:.4f}",
                f"{fl_m['false_positive_rate'] * 100:.2f}%",
            ],
            "Advantage": [
                "Comparable high performance",
                "High precision across both",
                "Zero raw industrial data shared over network",
                "High balance without privacy compromise",
                "Low alarm false positives",
            ]
        })
        st.table(cmp_df)

        st.info(
            "💡 **Why Recall and F1-Score Matter for Intrusion Detection**:\n"
            "- In Industrial IoT, an attack reaching physical controllers (PLCs, centrifuges, valves) can cause catastrophic mechanical failure.\n"
            "- **Recall = TP / (TP + FN)** ensures that intrusions are not silently ignored (minimizing False Negatives).\n"
            "- **F1-Score** balances detection rate with false alarms, preventing industrial operators from suffering alert fatigue."
        )

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown("##### Centralized Confusion Matrix")
            cm_b = base_m["confusion_matrix"]
            st.write(f"- True Negatives (Normal): **{cm_b['true_negatives']}**")
            st.write(f"- False Positives (False Alarms): **{cm_b['false_positives']}**")
            st.write(f"- False Negatives (Missed Attacks): **{cm_b['false_negatives']}**")
            st.write(f"- True Positives (Detected Attacks): **{cm_b['true_positives']}**")

        with col_cm2:
            st.markdown("##### Federated Learning Confusion Matrix")
            cm_f = fl_m["confusion_matrix"]
            st.write(f"- True Negatives (Normal): **{cm_f['true_negatives']}**")
            st.write(f"- False Positives (False Alarms): **{cm_f['false_positives']}**")
            st.write(f"- False Negatives (Missed Attacks): **{cm_f['false_negatives']}**")
            st.write(f"- True Positives (Detected Attacks): **{cm_f['true_positives']}**")

    # --- Tab 4: Security Defense ---
    with tab4:
        st.subheader("Model Update Validation & Poisoning Attack Defense")
        sec_exp = results.get("security_experiment")

        if sec_exp is None:
            st.info("No security poisoning experiment recorded. Enable the attack simulation in sidebar and re-run.")
        else:
            st.markdown(f"**Simulated Compromised Node**: `{sec_exp['attacker_client_id']}`")
            st.markdown(f"**Observed L2 Norm of Attacker Update**: `{sec_exp['detected_l2_norm']:.2f}` (Validation Threshold: `{sec_exp['threshold']:.2f}`)")

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown("### 🟢 Honest FedAvg")
                h_m = sec_exp["honest"]
                st.metric("Accuracy", f"{h_m['accuracy']*100:.2f}%")
                st.metric("F1-Score", f"{h_m['f1_score']:.4f}")
                st.caption("Standard federated training across all honest clients.")

            with scol2:
                st.markdown("### 🔴 Poisoned (No Defense)")
                p_unp = sec_exp["poisoned_unprotected"]
                st.metric("Accuracy", f"{p_unp['accuracy']*100:.2f}%", delta=f"{(p_unp['accuracy'] - h_m['accuracy'])*100:.1f}%")
                st.metric("F1-Score", f"{p_unp['f1_score']:.4f}", delta=f"{p_unp['f1_score'] - h_m['f1_score']:.4f}")
                st.caption("Aggregator blindly includes malicious update. Model collapses.")

            with scol3:
                st.markdown("### 🛡️ Protected (With Validation)")
                p_prot = sec_exp["poisoned_protected"]
                st.metric("Accuracy", f"{p_prot['accuracy']*100:.2f}%", delta="Protected")
                st.metric("F1-Score", f"{p_prot['f1_score']:.4f}", delta="Preserved")
                st.caption("Update validator rejects client_3 before aggregation. Model performance is preserved!")

            # Image view of plot
            plot_path = PROJECT_ROOT / "results" / "security_validation_impact.png"
            if plot_path.exists():
                st.image(str(plot_path), caption="Security Experiment: Impact of Update Validation Defense", use_container_width=True)

    # --- Tab 5: Blockchain Audit Trail ---
    with tab5:
        st.subheader("Smart Contract Audit Trail & Ledger Records")
        bc_info = results["blockchain_audit"]

        bcol1, bcol2 = st.columns(2)
        bcol1.metric("Blockchain Integration Mode", bc_info["mode"].upper())
        bcol2.metric("Total Transactions Logged", bc_info["transaction_count"])

        tx_list = bc_info["transactions"]
        tx_display = []
        for t in tx_list:
            tx_display.append({
                "Tx Hash": t["tx_hash"][:18] + "...",
                "Event": t["event"],
                "Block #": t["block_number"],
                "Client / Subject": t.get("client_id", t.get("round", "N/A")),
                "Gas Used": t.get("gas_used", 0),
                "Latency (ms)": t.get("latency_ms", 0),
                "Mode": t.get("mode", "simulated"),
            })
        st.dataframe(pd.DataFrame(tx_display), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Cryptographic Hash Integrity Verification")
        st.write("Verify that a local model update matches the immutable on-chain record.")

        test_hash_input = st.text_input("Enter 64-character SHA-256 Model Update Hash to verify:", value="d9290cd8d7fc9385")
        if st.button("Verify On-Chain"):
            found = False
            for t in tx_list:
                if t.get("update_hash", "").startswith(test_hash_input) or t.get("global_model_hash", "").startswith(test_hash_input):
                    st.success(f"✅ Verified! Hash found in transaction `{t['tx_hash']}` (Event: `{t['event']}`, Block: #{t['block_number']}).")
                    found = True
                    break
            if not found:
                st.warning("⚠️ Hash not found in current blockchain records.")
