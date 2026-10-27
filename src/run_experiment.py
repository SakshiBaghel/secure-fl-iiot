"""End-to-End Orchestrator for Blockchain-Enabled Secure Federated Learning in IIoT.

Executes:
1. Data loading, cleaning, and preprocessing with zero data leakage.
2. Training data partition among 3 simulated industrial clients.
3. Centralized baseline training (scikit-learn RandomForest) on held-out test set.
4. Federated training rounds with local PyTorch training and FedAvg.
5. Smart contract registration, round tracking, and update metadata logging via Web3.
6. Model update security validation (L2-norm deviation check).
7. Controlled security experiment demonstrating impact of validation under poisoning attack.
8. Evaluation on held-out test set and persistence of metrics and plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baseline import train_and_evaluate_centralized
from src.blockchain_client import BlockchainClient
from src.client import IndustrialClient
from src.evaluate import (
    compute_intrusion_metrics,
    print_metrics_summary,
    save_metrics_to_json,
)
from src.fedavg import evaluate_global_model, federated_averaging
from src.model import (
    IIoTIntrusionMLP,
    get_weights,
    hash_model_weights,
    set_weights,
)
from src.preprocess import load_dataset, partition_for_clients, preprocess_and_split
from src.validate_update import UpdateValidator, ValidationReason


def plot_experiment_results(
    history: List[Dict[str, Any]],
    baseline_metrics: Dict[str, Any],
    security_results: Optional[Dict[str, Any]] = None,
    output_dir: str = "results",
) -> None:
    """Generates informative visualization plots for the experiment."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rounds = [h["round"] for h in history]
    fl_acc = [h["metrics"]["accuracy"] * 100 for h in history]
    fl_f1 = [h["metrics"]["f1_score"] for h in history]
    fl_rec = [h["metrics"]["recall"] * 100 for h in history]

    # Plot 1: Federated Training Progress vs Centralized Baseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, fl_acc, marker="o", color="#1f77b4", linewidth=2, label="Federated Global Model")
    ax1.axhline(baseline_metrics["accuracy"] * 100, color="#d62728", linestyle="--", label=f"Centralized Baseline ({baseline_metrics['accuracy']*100:.1f}%)")
    ax1.set_title("Detection Accuracy across FL Rounds", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Communication Round", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_xticks(rounds)
    ax1.set_ylim([max(0, min(fl_acc) - 10), 105])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(rounds, fl_f1, marker="s", color="#2ca02c", linewidth=2, label="FL F1-Score")
    ax2.plot(rounds, [r / 100 for r in fl_rec], marker="^", color="#ff7f0e", linewidth=2, label="FL Recall (DR)")
    ax2.axhline(baseline_metrics["f1_score"], color="#d62728", linestyle="--", label=f"Centralized F1 ({baseline_metrics['f1_score']:.3f})")
    ax2.set_title("Recall & F1-Score (Intrusion Detection)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Communication Round", fontsize=10)
    ax2.set_ylabel("Score (0.0 - 1.0)", fontsize=10)
    ax2.set_xticks(rounds)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path / "fl_convergence_metrics.png", dpi=300)
    plt.close(fig)

    # Plot 2: Security Validation Impact (if security experiment conducted)
    if security_results is not None:
        fig2, ax = plt.subplots(figsize=(8, 5))
        scenarios = ["Honest FL", "Poisoned (No Defense)", "Poisoned (With Validation)"]
        acc_vals = [
            security_results["honest"]["accuracy"] * 100,
            security_results["poisoned_unprotected"]["accuracy"] * 100,
            security_results["poisoned_protected"]["accuracy"] * 100,
        ]
        f1_vals = [
            security_results["honest"]["f1_score"] * 100,
            security_results["poisoned_unprotected"]["f1_score"] * 100,
            security_results["poisoned_protected"]["f1_score"] * 100,
        ]

        x = np.arange(len(scenarios))
        width = 0.35

        rects1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy (%)", color="#3b82f6")
        rects2 = ax.bar(x + width/2, f1_vals, width, label="F1-Score (%)", color="#10b981")

        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title("Security Experiment: Model Poisoning Defense in FedAvg", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=10)
        ax.set_ylim([0, 115])
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Annotations
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig2.savefig(out_path / "security_validation_impact.png", dpi=300)
        plt.close(fig2)


def run_federated_pipeline(
    dataset_path: Optional[str] = None,
    num_clients: int = 3,
    rounds: int = 5,
    local_epochs: int = 3,
    lr: float = 0.005,
    batch_size: int = 32,
    validation_l2_threshold: float = 15.0,
    simulate_poisoning_experiment: bool = True,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs complete end-to-end Blockchain-Enabled Federated Learning workflow."""
    print("=" * 75)
    print("  BLOCKCHAIN-ENABLED SECURE FEDERATED LEARNING FOR IIoT")
    print("=" * 75)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading and Preprocessing
    df, is_synthetic = load_dataset(dataset_path=dataset_path, random_state=seed)
    print(f"\n[Step 1] Dataset Loaded: {len(df)} total flows. (Synthetic Fallback: {is_synthetic})")

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
        df, test_size=0.20, random_state=seed
    )
    print(f"[Step 1] Preprocessed: {X_train.shape[0]} training samples | {X_test.shape[0]} held-out test samples")
    print(f"[Step 1] Input Feature Dimensions: {X_train.shape[1]}")

    # 2. Centralized ML Baseline
    print("\n[Step 2] Training Centralized Baseline (Random Forest Benchmark)...")
    baseline_metrics, _ = train_and_evaluate_centralized(
        dataset_path=dataset_path,
        model_type="random_forest",
        random_state=seed,
        output_dir=output_dir,
    )

    # 3. Client Partitioning
    client_partitions = partition_for_clients(
        X_train, y_train, num_clients=num_clients, non_iid=False, random_state=seed
    )
    clients: List[IndustrialClient] = []
    authorized_client_ids = set()

    for idx, (x_part, y_part) in enumerate(client_partitions):
        cid = f"client_{idx + 1}"
        client_obj = IndustrialClient(
            client_id=cid,
            X_local=x_part,
            y_local=y_part,
            input_dim=len(feature_names),
        )
        clients.append(client_obj)
        authorized_client_ids.add(cid)
        print(f"[Step 3] Initialized {cid}: {client_obj.sample_count} local flows "
              f"({client_obj.benign_count} Benign, {client_obj.malicious_count} Attacks)")

    # 4. Blockchain & Smart Contract Integration
    print("\n[Step 4] Connecting to Blockchain...")
    bc_client = BlockchainClient()
    print(f"[Step 4] Blockchain Client initialized in '{bc_client.mode.upper()}' mode.")

    # Register participants
    for c in clients:
        addr = bc_client.client_account_map.get(c.client_id, "0x0000000000000000000000000000000000000000")
        reg_log = bc_client.register_participant(participant_address=addr, client_id=c.client_id, role=0)
        print(f"  -> On-Chain Registered {c.client_id} at {addr[:10]}... [Tx: {reg_log['tx_hash'][:14]}...]")

    # 5. Initialize Global Model & Validator
    global_model = IIoTIntrusionMLP(input_dim=len(feature_names))
    validator = UpdateValidator(
        max_l2_threshold=validation_l2_threshold,
        authorized_clients=authorized_client_ids,
    )

    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print(f"  STARTING FEDERATED LEARNING ({rounds} Communication Rounds)")
    print("=" * 75)

    start_fl_time = time.time()

    for r in range(1, rounds + 1):
        round_start_time = time.time()
        print(f"\n--- [Round {r}/{rounds}] ---")

        # Snapshot current global weights
        current_global_weights = get_weights(global_model)
        global_hash = hash_model_weights(current_global_weights)

        # On-chain round start record
        bc_client.start_training_round(round_number=r, global_model_hash=global_hash)
        print(f"[Round {r}] Global Model Hash (Start): {global_hash[:16]}...")

        # Client Training & Submission
        accepted_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        round_validations: List[Dict[str, Any]] = []

        for client in clients:
            # 1. Distribute current global weights to client
            client.set_parameters(current_global_weights)

            # 2. Local Training
            updated_weights, update_hash, sample_cnt = client.train(
                epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
            )

            # 3. Update Validation
            val_result = validator.validate_update(
                client_id=client.client_id,
                client_weights=updated_weights,
                global_weights=current_global_weights,
                expected_hash=update_hash,
                actual_hash=update_hash,
            )

            # 4. Record Update Metadata on Blockchain
            tx_rec = bc_client.record_update(
                round_number=r,
                client_id=client.client_id,
                update_hash=update_hash,
                sample_count=sample_cnt,
                is_valid=val_result.is_valid,
                reason_code=val_result.reason_code,
            )

            round_validations.append({
                "client_id": client.client_id,
                "update_hash": update_hash,
                "is_valid": val_result.is_valid,
                "reason_code": int(val_result.reason_code),
                "reason_message": val_result.reason_message,
                "l2_norm": round(val_result.l2_norm, 4),
                "tx_hash": tx_rec["tx_hash"],
            })

            if val_result.is_valid:
                accepted_updates.append((updated_weights, sample_cnt))
                print(f"  [{client.client_id}] Validated (L2={val_result.l2_norm:.3f}) -> ACCEPTED")
            else:
                print(f"  [{client.client_id}] Flagged ({val_result.reason_message}) -> REJECTED")

        # 5. FedAvg Aggregation of Accepted Updates
        if accepted_updates:
            new_global_weights = federated_averaging(accepted_updates)
            set_weights(global_model, new_global_weights)
            new_global_hash = hash_model_weights(new_global_weights)
        else:
            print(f"[Round {r}] WARNING: No updates accepted! Retaining previous global model.")
            new_global_hash = global_hash

        # 6. Complete Round on Blockchain
        bc_client.complete_round(
            round_number=r,
            new_global_model_hash=new_global_hash,
            accepted_count=len(accepted_updates),
        )

        # 7. Evaluate Global Model on Held-out Test Set
        eval_metrics = evaluate_global_model(
            model=global_model,
            X_test=X_test,
            y_test=y_test,
        )

        round_duration = time.time() - round_start_time

        print(f"[Round {r} Complete] Accuracy: {eval_metrics['accuracy']*100:.2f}% | "
              f"Precision: {eval_metrics['precision']*100:.2f}% | "
              f"Recall (DR): {eval_metrics['recall']*100:.2f}% | "
              f"F1: {eval_metrics['f1_score']:.4f} | Time: {round_duration:.2f}s")

        history.append({
            "round": r,
            "global_hash_start": global_hash,
            "global_hash_end": new_global_hash,
            "accepted_updates": len(accepted_updates),
            "validations": round_validations,
            "metrics": eval_metrics,
            "round_duration_s": round(round_duration, 2),
        })

    total_fl_time = time.time() - start_fl_time
    final_metrics = history[-1]["metrics"]

    print("\n" + "=" * 75)
    print("  FEDERATED LEARNING COMPLETED")
    print("=" * 75)
    print_metrics_summary("Final Federated Global Model Evaluation", final_metrics)
    print(f"Total FL Pipeline Execution Time: {total_fl_time:.2f} seconds")

    # 6. Controlled Security Poisoning Experiment
    security_comparison = None
    if simulate_poisoning_experiment:
        print("\n" + "=" * 75)
        print("  CONTROLLED SECURITY EXPERIMENT: MODEL POISONING DEFENSE")
        print("=" * 75)
        print("Simulating compromised malicious node ('client_3') injecting poisoned weights...")

        # Base snapshot
        base_weights = get_weights(global_model)

        # 1. Honest baseline outcome (current final_metrics)
        honest_eval = final_metrics

        # 2. Poisoned update without validation defense
        print("\nScenario A: Aggregating poisoned update WITHOUT validation defense...")
        poisoned_model_unprotected = IIoTIntrusionMLP(input_dim=len(feature_names))
        set_weights(poisoned_model_unprotected, base_weights)

        # Honest updates from client 1 & 2
        clients[0].set_parameters(base_weights)
        w1, _, c1 = clients[0].train(epochs=1, lr=lr)

        clients[1].set_parameters(base_weights)
        w2, _, c2 = clients[1].train(epochs=1, lr=lr)

        # Poisoned update from client 3
        clients[2].set_parameters(base_weights)
        w3_poison, h3_poison, c3 = clients[2].craft_poisoned_update(scale=20.0)

        # Unprotected aggregation (blindly includes poisoned update)
        unprotected_weights = federated_averaging([
            (w1, c1),
            (w2, c2),
            (w3_poison, c3),
        ])
        set_weights(poisoned_model_unprotected, unprotected_weights)
        poisoned_unprotected_eval = evaluate_global_model(
            model=poisoned_model_unprotected,
            X_test=X_test,
            y_test=y_test,
        )
        print(f" -> Result WITHOUT Defense: Accuracy = {poisoned_unprotected_eval['accuracy']*100:.2f}%, F1 = {poisoned_unprotected_eval['f1_score']:.4f}")

        # 3. Poisoned update WITH validation defense
        print("\nScenario B: Aggregating WITH update validation defense...")
        poisoned_model_protected = IIoTIntrusionMLP(input_dim=len(feature_names))
        set_weights(poisoned_model_protected, base_weights)

        # Run validation
        val_w1 = validator.validate_update("client_1", w1, base_weights)
        val_w2 = validator.validate_update("client_2", w2, base_weights)
        val_w3 = validator.validate_update("client_3", w3_poison, base_weights)

        print(f"  [Validator] client_1: Valid={val_w1.is_valid} (L2={val_w1.l2_norm:.2f})")
        print(f"  [Validator] client_2: Valid={val_w2.is_valid} (L2={val_w2.l2_norm:.2f})")
        print(f"  [Validator] client_3: Valid={val_w3.is_valid} (L2={val_w3.l2_norm:.2f}) -> Flagged: {val_w3.reason_message}")

        protected_updates = []
        if val_w1.is_valid:
            protected_updates.append((w1, c1))
        if val_w2.is_valid:
            protected_updates.append((w2, c2))
        if val_w3.is_valid:
            protected_updates.append((w3_poison, c3))

        protected_weights = federated_averaging(protected_updates)
        set_weights(poisoned_model_protected, protected_weights)

        poisoned_protected_eval = evaluate_global_model(
            model=poisoned_model_protected,
            X_test=X_test,
            y_test=y_test,
        )
        print(f" -> Result WITH Defense: Accuracy = {poisoned_protected_eval['accuracy']*100:.2f}%, F1 = {poisoned_protected_eval['f1_score']:.4f}")

        security_comparison = {
            "honest": honest_eval,
            "poisoned_unprotected": poisoned_unprotected_eval,
            "poisoned_protected": poisoned_protected_eval,
            "attacker_client_id": "client_3",
            "detected_l2_norm": round(val_w3.l2_norm, 3),
            "threshold": validation_l2_threshold,
        }

    # 7. Persistence and Visualization
    print("\n[Step 7] Persisting Results & Plots...")
    full_results = {
        "dataset_info": {
            "is_synthetic": is_synthetic,
            "total_samples": len(df),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "num_features": len(feature_names),
            "feature_names": feature_names,
        },
        "config": {
            "num_clients": num_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "validation_l2_threshold": validation_l2_threshold,
            "random_seed": seed,
        },
        "centralized_baseline": baseline_metrics,
        "federated_learning": {
            "final_metrics": final_metrics,
            "total_execution_time_s": round(total_fl_time, 2),
            "round_history": history,
        },
        "blockchain_audit": {
            "mode": bc_client.mode,
            "transaction_count": len(bc_client.get_audit_trail()),
            "transactions": bc_client.get_audit_trail(),
        },
        "security_experiment": security_comparison,
    }

    save_metrics_to_json(full_results, out_path / "experiment_results.json")
    save_metrics_to_json(final_metrics, out_path / "fl_final_metrics.json")

    # Generate plots
    plot_experiment_results(
        history=history,
        baseline_metrics=baseline_metrics,
        security_results=security_comparison,
        output_dir=output_dir,
    )
    print(f"[Step 7] All plots and JSON metrics saved to '{output_dir}/'")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run Blockchain-Enabled Secure FL for IIoT")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path (CSV)")
    parser.add_argument("--clients", type=int, default=3, help="Number of simulated clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of communication rounds")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.005, help="Client learning rate")
    parser.add_argument("--threshold", type=float, default=15.0, help="L2 deviation threshold")
    parser.add_argument("--no-attack", action="store_true", help="Skip security poisoning experiment")
    parser.add_argument("--outdir", type=str, default="results", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_federated_pipeline(
        dataset_path=args.dataset,
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.epochs,
        lr=args.lr,
        validation_l2_threshold=args.threshold,
        simulate_poisoning_experiment=not args.no_attack,
        output_dir=args.outdir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
