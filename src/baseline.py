"""Centralized Machine Learning baseline for IIoT intrusion detection.

Trains scikit-learn classifiers on the aggregated centralized dataset and evaluates
on the held-out test set to establish an empirical benchmark against Federated Learning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import compute_intrusion_metrics, print_metrics_summary, save_metrics_to_json
from src.preprocess import load_dataset, preprocess_and_split


def train_and_evaluate_centralized(
    dataset_path: Optional[str] = None,
    model_type: str = "random_forest",
    random_state: int = 42,
    output_dir: str = "results",
) -> Tuple[Dict[str, Any], Any]:
    """Trains a centralized baseline classifier and evaluates it on held-out test data.

    Args:
        dataset_path: Path to dataset CSV (or None for auto-detect/synthetic fallback).
        model_type: 'random_forest' or 'logistic_regression'.
        random_state: Random seed for reproducibility.
        output_dir: Directory where results JSON will be saved.

    Returns:
        Tuple of (metrics_dict, trained_model).
    """
    # 1. Load dataset
    df, is_synthetic = load_dataset(dataset_path=dataset_path, random_state=random_state)
    print(f"[Centralized Baseline] Loaded dataset with {len(df)} samples. (Synthetic: {is_synthetic})")

    # 2. Split and scale (zero data leakage)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
        df, test_size=0.20, random_state=random_state
    )
    print(f"[Centralized Baseline] Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Initialize and train model
    if model_type.lower() == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)
        model_name = "RandomForest (Centralized Baseline)"
    elif model_type.lower() == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        model_name = "LogisticRegression (Centralized Baseline)"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"[Centralized Baseline] Fitting {model_name}...")
    clf.fit(X_train, y_train)

    # 4. Predict on held-out test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # 5. Compute metrics
    metrics = compute_intrusion_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    metrics["model_type"] = model_name
    metrics["is_synthetic"] = is_synthetic
    metrics["num_features"] = len(feature_names)
    metrics["train_samples"] = int(X_train.shape[0])
    metrics["test_samples"] = int(X_test.shape[0])

    print_metrics_summary(f"Evaluation: {model_name}", metrics)

    # 6. Save results
    out_file = Path(output_dir) / "centralized_baseline.json"
    save_metrics_to_json(metrics, out_file)
    print(f"[Centralized Baseline] Metrics successfully saved to {out_file}")

    return metrics, clf


def main():
    parser = argparse.ArgumentParser(description="Centralized ML baseline for IIoT Intrusion Detection")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom CSV dataset")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save metrics")
    args = parser.parse_args()

    train_and_evaluate_centralized(
        dataset_path=args.dataset,
        model_type=args.model,
        random_state=args.seed,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
