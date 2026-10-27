"""Unit tests for dataset loading, preprocessing, zero-data-leakage splitting, and client partitioning."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    clean_and_prepare_data,
    generate_synthetic_iiot_dataset,
    load_dataset,
    partition_for_clients,
    preprocess_and_split,
)


def test_generate_synthetic_iiot_dataset():
    df = generate_synthetic_iiot_dataset(n_samples=500, malicious_ratio=0.3, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})
    assert df["label"].sum() == 150  # 30% of 500


def test_load_dataset_fallback():
    df, is_synth = load_dataset(dataset_path="non_existent_file.csv", fallback_synthetic=True, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert is_synth is True
    assert len(df) > 0


def test_clean_and_prepare_data_handles_nans_and_infs():
    raw_data = pd.DataFrame({
        "flow_duration": [10.0, np.nan, 30.0, np.inf],
        "pkt_count": [1, 2, -np.inf, 4],
        "category": ["A", "B", "A", "B"],
        "label": ["Normal", "Attack", "Normal", "DDoS"],
    })
    feature_df, labels = clean_and_prepare_data(raw_data)
    assert len(feature_df) == 4
    assert not feature_df.isna().any().any()
    assert not np.isinf(feature_df.values).any()
    assert list(labels.values) == [0, 1, 0, 1]


def test_preprocess_and_split_zero_leakage():
    df = generate_synthetic_iiot_dataset(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
        df, test_size=0.20, random_state=42
    )

    # 1. Check dimensional consistency
    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1] == len(feature_names)
    assert len(y_train) == 800
    assert len(y_test) == 200

    # 2. Strict Zero Data Leakage verification:
    # Scaler was fit ONLY on train, so training features have mean approx 0 and std approx 1
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-3)

    # Test set transformed with train parameters, so its mean will NOT be identically zero
    assert not np.allclose(X_test.mean(axis=0), 0.0, atol=1e-6)


def test_partition_for_clients_iid():
    X_train = np.random.randn(900, 10).astype(np.float32)
    y_train = np.random.choice([0, 1], size=900)

    partitions = partition_for_clients(X_train, y_train, num_clients=3, non_iid=False, random_state=42)
    assert len(partitions) == 3

    total_samples = 0
    for x_part, y_part in partitions:
        assert len(x_part) == len(y_part)
        assert len(x_part) == 300
        total_samples += len(x_part)

    assert total_samples == 900


def test_partition_for_clients_non_iid():
    X_train = np.random.randn(600, 10).astype(np.float32)
    y_train = np.array([0] * 400 + [1] * 200)

    partitions = partition_for_clients(X_train, y_train, num_clients=3, non_iid=True, random_state=42)
    assert len(partitions) == 3
    total_samples = sum(len(x_part) for x_part, _ in partitions)
    assert total_samples == 600
