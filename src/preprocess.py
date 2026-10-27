"""Data loading, cleaning, preprocessing, and client partitioning for IIoT intrusion detection.

Ensures strict separation of train/test sets to prevent data leakage (scaler fit on train only).
Supports real datasets (Edge-IIoTset, CIC-IDS2017) and provides a realistic synthetic fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_iiot_dataset(
    n_samples: int = 3000,
    malicious_ratio: float = 0.35,
    random_state: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generates a realistic synthetic IIoT network telemetry dataset.

    Simulates industrial telemetry (e.g., Modbus/TCP, MQTT, OPC-UA sensors)
    with benign operational traffic and industrial cyber-attacks (DDoS, SYN Flooding,
    Port Scanning, and Command Injection).

    Args:
        n_samples: Total number of network flow records to generate.
        malicious_ratio: Proportion of malicious network flows.
        random_state: Random seed for reproducibility.
        output_path: Optional file path to persist the generated CSV.

    Returns:
        pd.DataFrame containing 15 numeric network flow features and a binary 'label' column.
    """
    rng = np.random.default_rng(random_state)
    n_malicious = int(n_samples * malicious_ratio)
    n_benign = n_samples - n_malicious

    # --- Benign IIoT Telemetry ---
    # Normal SCADA / PLC poll-response patterns: steady durations, moderate packet sizes, low flags
    benign_duration = rng.gamma(shape=5.0, scale=20.0, size=n_benign)  # ms
    benign_fwd_pkts = rng.integers(low=2, high=25, size=n_benign)
    benign_bwd_pkts = rng.integers(low=2, high=25, size=n_benign)
    benign_fwd_bytes = benign_fwd_pkts * rng.normal(loc=120.0, scale=20.0, size=n_benign).clip(40, 500)
    benign_bwd_bytes = benign_bwd_pkts * rng.normal(loc=250.0, scale=40.0, size=n_benign).clip(40, 1000)
    benign_pkt_len_mean = (benign_fwd_bytes + benign_bwd_bytes) / (benign_fwd_pkts + benign_bwd_pkts)
    benign_pkt_len_std = rng.uniform(5.0, 30.0, size=n_benign)
    benign_flow_pkts_sec = (benign_fwd_pkts + benign_bwd_pkts) / (benign_duration / 1000.0 + 1e-4)
    benign_flow_bytes_sec = (benign_fwd_bytes + benign_bwd_bytes) / (benign_duration / 1000.0 + 1e-4)
    benign_syn = rng.choice([0, 1], p=[0.95, 0.05], size=n_benign)
    benign_ack = rng.choice([1, 2], p=[0.8, 0.2], size=n_benign)
    benign_rst = rng.choice([0, 1], p=[0.99, 0.01], size=n_benign)
    benign_entropy = rng.normal(loc=2.2, scale=0.3, size=n_benign).clip(0.5, 4.0)
    benign_cpu = rng.uniform(15.0, 45.0, size=n_benign)
    benign_modbus_fc = rng.choice([3, 4, 16], p=[0.6, 0.3, 0.1], size=n_benign)  # Read Holding, Read Input, Write
    benign_labels = np.zeros(n_benign, dtype=int)

    # --- Malicious Cyber-Attacks ---
    # Anomalous high packet rates, SYN floods, port scan entropy, high CPU spikes
    mal_duration = rng.exponential(scale=15.0, size=n_malicious)  # rapid bursts
    mal_fwd_pkts = rng.integers(low=50, high=800, size=n_malicious)
    mal_bwd_pkts = rng.integers(low=0, high=10, size=n_malicious)
    mal_fwd_bytes = mal_fwd_pkts * rng.normal(loc=64.0, scale=10.0, size=n_malicious).clip(20, 200)
    mal_bwd_bytes = mal_bwd_pkts * rng.normal(loc=40.0, scale=10.0, size=n_malicious).clip(0, 100)
    mal_pkt_len_mean = (mal_fwd_bytes + mal_bwd_bytes) / (mal_fwd_pkts + mal_bwd_pkts + 1e-5)
    mal_pkt_len_std = rng.uniform(0.5, 10.0, size=n_malicious)  # uniform small attack packets
    mal_flow_pkts_sec = (mal_fwd_pkts + mal_bwd_pkts) / (mal_duration / 1000.0 + 1e-4)
    mal_flow_bytes_sec = (mal_fwd_bytes + mal_bwd_bytes) / (mal_duration / 1000.0 + 1e-4)
    mal_syn = rng.choice([1, 5, 20], p=[0.2, 0.5, 0.3], size=n_malicious)
    mal_ack = rng.choice([0, 1], p=[0.7, 0.3], size=n_malicious)
    mal_rst = rng.choice([0, 3, 10], p=[0.3, 0.4, 0.3], size=n_malicious)
    mal_entropy = rng.normal(loc=5.8, scale=0.6, size=n_malicious).clip(3.5, 8.0)  # High entropy scanning
    mal_cpu = rng.uniform(70.0, 99.5, size=n_malicious)  # Device under stress
    mal_modbus_fc = rng.choice([0, 8, 99], p=[0.4, 0.3, 0.3], size=n_malicious)  # Illegal / Diagnostic FCs
    mal_labels = np.ones(n_malicious, dtype=int)

    df_benign = pd.DataFrame({
        "flow_duration_ms": benign_duration,
        "fwd_packets": benign_fwd_pkts,
        "bwd_packets": benign_bwd_pkts,
        "fwd_bytes": benign_fwd_bytes,
        "bwd_bytes": benign_bwd_bytes,
        "pkt_len_mean": benign_pkt_len_mean,
        "pkt_len_std": benign_pkt_len_std,
        "flow_pkts_per_sec": benign_flow_pkts_sec,
        "flow_bytes_per_sec": benign_flow_bytes_sec,
        "syn_flag_count": benign_syn,
        "ack_flag_count": benign_ack,
        "rst_flag_count": benign_rst,
        "port_entropy": benign_entropy,
        "cpu_utilization_pct": benign_cpu,
        "modbus_function_code": benign_modbus_fc,
        "label": benign_labels,
    })

    df_malicious = pd.DataFrame({
        "flow_duration_ms": mal_duration,
        "fwd_packets": mal_fwd_pkts,
        "bwd_packets": mal_bwd_pkts,
        "fwd_bytes": mal_fwd_bytes,
        "bwd_bytes": mal_bwd_bytes,
        "pkt_len_mean": mal_pkt_len_mean,
        "pkt_len_std": mal_pkt_len_std,
        "flow_pkts_per_sec": mal_flow_pkts_sec,
        "flow_bytes_per_sec": mal_flow_bytes_sec,
        "syn_flag_count": mal_syn,
        "ack_flag_count": mal_ack,
        "rst_flag_count": mal_rst,
        "port_entropy": mal_entropy,
        "cpu_utilization_pct": mal_cpu,
        "modbus_function_code": mal_modbus_fc,
        "label": mal_labels,
    })

    df = pd.concat([df_benign, df_malicious], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    fallback_synthetic: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, bool]:
    """Loads an intrusion detection dataset from disk or falls back to synthetic generation.

    Args:
        dataset_path: Path to dataset file.
        fallback_synthetic: If True, generate synthetic data when file is absent.
        random_state: Seed for synthetic generation.

    Returns:
        Tuple of (DataFrame, is_synthetic_flag).
    """
    candidate_paths = []
    if dataset_path:
        candidate_paths.append(Path(dataset_path))
    candidate_paths.extend([
        Path("data/edge_iiotset.csv"),
        Path("data/cic_ids2017.csv"),
        Path("data/synthetic_iiot_traffic.csv"),
    ])

    for p in candidate_paths:
        if p.exists() and p.is_file():
            df = pd.read_csv(p)
            is_synthetic = "synthetic" in p.name.lower()
            return df, is_synthetic

    if fallback_synthetic:
        synth_path = Path("data/synthetic_iiot_traffic.csv")
        df = generate_synthetic_iiot_dataset(output_path=synth_path, random_state=random_state)
        return df, True

    raise FileNotFoundError(
        f"No dataset found at {dataset_path} and synthetic fallback is disabled."
    )


def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleans raw dataset, standardizes column names, encodes binary labels.

    Target column will be mapped to 0 (Normal) and 1 (Attack).
    Inf and NaN values are properly scrubbed.
    """
    df = df.copy()

    # Identify label column
    label_col = None
    for candidate in ["label", "Label", "Attack_type", "attack_type", "class", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # Default to last column
        label_col = df.columns[-1]

    # Convert label to binary: 0 for normal/benign, 1 for intrusion
    raw_labels = df[label_col].astype(str).str.strip()
    normal_patterns = ["0", "normal", "benign", "0.0", "normal traffic"]
    binary_labels = raw_labels.apply(
        lambda x: 0 if x.lower() in normal_patterns else 1
    ).astype(np.int64)

    # Feature columns
    feature_df = df.drop(columns=[label_col])

    # Convert non-numeric columns if any
    for col in feature_df.columns:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            # Attempt numeric conversion, fallback to categorical factorization
            converted = pd.to_numeric(feature_df[col], errors="coerce")
            if converted.notna().sum() > len(feature_df) * 0.5:
                feature_df[col] = converted
            else:
                feature_df[col] = pd.factorize(feature_df[col])[0]

    # Replace inf with nan
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs with column median (fit on full dataset here, or cleaned)
    feature_df = feature_df.fillna(feature_df.median())

    # If any column is entirely constant, drop or retain with small noise
    for col in feature_df.columns:
        if feature_df[col].std() == 0 or np.isnan(feature_df[col].std()):
            feature_df[col] = feature_df[col].fillna(0.0)

    return feature_df, binary_labels


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Splits dataset into train/test and scales features with zero leakage.

    StandardScaler is fitted strictly on X_train, then transforms X_train and X_test.

    Returns:
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    feature_df, labels = clean_and_prepare_data(df)
    feature_names = list(feature_df.columns)

    X = feature_df.values.astype(np.float32)
    y = labels.values.astype(np.int64)

    # Stratified split to preserve class ratios
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler ONLY on training data to prevent data leakage!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def partition_for_clients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = 3,
    non_iid: bool = False,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Divides the training set among simulated industrial clients.

    Args:
        X_train: Preprocessed scaled training features.
        y_train: Training labels (0 or 1).
        num_clients: Number of simulated clients (default 3).
        non_iid: If True, partitions with heterogeneous class distributions.
        random_state: Reproducibility seed.

    Returns:
        List of (X_client_i, y_client_i) tuples.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X_train)

    if not non_iid:
        # Uniform IID partition
        indices = rng.permutation(n_samples)
        client_chunks = np.array_split(indices, num_clients)
        client_data = [
            (X_train[idx], y_train[idx])
            for idx in client_chunks
        ]
    else:
        # Non-IID partition (simulate industrial segment specialization)
        # Client 0: Benign dominant (e.g. well-isolated SCADA plant)
        # Client 1: Attack heavy (e.g. exposed perimeter gateway)
        # Client 2: Mixed telemetry
        benign_indices = np.where(y_train == 0)[0]
        mal_indices = np.where(y_train == 1)[0]
        rng.shuffle(benign_indices)
        rng.shuffle(mal_indices)

        # Distribute non-identically
        b_splits = np.array_split(benign_indices, [int(0.5 * len(benign_indices)), int(0.75 * len(benign_indices))])
        m_splits = np.array_split(mal_indices, [int(0.15 * len(mal_indices)), int(0.70 * len(mal_indices))])

        client_data = []
        for i in range(num_clients):
            c_idx = np.concatenate([b_splits[i], m_splits[i]])
            rng.shuffle(c_idx)
            client_data.append((X_train[c_idx], y_train[c_idx]))

    return client_data
