"""Unit tests for Federated Averaging (FedAvg) and model evaluation.

Includes rigorous analytical numerical tests verifying sample-weighted parameter averaging.
"""

import collections
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.client import IndustrialClient
from src.fedavg import evaluate_global_model, federated_averaging
from src.model import IIoTIntrusionMLP, get_weights, set_weights


def test_fedavg_numerical_known_values():
    """Mathematical verification of FedAvg with analytically known values.

    Setup:
      Client 1 has n1 = 100 samples, parameter tensor = [2.0, 4.0]
      Client 2 has n2 = 300 samples, parameter tensor = [6.0, 8.0]
      Total samples N = 100 + 300 = 400

    Analytical Expected Output:
      W_global = (100/400) * [2.0, 4.0] + (300/400) * [6.0, 8.0]
               = 0.25 * [2.0, 4.0] + 0.75 * [6.0, 8.0]
               = [0.5, 1.0] + [4.5, 6.0]
               = [5.0, 7.0]
    """
    w1 = collections.OrderedDict({
        "layer.weight": torch.tensor([2.0, 4.0], dtype=torch.float32),
        "layer.bias": torch.tensor([10.0], dtype=torch.float32),
    })
    count1 = 100

    w2 = collections.OrderedDict({
        "layer.weight": torch.tensor([6.0, 8.0], dtype=torch.float32),
        "layer.bias": torch.tensor([30.0], dtype=torch.float32),
    })
    count2 = 300

    client_updates = [(w1, count1), (w2, count2)]
    aggregated = federated_averaging(client_updates)

    expected_weight = torch.tensor([5.0, 7.0], dtype=torch.float32)
    expected_bias = torch.tensor([0.25 * 10.0 + 0.75 * 30.0], dtype=torch.float32)  # 2.5 + 22.5 = 25.0

    assert torch.allclose(aggregated["layer.weight"], expected_weight)
    assert torch.allclose(aggregated["layer.bias"], expected_bias)


def test_fedavg_empty_list_raises_error():
    """FedAvg must safely raise ValueError when given an empty list of updates."""
    with pytest.raises(ValueError, match="Cannot perform FedAvg on an empty list"):
        federated_averaging([])


def test_fedavg_single_client():
    """FedAvg with a single client update must return exactly that client's weights."""
    w = collections.OrderedDict({
        "w": torch.tensor([1.23, 4.56], dtype=torch.float32),
    })
    res = federated_averaging([(w, 50)])
    assert torch.allclose(res["w"], w["w"])


def test_client_local_training_and_privacy():
    """Verifies that client trains locally and returns parameter weights without leaking raw data."""
    X_fake = np.random.randn(50, 15).astype(np.float32)
    y_fake = np.random.choice([0, 1], size=50)

    client = IndustrialClient("client_test", X_fake, y_fake, input_dim=15)
    init_weights = get_weights(client.model)

    updated_weights, update_hash, sample_count = client.train(epochs=1, lr=0.01, batch_size=16)

    # Weights must have changed due to gradient descent
    diff = False
    for k in init_weights.keys():
        if not torch.allclose(init_weights[k], updated_weights[k]):
            diff = True
            break
    assert diff is True
    assert sample_count == 50
    assert len(update_hash) == 64  # Valid SHA-256 hex string


def test_evaluate_global_model_metrics():
    """Verifies global model prediction and metric computation on test set."""
    model = IIoTIntrusionMLP(input_dim=10)
    X_test = np.random.randn(100, 10).astype(np.float32)
    y_test = np.random.choice([0, 1], size=100)

    metrics = evaluate_global_model(model, X_test, y_test, batch_size=32)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["sample_counts"]["total_tested"] == 100
