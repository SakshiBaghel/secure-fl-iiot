"""Unit tests for the model update security validation module."""

import collections
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.model import hash_model_weights
from src.validate_update import UpdateValidator, ValidationReason


@pytest.fixture
def base_weights():
    return collections.OrderedDict({
        "layer.weight": torch.zeros((5, 5), dtype=torch.float32),
        "layer.bias": torch.zeros((5,), dtype=torch.float32),
    })


def test_validator_accepts_honest_update(base_weights):
    validator = UpdateValidator(max_l2_threshold=10.0, authorized_clients={"client_1"})

    # Small honest gradient update
    honest_update = collections.OrderedDict({
        "layer.weight": torch.ones((5, 5), dtype=torch.float32) * 0.1,  # norm = sqrt(25 * 0.01) = 0.5
        "layer.bias": torch.zeros((5,), dtype=torch.float32),
    })

    result = validator.validate_update(
        client_id="client_1",
        client_weights=honest_update,
        global_weights=base_weights,
    )

    assert result.is_valid is True
    assert result.reason_code == ValidationReason.ACCEPTED
    assert result.l2_norm < 10.0


def test_validator_rejects_exceeded_l2_norm(base_weights):
    validator = UpdateValidator(max_l2_threshold=5.0, authorized_clients={"client_1"})

    # Malicious poisoned update with huge magnitude
    poisoned_update = collections.OrderedDict({
        "layer.weight": torch.ones((5, 5), dtype=torch.float32) * 10.0,  # norm = sqrt(25 * 100) = 50.0
        "layer.bias": torch.zeros((5,), dtype=torch.float32),
    })

    result = validator.validate_update(
        client_id="client_1",
        client_weights=poisoned_update,
        global_weights=base_weights,
    )

    assert result.is_valid is False
    assert result.reason_code == ValidationReason.EXCEEDED_L2_NORM
    assert "exceeded threshold" in result.reason_message.lower()


def test_validator_rejects_nan_or_inf(base_weights):
    validator = UpdateValidator(max_l2_threshold=100.0, authorized_clients={"client_1"})

    corrupted_update = collections.OrderedDict({
        "layer.weight": torch.tensor([[float("nan"), 1.0], [0.0, 0.0]], dtype=torch.float32),
        "layer.bias": torch.zeros((2,), dtype=torch.float32),
    })

    result = validator.validate_update(
        client_id="client_1",
        client_weights=corrupted_update,
        global_weights=base_weights,
    )

    assert result.is_valid is False
    assert result.reason_code == ValidationReason.INVALID_NUMERICAL_VALUES


def test_validator_rejects_unauthorized_client(base_weights):
    validator = UpdateValidator(max_l2_threshold=100.0, authorized_clients={"client_1", "client_2"})

    result = validator.validate_update(
        client_id="client_rogue_99",
        client_weights=base_weights,
        global_weights=base_weights,
    )

    assert result.is_valid is False
    assert result.reason_code == ValidationReason.UNAUTHORIZED_CLIENT


def test_validator_detects_hash_tampering(base_weights):
    validator = UpdateValidator(max_l2_threshold=100.0, authorized_clients={"client_1"})

    calculated_hash = hash_model_weights(base_weights)
    tampered_hash = "f" * 64

    result = validator.validate_update(
        client_id="client_1",
        client_weights=base_weights,
        global_weights=base_weights,
        expected_hash=tampered_hash,
        actual_hash=calculated_hash,
    )

    assert result.is_valid is False
    assert result.reason_code == ValidationReason.HASH_INTEGRITY_MISMATCH
