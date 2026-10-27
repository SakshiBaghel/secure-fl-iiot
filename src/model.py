"""PyTorch Neural Network architecture and deterministic serialization for IIoT intrusion detection.

Provides the local model architecture (IIoTIntrusionMLP) and utilities for extracting,
loading, serializing, and cryptographically hashing model weight updates.
"""

from __future__ import annotations

import collections
import hashlib
import io
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn


class IIoTIntrusionMLP(nn.Module):
    """Multilayer Perceptron (MLP) for Industrial IoT network intrusion detection.

    Architecture:
        Input Layer (num_features)
        -> Linear(num_features, 64) -> BatchNorm1d(64) -> ReLU() -> Dropout(0.2)
        -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU() -> Dropout(0.1)
        -> Linear(32, 2)  [Logits: Class 0 = Normal, Class 1 = Malicious Attack]
    """

    def __init__(self, input_dim: int = 15, hidden_dim1: int = 64, hidden_dim2: int = 32, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.net(x)


def get_weights(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Extracts a detached, CPU-bound copy of model parameters."""
    return collections.OrderedDict({
        k: v.detach().clone().cpu()
        for k, v in model.state_dict().items()
    })


def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Loads a parameter dictionary into the model."""
    state_dict = {
        k: v.clone().to(next(model.parameters()).device)
        if isinstance(v, torch.Tensor) else torch.tensor(v)
        for k, v in weights.items()
    }
    model.load_state_dict(state_dict)


def serialize_weights_deterministically(weights: Dict[str, torch.Tensor]) -> bytes:
    """Serializes model weights deterministically for cryptographic hashing.

    Procedure:
    1. Sort parameter keys lexicographically to guarantee consistent key ordering.
    2. Convert each tensor to continuous float32 numpy arrays on CPU.
    3. Pack key names and raw array bytes into an ordered binary stream.
    4. Eliminates OS, timestamp, and runtime variability from the serialization.

    Returns:
        Deterministic bytes representation of the model state.
    """
    buffer = io.BytesIO()
    for key in sorted(weights.keys()):
        key_bytes = key.encode("utf-8")
        buffer.write(len(key_bytes).to_bytes(4, byteorder="big"))
        buffer.write(key_bytes)
        tensor_data = weights[key].detach().cpu().numpy().astype(np.float32)
        # Write array shape
        buffer.write(len(tensor_data.shape).to_bytes(4, byteorder="big"))
        for dim in tensor_data.shape:
            buffer.write(int(dim).to_bytes(4, byteorder="big"))
        # Write raw bytes
        buffer.write(tensor_data.tobytes())

    return buffer.getvalue()


def hash_model_weights(weights: Dict[str, torch.Tensor]) -> str:
    """Computes a deterministic SHA-256 cryptographic hash of model parameters.

    Important Security Context:
    The resulting hash allows participants and aggregators to verify the integrity
    of submitted updates against on-chain records. As documented in project requirements,
    a cryptographic hash verifies integrity against tampering, but does NOT by itself
    prove that an update is benign or free from poisoning attacks.

    Returns:
        Hex-encoded SHA-256 string (64 characters).
    """
    serialized = serialize_weights_deterministically(weights)
    return hashlib.sha256(serialized).hexdigest()


def compute_l2_norm(weights: Dict[str, torch.Tensor]) -> float:
    """Calculates the global L2 (Euclidean) norm across all model parameters."""
    total_sq = 0.0
    for tensor in weights.values():
        if tensor.dtype.is_floating_point:
            total_sq += float(torch.sum(tensor ** 2).item())
    return float(np.sqrt(total_sq))


def compute_weight_difference(
    w_new: Dict[str, torch.Tensor],
    w_base: Dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Calculates parameter update delta: Delta_W = W_new - W_base."""
    diff = collections.OrderedDict()
    for k in w_base.keys():
        diff[k] = w_new[k] - w_base[k]
    return diff
