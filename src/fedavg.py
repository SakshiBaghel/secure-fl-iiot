"""Federated Averaging (FedAvg) aggregation and global model evaluation.

Aggregates model parameter updates from authorized, validated industrial clients
weighted by their local training dataset size:
    W_global = sum_{k in Accepted} (n_k / N_total) * W_k
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import compute_intrusion_metrics
from src.model import get_weights, set_weights


def federated_averaging(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> OrderedDict[str, torch.Tensor]:
    """Computes sample-weighted Federated Averaging across accepted client updates.

    Args:
        client_updates: List of (weights_dict, sample_count) tuples for accepted clients.

    Returns:
        Aggregated global model weights OrderedDict.

    Raises:
        ValueError: If client_updates is empty or sample counts are invalid.
    """
    if not client_updates:
        raise ValueError("Cannot perform FedAvg on an empty list of client updates.")

    total_samples = sum(count for _, count in client_updates)
    if total_samples <= 0:
        raise ValueError(f"Total sample count across clients must be > 0, got {total_samples}")

    # Inspect first update structure
    first_weights, _ = client_updates[0]
    aggregated_weights = collections.OrderedDict()

    for key, template_tensor in first_weights.items():
        if template_tensor.dtype.is_floating_point:
            # Weighted average for floating-point weights and biases
            weighted_sum = torch.zeros_like(template_tensor, dtype=torch.float32)
            for weights, count in client_updates:
                weight_factor = float(count) / float(total_samples)
                tensor_val = weights[key].detach().cpu().to(torch.float32)
                weighted_sum += tensor_val * weight_factor
            aggregated_weights[key] = weighted_sum
        else:
            # Non-floating point tensors (e.g., BatchNorm num_batches_tracked int64)
            # Take mode or maximum across clients
            aggregated_weights[key] = template_tensor.detach().cpu().clone()

    return aggregated_weights


def evaluate_global_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates the global PyTorch model on the held-out test set.

    Args:
        model: Global neural network.
        X_test: Held-out test features (scaled).
        y_test: Held-out test labels.
        batch_size: Batch size for evaluation.
        device: Device ('cpu').

    Returns:
        Evaluation metrics dictionary.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.long)

    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch_x = X_t[i : i + batch_size].to(dev)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    return compute_intrusion_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
