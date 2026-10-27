"""Simulated Industrial Client for Federated Learning.

Represents an autonomous industrial site (e.g., manufacturing cell, power substation,
smart pipeline gateway) that trains an intrusion detection model locally on private telemetry.
Raw training data never leaves the client.
"""

from __future__ import annotations

import collections
import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import (
    IIoTIntrusionMLP,
    get_weights,
    hash_model_weights,
    set_weights,
)


class IndustrialClient:
    """Simulated IIoT participant in the Federated Learning network."""

    def __init__(
        self,
        client_id: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        input_dim: int = 15,
        device: str = "cpu",
    ):
        """Initializes client with private local dataset.

        Args:
            client_id: Unique identifier for the client (e.g. 'client_1').
            X_local: Local training features (kept strictly private).
            y_local: Local training labels (kept strictly private).
            input_dim: Feature dimension.
            device: Training device ('cpu').
        """
        self.client_id = client_id
        self.X_local = torch.tensor(X_local, dtype=torch.float32)
        self.y_local = torch.tensor(y_local, dtype=torch.long)
        self.input_dim = input_dim
        self.device = torch.device(device)

        # Local neural network replica
        self.model = IIoTIntrusionMLP(input_dim=input_dim).to(self.device)

        # Dataset statistics
        self.sample_count = len(self.X_local)
        self.benign_count = int((self.y_local == 0).sum().item())
        self.malicious_count = int((self.y_local == 1).sum().item())

    def set_parameters(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Synchronizes local model weights with current global model broadcast."""
        set_weights(self.model, global_weights)

    def train(
        self,
        epochs: int = 3,
        lr: float = 0.005,
        batch_size: int = 32,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        """Trains the local model on private industrial traffic.

        Args:
            epochs: Number of local training passes.
            lr: Learning rate for local Adam optimizer.
            batch_size: Mini-batch size.

        Returns:
            Tuple of (updated_weights, update_sha256_hash, sample_count).
        """
        self.model.train()
        dataset = TensorDataset(self.X_local, self.y_local)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        updated_weights = get_weights(self.model)
        update_hash = hash_model_weights(updated_weights)

        return updated_weights, update_hash, self.sample_count

    def craft_poisoned_update(
        self,
        attack_type: str = "noise_injection",
        scale: float = 15.0,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        """Simulates an adversarial / compromised IIoT node crafting a malicious update.

        Used in security experiments to test whether the update validation mechanism
        detects and rejects abnormal updates before FedAvg aggregation.

        Args:
            attack_type: 'noise_injection', 'sign_flip', or 'weight_explosion'.
            scale: Multiplier / intensity of the malicious perturbation.

        Returns:
            Tuple of (poisoned_weights, poisoned_sha256_hash, sample_count).
        """
        honest_weights = get_weights(self.model)
        poisoned_weights = collections.OrderedDict()

        for k, v in honest_weights.items():
            if v.dtype.is_floating_point:
                if attack_type == "noise_injection":
                    noise = torch.randn_like(v) * scale
                    poisoned_weights[k] = v + noise
                elif attack_type == "sign_flip":
                    poisoned_weights[k] = -v * scale
                elif attack_type == "weight_explosion":
                    poisoned_weights[k] = v * (1.0 + scale)
                else:
                    poisoned_weights[k] = v * scale
            else:
                poisoned_weights[k] = v.clone()

        poison_hash = hash_model_weights(poisoned_weights)
        return poisoned_weights, poison_hash, self.sample_count
