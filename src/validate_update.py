"""Security validation of client model updates for Federated Learning in IIoT.

Detects and filters out abnormal, poisoned, or corrupted model updates before
they can contaminate the FedAvg aggregation.

Limitations:
This heuristic validation filters large-magnitude gradient anomalies and model poisoning attacks.
It does not guarantee detection of stealthy, small-norm backdoor attacks or subtle adversarial
perturbations without advanced statistical clustering (e.g. Krum, Multi-Krum, or Bulyan).
"""

from __future__ import annotations

import enum
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch

from src.model import compute_l2_norm, compute_weight_difference


class ValidationReason(enum.IntEnum):
    """Reason codes recorded for model update validation."""
    ACCEPTED = 0
    EXCEEDED_L2_NORM = 1
    INVALID_NUMERICAL_VALUES = 2
    UNAUTHORIZED_CLIENT = 3
    HASH_INTEGRITY_MISMATCH = 4


class ValidationResult(NamedTuple):
    """Result of update validation."""
    is_valid: bool
    reason_code: ValidationReason
    reason_message: str
    l2_norm: float
    client_id: str


class UpdateValidator:
    """Configurable model update validator based on L2 deviation thresholding."""

    def __init__(
        self,
        max_l2_threshold: float = 15.0,
        authorized_clients: Optional[set] = None,
    ):
        """Initializes validator.

        Args:
            max_l2_threshold: Maximum allowable L2 norm of the parameter delta.
            authorized_clients: Set of authorized client IDs. If None, authorization check is skipped.
        """
        self.max_l2_threshold = max_l2_threshold
        self.authorized_clients = authorized_clients or {"client_1", "client_2", "client_3"}

    def validate_update(
        self,
        client_id: str,
        client_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
    ) -> ValidationResult:
        """Validates a client's submitted model weights against security constraints.

        Args:
            client_id: Identifier of the submitting client.
            client_weights: The candidate model weights submitted by client.
            global_weights: The reference global model weights before local training.
            expected_hash: On-chain or advertised SHA-256 hash (if verifying integrity).
            actual_hash: Calculated SHA-256 hash of submitted weights.

        Returns:
            ValidationResult containing decision, reason code, message, and computed L2 norm.
        """
        # 1. Authorization check
        if self.authorized_clients is not None and client_id not in self.authorized_clients:
            return ValidationResult(
                is_valid=False,
                reason_code=ValidationReason.UNAUTHORIZED_CLIENT,
                reason_message=f"Client '{client_id}' is not in authorized participant list.",
                l2_norm=0.0,
                client_id=client_id,
            )

        # 2. Hash integrity check (if both hashes provided)
        if expected_hash is not None and actual_hash is not None:
            if expected_hash.lower() != actual_hash.lower():
                return ValidationResult(
                    is_valid=False,
                    reason_code=ValidationReason.HASH_INTEGRITY_MISMATCH,
                    reason_message=f"Hash mismatch: expected {expected_hash}, calculated {actual_hash}",
                    l2_norm=0.0,
                    client_id=client_id,
                )

        # 3. Numerical integrity check (detect NaNs or Infs)
        for key, tensor in client_weights.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return ValidationResult(
                    is_valid=False,
                    reason_code=ValidationReason.INVALID_NUMERICAL_VALUES,
                    reason_message=f"Tensor '{key}' contains NaN or Inf values.",
                    l2_norm=float("inf"),
                    client_id=client_id,
                )

        # 4. Parameter deviation / L2 norm check
        delta = compute_weight_difference(client_weights, global_weights)
        l2_deviation = compute_l2_norm(delta)

        if l2_deviation > self.max_l2_threshold:
            return ValidationResult(
                is_valid=False,
                reason_code=ValidationReason.EXCEEDED_L2_NORM,
                reason_message=(
                    f"Update L2 deviation ({l2_deviation:.3f}) exceeded threshold ({self.max_l2_threshold:.3f}). "
                    "Suspected model poisoning or abnormal gradient surge."
                ),
                l2_norm=l2_deviation,
                client_id=client_id,
            )

        # Passed all validation checks
        return ValidationResult(
            is_valid=True,
            reason_code=ValidationReason.ACCEPTED,
            reason_message="Update passed validation checks.",
            l2_norm=l2_deviation,
            client_id=client_id,
        )
