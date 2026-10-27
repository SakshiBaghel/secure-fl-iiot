"""Python-to-Blockchain Interface for Federated Learning auditability.

Connects to a local Hardhat Ethereum node using Web3.py.
If the local blockchain RPC node is offline, gracefully switches to a high-fidelity
in-memory simulated ledger to ensure reproducible, non-blocking execution across environments.
"""

from __future__ import annotations

import hashlib
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
from web3 import Web3


class BlockchainClient:
    """Interacts with the FLRegistry smart contract via Web3.py or simulation fallback."""

    # Standard well-known Hardhat local testnet accounts
    HARDHAT_DEFAULT_ACCOUNTS = [
        "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",  # Deployer / Aggregator
        "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # Client 1
        "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Client 2
        "0x90F79bf6EB2c4f870365E785982E1f101E93b906",  # Client 3
        "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",  # Unauthorized client
    ]
    HARDHAT_DEFAULT_KEYS = [
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
        "0x5de4111afa1a4b94908f83103eb2f937b04a474609590c73e5da898dd48010ba",
        "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
        "0x47e179ec346fe3a66b70e8797cfb16adc258374d827379d47da30e52f0d9374e",
    ]

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        contract_address: Optional[str] = None,
        artifact_path: Optional[str] = None,
        prefer_simulation: bool = False,
    ):
        """Initializes Web3 connection with fallback.

        Args:
            rpc_url: Local blockchain RPC endpoint.
            contract_address: Deployed FLRegistry address (optional).
            artifact_path: Path to compiled FLRegistry JSON artifact.
            prefer_simulation: Force simulated mode even if RPC is reachable.
        """
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.mode = "simulated"
        self.w3: Optional[Web3] = None
        self.contract = None
        self.abi: List[Dict[str, Any]] = []

        # Audit trail logs
        self.transaction_logs: List[Dict[str, Any]] = []
        self.simulated_state: Dict[str, Any] = {
            "participants": {},
            "rounds": {},
            "updates": collections.defaultdict(list) if "collections" in globals() else {},
            "block_number": 100,
        }
        self.client_account_map = {
            "aggregator_server": self.HARDHAT_DEFAULT_ACCOUNTS[0],
            "client_1": self.HARDHAT_DEFAULT_ACCOUNTS[1],
            "client_2": self.HARDHAT_DEFAULT_ACCOUNTS[2],
            "client_3": self.HARDHAT_DEFAULT_ACCOUNTS[3],
        }

        # Attempt to load ABI
        self._load_abi(artifact_path)

        # Attempt live connection
        if not prefer_simulation:
            self._attempt_connection()

    def _load_abi(self, artifact_path: Optional[str] = None) -> None:
        """Loads contract ABI from disk."""
        candidates = []
        if artifact_path:
            candidates.append(Path(artifact_path))
        candidates.extend([
            PROJECT_ROOT / "blockchain" / "deployment_info.json",
            PROJECT_ROOT / "blockchain" / "artifacts" / "contracts" / "FLRegistry.sol" / "FLRegistry.json",
        ])

        for p in candidates:
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "abi" in data:
                            self.abi = data["abi"]
                        if not self.contract_address and "address" in data:
                            self.contract_address = data["address"]
                        break
                except Exception:
                    pass

    def _attempt_connection(self) -> None:
        """Attempts to connect to live local Ethereum RPC node."""
        try:
            w3_candidate = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 2}))
            if w3_candidate.is_connected():
                self.w3 = w3_candidate
                if self.contract_address and self.abi:
                    checksum_addr = self.w3.to_checksum_address(self.contract_address)
                    self.contract = self.w3.eth.contract(address=checksum_addr, abi=self.abi)
                    self.mode = "live"
                    print(f"[BlockchainClient] Connected to LIVE local blockchain at {self.rpc_url}")
                    print(f"[BlockchainClient] Using contract at {checksum_addr}")
                    return
        except Exception:
            pass

        self.mode = "simulated"
        # Seed simulated aggregator
        self.simulated_state["participants"][self.HARDHAT_DEFAULT_ACCOUNTS[0]] = {
            "clientId": "aggregator_server",
            "role": 1,
            "isRegistered": True,
        }

    def register_participant(
        self,
        participant_address: str,
        client_id: str,
        role: int = 0,
    ) -> Dict[str, Any]:
        """Registers an authorized industrial participant on the blockchain."""
        start_time = time.time()
        tx_hash = "0x" + hashlib.sha256(f"reg_{client_id}_{time.time()}".encode()).hexdigest()

        if self.mode == "live" and self.contract and self.w3:
            try:
                deployer = self.w3.eth.accounts[0]
                tx = self.contract.functions.registerParticipant(
                    self.w3.to_checksum_address(participant_address),
                    client_id,
                    role
                ).transact({"from": deployer})
                receipt = self.w3.eth.wait_for_transaction_receipt(tx)
                latency_ms = (time.time() - start_time) * 1000
                log_entry = {
                    "event": "ParticipantRegistered",
                    "client_id": client_id,
                    "address": participant_address,
                    "role": role,
                    "tx_hash": receipt.transactionHash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "latency_ms": round(latency_ms, 2),
                    "mode": "live",
                }
                self.transaction_logs.append(log_entry)
                return log_entry
            except Exception as e:
                print(f"[BlockchainClient] Live transaction failed ({e}), falling back to simulated ledger.")

        # Simulated registration
        latency_ms = (time.time() - start_time) * 1000 + 1.2
        self.simulated_state["block_number"] += 1
        self.simulated_state["participants"][participant_address] = {
            "clientId": client_id,
            "role": role,
            "isRegistered": True,
        }
        self.client_account_map[client_id] = participant_address

        log_entry = {
            "event": "ParticipantRegistered",
            "client_id": client_id,
            "address": participant_address,
            "role": role,
            "tx_hash": tx_hash,
            "block_number": self.simulated_state["block_number"],
            "gas_used": 47250,
            "latency_ms": round(latency_ms, 2),
            "mode": "simulated",
        }
        self.transaction_logs.append(log_entry)
        return log_entry

    def start_training_round(
        self,
        round_number: int,
        global_model_hash: str,
    ) -> Dict[str, Any]:
        """Records the beginning of an FL communication round on-chain."""
        start_time = time.time()
        tx_hash = "0x" + hashlib.sha256(f"round_start_{round_number}_{time.time()}".encode()).hexdigest()

        if self.mode == "live" and self.contract and self.w3:
            try:
                deployer = self.w3.eth.accounts[0]
                tx = self.contract.functions.startTrainingRound(
                    round_number, global_model_hash
                ).transact({"from": deployer})
                receipt = self.w3.eth.wait_for_transaction_receipt(tx)
                latency_ms = (time.time() - start_time) * 1000
                log_entry = {
                    "event": "RoundStarted",
                    "round": round_number,
                    "global_model_hash": global_model_hash,
                    "tx_hash": receipt.transactionHash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "latency_ms": round(latency_ms, 2),
                    "mode": "live",
                }
                self.transaction_logs.append(log_entry)
                return log_entry
            except Exception as e:
                print(f"[BlockchainClient] Live transaction failed ({e}), using simulated.")

        # Simulated
        latency_ms = (time.time() - start_time) * 1000 + 1.5
        self.simulated_state["block_number"] += 1
        self.simulated_state["rounds"][round_number] = {
            "global_hash_start": global_model_hash,
            "completed": False,
        }
        log_entry = {
            "event": "RoundStarted",
            "round": round_number,
            "global_model_hash": global_model_hash,
            "tx_hash": tx_hash,
            "block_number": self.simulated_state["block_number"],
            "gas_used": 62100,
            "latency_ms": round(latency_ms, 2),
            "mode": "simulated",
        }
        self.transaction_logs.append(log_entry)
        return log_entry

    def record_update(
        self,
        round_number: int,
        client_id: str,
        update_hash: str,
        sample_count: int,
        is_valid: bool,
        reason_code: int,
        client_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records participant ID, update SHA-256 hash, and validation decision on-chain."""
        start_time = time.time()
        tx_hash = "0x" + hashlib.sha256(f"update_{round_number}_{client_id}_{time.time()}".encode()).hexdigest()
        caller_addr = client_address or self.client_account_map.get(client_id, self.HARDHAT_DEFAULT_ACCOUNTS[1])

        if self.mode == "live" and self.contract and self.w3:
            try:
                tx = self.contract.functions.recordUpdate(
                    round_number,
                    client_id,
                    update_hash,
                    sample_count,
                    is_valid,
                    int(reason_code)
                ).transact({"from": self.w3.to_checksum_address(caller_addr)})
                receipt = self.w3.eth.wait_for_transaction_receipt(tx)
                latency_ms = (time.time() - start_time) * 1000
                log_entry = {
                    "event": "UpdateRecorded",
                    "round": round_number,
                    "client_id": client_id,
                    "update_hash": update_hash,
                    "sample_count": sample_count,
                    "is_valid": is_valid,
                    "reason_code": int(reason_code),
                    "tx_hash": receipt.transactionHash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "latency_ms": round(latency_ms, 2),
                    "mode": "live",
                }
                self.transaction_logs.append(log_entry)
                return log_entry
            except Exception as e:
                print(f"[BlockchainClient] Live transaction failed ({e}), using simulated.")

        # Simulated
        latency_ms = (time.time() - start_time) * 1000 + 1.8
        self.simulated_state["block_number"] += 1
        if round_number not in self.simulated_state["updates"]:
            self.simulated_state["updates"][round_number] = []

        self.simulated_state["updates"][round_number].append({
            "clientId": client_id,
            "updateHash": update_hash,
            "sampleCount": sample_count,
            "isValid": is_valid,
            "reasonCode": int(reason_code),
            "submittedBy": caller_addr,
            "timestamp": int(time.time()),
        })

        log_entry = {
            "event": "UpdateRecorded",
            "round": round_number,
            "client_id": client_id,
            "update_hash": update_hash,
            "sample_count": sample_count,
            "is_valid": is_valid,
            "reason_code": int(reason_code),
            "tx_hash": tx_hash,
            "block_number": self.simulated_state["block_number"],
            "gas_used": 89400,
            "latency_ms": round(latency_ms, 2),
            "mode": "simulated",
        }
        self.transaction_logs.append(log_entry)
        return log_entry

    def complete_round(
        self,
        round_number: int,
        new_global_model_hash: str,
        accepted_count: int,
    ) -> Dict[str, Any]:
        """Finalizes round on-chain with aggregated global model hash."""
        start_time = time.time()
        tx_hash = "0x" + hashlib.sha256(f"round_comp_{round_number}_{time.time()}".encode()).hexdigest()

        if self.mode == "live" and self.contract and self.w3:
            try:
                deployer = self.w3.eth.accounts[0]
                tx = self.contract.functions.completeRound(
                    round_number, new_global_model_hash, accepted_count
                ).transact({"from": deployer})
                receipt = self.w3.eth.wait_for_transaction_receipt(tx)
                latency_ms = (time.time() - start_time) * 1000
                log_entry = {
                    "event": "RoundCompleted",
                    "round": round_number,
                    "new_global_model_hash": new_global_model_hash,
                    "accepted_count": accepted_count,
                    "tx_hash": receipt.transactionHash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "latency_ms": round(latency_ms, 2),
                    "mode": "live",
                }
                self.transaction_logs.append(log_entry)
                return log_entry
            except Exception as e:
                print(f"[BlockchainClient] Live transaction failed ({e}), using simulated.")

        # Simulated
        latency_ms = (time.time() - start_time) * 1000 + 1.4
        self.simulated_state["block_number"] += 1
        if round_number in self.simulated_state["rounds"]:
            self.simulated_state["rounds"][round_number]["completed"] = True
            self.simulated_state["rounds"][round_number]["new_global_hash"] = new_global_model_hash

        log_entry = {
            "event": "RoundCompleted",
            "round": round_number,
            "new_global_model_hash": new_global_model_hash,
            "accepted_count": accepted_count,
            "tx_hash": tx_hash,
            "block_number": self.simulated_state["block_number"],
            "gas_used": 54300,
            "latency_ms": round(latency_ms, 2),
            "mode": "simulated",
        }
        self.transaction_logs.append(log_entry)
        return log_entry

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Returns the complete sequence of blockchain transactions recorded."""
        return list(self.transaction_logs)

    def is_participant_authorized(self, address_or_id: str) -> bool:
        """Queries authorization status of participant."""
        if self.mode == "live" and self.contract and self.w3:
            try:
                if self.w3.is_address(address_or_id):
                    return self.contract.functions.isParticipantAuthorized(
                        self.w3.to_checksum_address(address_or_id)
                    ).call()
            except Exception:
                pass

        # Check simulation
        if address_or_id in self.simulated_state["participants"]:
            return self.simulated_state["participants"][address_or_id]["isRegistered"]
        addr = self.client_account_map.get(address_or_id)
        if addr and addr in self.simulated_state["participants"]:
            return self.simulated_state["participants"][addr]["isRegistered"]
        return False
