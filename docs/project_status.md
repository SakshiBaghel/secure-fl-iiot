# Project Status & Requirements Traceability

This document records the implementation status, architectural assumptions, and known limitations of the **Blockchain-Enabled Secure Federated Learning for IIoT** prototype.

---

## 1. Completed Features

| Category | Feature | Status | Implementation File | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset & Preprocessing** | Real dataset support (Edge-IIoTset, CIC-IDS2017) | Completed | `src/preprocess.py` | Load paths tested |
| | Labelled Synthetic IIoT fallback generator | Completed | `src/preprocess.py` | 15 IIoT features, unit tested |
| | Strict zero-data-leakage split & scaling | Completed | `src/preprocess.py` | Scaler fit only on train |
| | Partitioning across 3 simulated clients | Completed | `src/preprocess.py` | IID and Non-IID supported |
| **Machine Learning** | Centralized baseline (Random Forest) | Completed | `src/baseline.py` | Evaluated on held-out test set |
| | PyTorch MLP (`IIoTIntrusionMLP`) | Completed | `src/model.py` | BatchNorm, Dropout, ReLU |
| | Deterministic weight serialization | Completed | `src/model.py` | Lexicographical keys + float32 |
| | SHA-256 parameter hashing | Completed | `src/model.py` | 64-char hex integrity hash |
| **Federated Learning** | Client local training without raw data sharing | Completed | `src/client.py` | PyTorch Adam + CrossEntropy |
| | FedAvg sample-weighted aggregation | Completed | `src/fedavg.py` | Analytically verified in unit tests |
| | Evaluation metrics (Accuracy, Precision, Recall, F1, CM) | Completed | `src/evaluate.py` | Evaluated after each round |
| **Blockchain** | Solidity Smart Contract (`FLRegistry.sol`) | Completed | `blockchain/contracts/FLRegistry.sol` | 7 Mocha unit tests passed |
| | Participant registration & authorization | Completed | `blockchain/contracts/FLRegistry.sol` | Access control modifiers |
| | Communication round lifecycle tracking | Completed | `blockchain/contracts/FLRegistry.sol` | Start/Complete events emitted |
| | Update metadata & validation status logging | Completed | `blockchain/contracts/FLRegistry.sol` | Hashes and sample counts logged |
| | Web3.py Python client with fallback simulation | Completed | `src/blockchain_client.py` | Zero crash on offline RPC |
| **Security Mechanism** | Parameter deviation $L_2$-norm threshold filter | Completed | `src/validate_update.py` | Configurable threshold |
| | Controlled model poisoning experiment | Completed | `src/run_experiment.py` | Compared Honest vs Unprotected vs Protected |
| **Interface & GUI** | Command-Line Interface (CLI) | Completed | `src/run_experiment.py` | Configurable flags (`--rounds`, `--threshold`) |
| | Streamlit Interactive Dashboard | Completed | `app.py` | 5 interactive tabs |
| **Testing** | 21 Python Unit & Integration Tests | Completed | `tests/` | 100% pytest pass rate |
| | 7 Smart Contract Unit Tests | Completed | `blockchain/test/` | 100% Hardhat test pass rate |

---

## 2. Unimplemented Features (Out of Current Prototype Scope)

Per user prompt instructions: *"Do not add Differential Privacy, Homomorphic Encryption, Secure Aggregation, token incentives, custom consensus algorithms, or Split Learning unless the required system already works and time remains."*

1. **Differential Privacy ($\epsilon, \delta$-DP)**: Local gradient clipping with Gaussian noise injection to prevent reconstruction attacks.
2. **Homomorphic Encryption (Paillier/CKKS)**: Encrypted gradient aggregation where the aggregator cannot view plaintext weights.
3. **Tokenized Economic Incentives**: ERC-20 token rewards for honest updates and slashing mechanisms for malicious clients.
4. **Physical IoT Hardware Testbed**: Integration with physical Raspberry Pi or ESP32 microcontrollers.

---

## 3. Assumptions

1. **Simulated Multi-Client Topology**: The 3 industrial client nodes execute on a single physical host within isolated Python object instances, simulating independent industrial sites.
2. **Deterministic Serialization**: Floating-point parameters are serialized as IEEE 754 32-bit floats in standard byte-order to ensure hash reproducibility across clients with identical weights.
3. **Binary Intrusion Classification**: Network traffic is classified into binary categories: `0` (Normal Operational Telemetry) and `1` (Malicious Cyber-Attack).

---

## 4. Known Limitations

1. **Cryptographic Hashes vs. Model Maliciousness**: Storing a SHA-256 hash on a blockchain proves data integrity (that the weights have not been altered in transit or post-submission). It does **not** mathematically prove that the model update is benign or free from backdoor attacks.
2. **Norm-Threshold Heuristics**: The $L_2$-norm validator reliably blocks coarse poisoning attacks and gradient explosion. Stealthy, constrained poisoning attacks with small norms could evade simple thresholding; robust multi-metric aggregators (Krum, Coordinate-wise Median) are required for full Byzantine robustness.
3. **Off-Chain Model Storage**: Only parameter hashes and metadata are stored on-chain due to Ethereum block gas limits. Neural network weights are transferred off-chain.
