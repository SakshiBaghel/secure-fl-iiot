# Project Report Outline: Blockchain-Enabled Secure Federated Learning for IIoT

## Title
**Blockchain-Enabled Secure Federated Learning for the Industrial Internet of Things (IIoT)**

---

## 1. Introduction & Problem Statement
- **Context**: The rapid expansion of Industry 4.0 and cyber-physical systems (CPS) introduces sophisticated network vulnerabilities across industrial control networks (SCADA, Modbus, MQTT).
- **Core Challenge**:
  - Traditional centralized Intrusion Detection Systems (IDS) create single points of failure and demand central pooling of raw telemetry data, violating commercial privacy and trade secret protections.
  - Conventional collaborative IDS suffer from trust deficits, lack of auditability, and vulnerability to malicious model poisoning attacks.
- **Proposed Solution**:
  - Decentralized collaborative intrusion detection via Federated Learning (FedAvg) where raw industrial telemetry remains local.
  - Integration of an Ethereum/Solidity smart contract (`FLRegistry.sol`) for immutable participant authorization, communication round tracking, and parameter hash auditing.
  - A pluggable update validation module ensuring contaminated or poisoned model updates are detected, recorded, and excluded prior to aggregation.

---

## 2. Literature Review
- **Evolution of Collaborative IDS**: Signature-based detection to distributed machine learning and collaborative anomaly detection.
- **Federated Learning in Cyber-Physical Systems**: McMahan et al. (FedAvg) and privacy-preserving collaborative ML in industrial domains.
- **Blockchain in Threat Intelligence & Security**: Immutable ledgers for auditability, smart contracts for automated access control, and hybrid consensus models.
- **Reference Paper Mapping**: Building directly upon the architecture presented in *"Blockchain-Integrated Collaborative Intrusion Detection Systems: Enhancing Network Security"*, specifically operationalizing the theoretical federated learning and smart contract recording layers.

---

## 3. Methodology & System Architecture
- **Threat Model**:
  - Semi-honest or Byzantine industrial participants.
  - Adversarial gradient injection / model poisoning attacks.
- **Mathematical Formulations**:
  - **FedAvg Aggregation**:
    $$W_{t+1} = \sum_{k \in \mathcal{S}_{accepted}} \frac{n_k}{N_{accepted}} W_{t}^k$$
  - **$L_2$ Deviation Validation**:
    $$\|\Delta W^k\|_2 = \|W_{t}^k - W_t\|_2 \le \theta_{threshold}$$
  - **SHA-256 Parameter Serialization**:
    $$H(W) = \text{SHA-256}(\text{SortKeys}(W) \parallel \text{Bytes}(W))$$
- **Smart Contract Interface**:
  - Data structure of `ModelUpdateRecord` and permissioned access control (`onlyOwner`, `onlyRegistered`).

---

## 4. Implementation Details
- **Dataset Preprocessing**: Zero-leakage data preparation, synthetic IIoT telemetry generator (15 features: flow durations, Modbus function codes, packet rates, port entropy), and support for Edge-IIoTset / CIC-IDS2017.
- **Deep Learning Model**: PyTorch `IIoTIntrusionMLP` with batch normalization and dropout for stable distributed convergence.
- **Blockchain Layer**: Solidity smart contract (`FLRegistry.sol`) tested in Hardhat with Web3.py Python integration and graceful fallback.
- **Visualization**: Interactive Streamlit GUI (`app.py`) and Matplotlib metric visualization.

---

## 5. Experimental Evaluation & Results
- **Benchmark Performance**:
  - Centralized Baseline: Random Forest benchmark (100% Accuracy, 1.0000 F1-Score).
  - Federated Global Model: 5 communication rounds achieving comparable 100% Accuracy, 100% Recall (Detection Rate), and 1.0000 F1-Score without sharing raw records.
- **Security Experiment**:
  - Attack scenario where a rogue participant (`client_3`) submits poisoned model weights ($L_2 = 1214.95$).
  - Unprotected outcome: F1-score collapses to 0.0000.
  - Protected outcome: Update validator detects anomaly, logs rejection on blockchain, aggregates honest clients, preserving 100% Accuracy and 1.0000 F1-Score.

---

## 6. Critical Security Discussion & Limitations
- **Hash Verification vs. Model Benignness**: Clarifying that an on-chain cryptographic hash guarantees update integrity and non-repudiation, not that the model is free from adversarial poisoning.
- **Threshold Sensitivity**: The limitations of norm-based filtering against subtle, small-norm backdoor attacks.
- **Blockchain Storage Constraints**: Why large raw datasets and full neural network parameters must reside off-chain, while only deterministic hashes and validation metadata are stored on-chain.

---

## 7. Conclusion & Future Research Directions
- **Summary**: Demonstration of an auditable, privacy-preserving collaborative intrusion detection system for IIoT.
- **Future Work**:
  - Byzantine-robust aggregation rules (Multi-Krum, Bulyan, Coordinate-wise Median).
  - Differential Privacy ($\epsilon, \delta$-DP) to protect against membership inference.
  - Hardware deployment on physical industrial PLCs (Raspberry Pi / Jetson Nano testbeds).
