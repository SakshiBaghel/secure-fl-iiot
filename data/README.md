# IIoT Intrusion Detection Datasets

This directory holds the network traffic datasets for the Blockchain-Enabled Secure Federated Learning pipeline.

## 1. Supported Real Datasets

### A. Edge-IIoTset (Recommended for Industrial IoT)
- **Source**: IEEE Dataport / University of Western Ontario
- **Reference**: Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications"
- **Download**: Obtain the CSV version (e.g. `Selected dataset for ML and DL.csv` or `Edge-IIoTset dataset.csv`)
- **Placement**: Place the CSV file into this directory and name it:
  ```
  data/edge_iiotset.csv
  ```
  Or set `DATASET_PATH=data/edge_iiotset.csv` in your `.env` file.

### B. CIC-IDS2017
- **Source**: Canadian Institute for Cybersecurity (UNB)
- **Reference**: Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
- **Placement**: Place any day's CSV (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) into:
  ```
  data/cic_ids2017.csv
  ```
  Or set `DATASET_PATH=data/cic_ids2017.csv` in your `.env`.

---

## 2. Automated Synthetic IIoT Fallback Mode

If no external CSV is placed in `data/`, the pipeline will automatically generate and use a clearly labelled synthetic dataset:
```
data/synthetic_iiot_traffic.csv
```
This synthetic dataset simulates 15 realistic IIoT sensor and SCADA network features (flow duration, packet size stats, TCP flag counts, port entropy, flow rates, sensor telemetry) with both benign operational telemetry and industrial cyber-attacks (DDoS, Port Scan, Command Injection).

> **Note on Data Authenticity**: The synthetic dataset is explicitly tagged with `is_synthetic = True` and is provided exclusively for automated unit testing and end-to-end pipeline demonstration when raw multi-gigabyte PCAP/CSV captures are not pre-downloaded.
