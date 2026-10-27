"""Unit tests for the Python Blockchain Client interface and on-chain record keeping."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.blockchain_client import BlockchainClient


@pytest.fixture
def bc_client():
    # Force simulated mode for headless pytest test isolation
    return BlockchainClient(prefer_simulation=True)


def test_participant_registration(bc_client):
    addr = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    log = bc_client.register_participant(participant_address=addr, client_id="client_1", role=0)

    assert log["event"] == "ParticipantRegistered"
    assert log["client_id"] == "client_1"
    assert log["tx_hash"].startswith("0x")
    assert bc_client.is_participant_authorized(addr) is True
    assert bc_client.is_participant_authorized("client_1") is True


def test_unregistered_participant_authorization(bc_client):
    assert bc_client.is_participant_authorized("0x0000000000000000000000000000000000000000") is False
    assert bc_client.is_participant_authorized("client_unknown") is False


def test_start_and_complete_training_round(bc_client):
    dummy_hash = "a" * 64
    start_log = bc_client.start_training_round(round_number=1, global_model_hash=dummy_hash)
    assert start_log["event"] == "RoundStarted"
    assert start_log["round"] == 1
    assert start_log["global_model_hash"] == dummy_hash

    new_hash = "b" * 64
    comp_log = bc_client.complete_round(round_number=1, new_global_model_hash=new_hash, accepted_count=3)
    assert comp_log["event"] == "RoundCompleted"
    assert comp_log["accepted_count"] == 3
    assert comp_log["new_global_model_hash"] == new_hash


def test_record_update_metadata(bc_client):
    bc_client.register_participant("0x70997970C51812dc3A010C7d01b50e0d17dc79C8", "client_1", 0)
    bc_client.start_training_round(1, "0" * 64)

    update_hash = "c" * 64
    rec_log = bc_client.record_update(
        round_number=1,
        client_id="client_1",
        update_hash=update_hash,
        sample_count=800,
        is_valid=True,
        reason_code=0,
    )

    assert rec_log["event"] == "UpdateRecorded"
    assert rec_log["client_id"] == "client_1"
    assert rec_log["update_hash"] == update_hash
    assert rec_log["sample_count"] == 800
    assert rec_log["is_valid"] is True
    assert rec_log["reason_code"] == 0


def test_audit_trail_retrieval(bc_client):
    bc_client.register_participant("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "client_2", 0)
    bc_client.start_training_round(1, "1" * 64)
    bc_client.record_update(1, "client_2", "2" * 64, 500, False, 1)

    trail = bc_client.get_audit_trail()
    assert len(trail) >= 3
    event_names = [t["event"] for t in trail]
    assert "ParticipantRegistered" in event_names
    assert "RoundStarted" in event_names
    assert "UpdateRecorded" in event_names
