// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FLRegistry
 * @dev Secure registry for Blockchain-Enabled Federated Learning in Industrial IoT.
 *
 * Implements decentralized participant authorization, communication round tracking,
 * cryptographic update hash recording, and security validation auditing.
 *
 * NOTE: Storing hashes on-chain verifies integrity and tamper-resistance of model
 * submissions; it does NOT prove that an update is mathematically benign. Security
 * validation occurs prior to FedAvg aggregation.
 */
contract FLRegistry {
    address public owner;
    uint256 public currentRound;
    uint256 public totalUpdatesCount;

    enum ParticipantRole { WORKER_CLIENT, AGGREGATOR, AUDITOR }

    struct Participant {
        bool isRegistered;
        string clientId;
        ParticipantRole role;
        uint256 registeredAt;
    }

    struct ModelUpdateRecord {
        string clientId;
        string updateHash;     // SHA-256 hash of model parameters
        uint256 sampleCount;   // Number of local training samples
        bool isValid;          // Outcome of update validation
        uint8 reasonCode;      // Validation reason (0=ACCEPTED, 1=EXCEEDED_L2, etc.)
        uint256 timestamp;
        address submittedBy;
    }

    // Mappings
    mapping(address => Participant) public participants;
    mapping(string => address) public clientAddressMap;
    mapping(uint256 => ModelUpdateRecord[]) private roundUpdates;
    mapping(uint256 => string) public roundGlobalModelHash;
    mapping(uint256 => bool) public roundCompleted;

    // Events
    event ParticipantRegistered(address indexed participantAddr, string clientId, ParticipantRole role, uint256 timestamp);
    event ParticipantRevoked(address indexed participantAddr, string clientId, uint256 timestamp);
    event RoundStarted(uint256 indexed roundNumber, string previousGlobalHash, uint256 timestamp);
    event UpdateRecorded(
        uint256 indexed roundNumber,
        string clientId,
        string updateHash,
        uint256 sampleCount,
        bool isValid,
        uint8 reasonCode,
        uint256 timestamp
    );
    event RoundCompleted(uint256 indexed roundNumber, string newGlobalHash, uint256 acceptedUpdatesCount, uint256 timestamp);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "FLRegistry: Caller is not the authorized owner");
        _;
    }

    modifier onlyRegistered() {
        require(participants[msg.sender].isRegistered || msg.sender == owner, "FLRegistry: Caller is not registered");
        _;
    }

    constructor() {
        owner = msg.sender;
        currentRound = 0;
        totalUpdatesCount = 0;
        // Register contract creator as Aggregator
        participants[msg.sender] = Participant({
            isRegistered: true,
            clientId: "aggregator_server",
            role: ParticipantRole.AGGREGATOR,
            registeredAt: block.timestamp
        });
    }

    /**
     * @dev Registers an authorized industrial participant node.
     */
    function registerParticipant(
        address _participant,
        string memory _clientId,
        ParticipantRole _role
    ) external onlyOwner {
        require(_participant != address(0), "FLRegistry: Invalid participant address");
        require(bytes(_clientId).length > 0, "FLRegistry: ClientId cannot be empty");

        participants[_participant] = Participant({
            isRegistered: true,
            clientId: _clientId,
            role: _role,
            registeredAt: block.timestamp
        });
        clientAddressMap[_clientId] = _participant;

        emit ParticipantRegistered(_participant, _clientId, _role, block.timestamp);
    }

    /**
     * @dev Revokes a participant's authorization.
     */
    function revokeParticipant(address _participant) external onlyOwner {
        require(participants[_participant].isRegistered, "FLRegistry: Participant not registered");
        string memory cid = participants[_participant].clientId;
        participants[_participant].isRegistered = false;

        emit ParticipantRevoked(_participant, cid, block.timestamp);
    }

    /**
     * @dev Initializes a new federated training communication round.
     */
    function startTrainingRound(uint256 _round, string memory _globalModelHash) external onlyOwner {
        require(_round > currentRound || (_round == 1 && currentRound == 0), "FLRegistry: Round must be greater than current");
        currentRound = _round;
        roundGlobalModelHash[_round] = _globalModelHash;
        roundCompleted[_round] = false;

        emit RoundStarted(_round, _globalModelHash, block.timestamp);
    }

    /**
     * @dev Records metadata for an industrial client's model update submission.
     */
    function recordUpdate(
        uint256 _round,
        string memory _clientId,
        string memory _updateHash,
        uint256 _sampleCount,
        bool _isValid,
        uint8 _reasonCode
    ) external onlyRegistered {
        require(_round == currentRound, "FLRegistry: Can only record update for current round");
        require(bytes(_updateHash).length == 64, "FLRegistry: Invalid SHA-256 hash length");
        require(_sampleCount > 0, "FLRegistry: Sample count must be > 0");

        ModelUpdateRecord memory record = ModelUpdateRecord({
            clientId: _clientId,
            updateHash: _updateHash,
            sampleCount: _sampleCount,
            isValid: _isValid,
            reasonCode: _reasonCode,
            timestamp: block.timestamp,
            submittedBy: msg.sender
        });

        roundUpdates[_round].push(record);
        totalUpdatesCount++;

        emit UpdateRecorded(
            _round,
            _clientId,
            _updateHash,
            _sampleCount,
            _isValid,
            _reasonCode,
            block.timestamp
        );
    }

    /**
     * @dev Finalizes communication round with aggregated global model hash.
     */
    function completeRound(
        uint256 _round,
        string memory _newGlobalModelHash,
        uint256 _acceptedUpdatesCount
    ) external onlyOwner {
        require(_round == currentRound, "FLRegistry: Round mismatch");
        require(!roundCompleted[_round], "FLRegistry: Round already completed");

        roundCompleted[_round] = true;
        roundGlobalModelHash[_round] = _newGlobalModelHash;

        emit RoundCompleted(_round, _newGlobalModelHash, _acceptedUpdatesCount, block.timestamp);
    }

    // View functions
    function isParticipantAuthorized(address _participant) external view returns (bool) {
        return participants[_participant].isRegistered;
    }

    function getUpdatesCount(uint256 _round) external view returns (uint256) {
        return roundUpdates[_round].length;
    }

    function getUpdateRecord(uint256 _round, uint256 _index) external view returns (
        string memory clientId,
        string memory updateHash,
        uint256 sampleCount,
        bool isValid,
        uint8 reasonCode,
        uint256 timestamp,
        address submittedBy
    ) {
        require(_index < roundUpdates[_round].length, "FLRegistry: Index out of range");
        ModelUpdateRecord storage rec = roundUpdates[_round][_index];
        return (
            rec.clientId,
            rec.updateHash,
            rec.sampleCount,
            rec.isValid,
            rec.reasonCode,
            rec.timestamp,
            rec.submittedBy
        );
    }
}
