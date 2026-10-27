const { expect } = require("chai");
const hre = require("hardhat");
const { ethers } = hre;

describe("FLRegistry Smart Contract Tests", function () {
  let FLRegistry;
  let registry;
  let owner;
  let client1;
  let client2;
  let unauthorizedUser;

  beforeEach(async function () {
    [owner, client1, client2, unauthorizedUser] = await ethers.getSigners();
    FLRegistry = await ethers.getContractFactory("FLRegistry");
    registry = await FLRegistry.deploy();
    await registry.waitForDeployment();
  });

  it("Should set the deployer as owner and register deployer as aggregator", async function () {
    expect(await registry.owner()).to.equal(owner.address);
    const participant = await registry.participants(owner.address);
    expect(participant.isRegistered).to.be.true;
    expect(participant.clientId).to.equal("aggregator_server");
  });

  it("Should register authorized industrial clients", async function () {
    await expect(
      registry.registerParticipant(client1.address, "client_1", 0)
    )
      .to.emit(registry, "ParticipantRegistered")
      .withArgs(client1.address, "client_1", 0, (val) => val > 0);

    const p = await registry.participants(client1.address);
    expect(p.isRegistered).to.be.true;
    expect(p.clientId).to.equal("client_1");
  });

  it("Should reject registration attempts from non-owner", async function () {
    await expect(
      registry.connect(unauthorizedUser).registerParticipant(client2.address, "client_2", 0)
    ).to.be.revertedWith("FLRegistry: Caller is not the authorized owner");
  });

  it("Should start training round and emit RoundStarted event", async function () {
    const dummyHash = "a".repeat(64);
    await expect(registry.startTrainingRound(1, dummyHash))
      .to.emit(registry, "RoundStarted")
      .withArgs(1, dummyHash, (val) => val > 0);

    expect(await registry.currentRound()).to.equal(1);
    expect(await registry.roundGlobalModelHash(1)).to.equal(dummyHash);
  });

  it("Should record model update metadata from registered participant", async function () {
    await registry.registerParticipant(client1.address, "client_1", 0);
    const globalHash = "a".repeat(64);
    await registry.startTrainingRound(1, globalHash);

    const updateHash = "b".repeat(64);
    await expect(
      registry.connect(client1).recordUpdate(1, "client_1", updateHash, 800, true, 0)
    )
      .to.emit(registry, "UpdateRecorded")
      .withArgs(1, "client_1", updateHash, 800, true, 0, (val) => val > 0);

    expect(await registry.getUpdatesCount(1)).to.equal(1);
    const rec = await registry.getUpdateRecord(1, 0);
    expect(rec[0]).to.equal("client_1");
    expect(rec[1]).to.equal(updateHash);
    expect(rec[2]).to.equal(800n);
    expect(rec[3]).to.equal(true);
    expect(rec[4]).to.equal(0);
  });

  it("Should reject update submission from unregistered caller", async function () {
    const globalHash = "a".repeat(64);
    await registry.startTrainingRound(1, globalHash);
    const updateHash = "c".repeat(64);

    await expect(
      registry.connect(unauthorizedUser).recordUpdate(1, "client_bad", updateHash, 500, false, 3)
    ).to.be.revertedWith("FLRegistry: Caller is not registered");
  });

  it("Should complete round and emit RoundCompleted", async function () {
    await registry.startTrainingRound(1, "a".repeat(64));
    const newGlobalHash = "d".repeat(64);

    await expect(registry.completeRound(1, newGlobalHash, 3))
      .to.emit(registry, "RoundCompleted")
      .withArgs(1, newGlobalHash, 3, (val) => val > 0);

    expect(await registry.roundCompleted(1)).to.be.true;
  });
});
