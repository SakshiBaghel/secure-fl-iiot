const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Deploying FLRegistry smart contract to local network...");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  const FLRegistry = await hre.ethers.getContractFactory("FLRegistry");
  const registry = await FLRegistry.deploy();
  await registry.waitForDeployment();

  const contractAddress = await registry.getAddress();
  console.log("FLRegistry deployed successfully at address:", contractAddress);

  // Prepare deployment artifact for Python Web3 client
  const artifactPath = path.join(
    __dirname,
    "../artifacts/contracts/FLRegistry.sol/FLRegistry.json"
  );
  let artifact = {};
  if (fs.existsSync(artifactPath)) {
    artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  }

  const deploymentInfo = {
    address: contractAddress,
    deployer: deployer.address,
    network: hre.network.name,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    abi: artifact.abi || [],
    deployedAt: new Date().toISOString(),
  };

  const outputPath = path.join(__dirname, "../deployment_info.json");
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  console.log("Deployment details saved to:", outputPath);
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
