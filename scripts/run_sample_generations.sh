#!/bin/bash
set -euo pipefail

echo "Using existing AlfWorld environment server on port 8001..."

# Get the cache directory and outputs directory
CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"

# Run sampling script in docker container with GPU access
echo "Running generation sampling in docker container..."
docker run --rm \
  --gpus all \
  --network host \
  --entrypoint python3 \
  -e ENV_SERVER_URL="http://localhost:8001" \
  -v "$CHECKPOINTS_DIR:/cache:ro" \
  -v "$OUTPUTS_DIR:/outputs:ro" \
  -v "$(pwd)/scripts:/scripts:ro" \
  standalone-text-trainer \
  /scripts/sample_alfworld_generations.py

echo "Done!"
