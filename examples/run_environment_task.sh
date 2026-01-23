#!/bin/bash
set -euo pipefail

TASK_ID="1"
MODEL="Qwen/Qwen2.5-3B-Instruct"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
DATASET_TYPE='{
  "environment_name": "alfworld"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=12

# For uploading the outputs
HUGGINGFACE_TOKEN=""
WANDB_TOKEN=""
HUGGINGFACE_USERNAME=""
WANDB_MODE="${WANDB_MODE:-online}"
EXPECTED_REPO_NAME="${EXPECTED_REPO_NAME:-environment_test}"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
DOCKER_BUILDKIT=1

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 777 "$CHECKPOINTS_DIR" 2>/dev/null || true
mkdir -p "$OUTPUTS_DIR"
chmod 777 "$OUTPUTS_DIR" 2>/dev/null || true

ENVIRONMENT_SERVER_CONTAINER_0="environment-server-0"
ENVIRONMENT_SERVER_CONTAINER_1="environment-server-1"
TRAINER_CONTAINER="grpo-text-trainer-example"
NETWORK_NAME="envtask-net"
NETWORK_CREATED=0

cleanup() {
  docker rm -f "$TRAINER_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$ENVIRONMENT_SERVER_CONTAINER_0" >/dev/null 2>&1 || true
  docker rm -f "$ENVIRONMENT_SERVER_CONTAINER_1" >/dev/null 2>&1 || true
  if [[ "$NETWORK_CREATED" -eq 1 ]]; then
    docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  docker network create "$NETWORK_NAME" >/dev/null
  NETWORK_CREATED=1
fi

echo "Starting environment servers (2 for parallel GPU access)..."
docker run -d \
  --name "$ENVIRONMENT_SERVER_CONTAINER_0" \
  --network "$NETWORK_NAME" \
  affinefoundation/agentgym:alfworld >/dev/null

docker run -d \
  --name "$ENVIRONMENT_SERVER_CONTAINER_1" \
  --network "$NETWORK_NAME" \
  affinefoundation/agentgym:alfworld >/dev/null

# Build the downloader image
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

echo "Starting training on GPUs [${CUDA_VISIBLE_DEVICES:-all}]..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network "$NETWORK_NAME" \
  -e WANDB_TOKEN="$WANDB_TOKEN" \
  -e WANDB_API_KEY="$WANDB_TOKEN" \
  -e WANDB_MODE="$WANDB_MODE" \
  -e ENVIRONMENT_SERVER_URLS="http://$ENVIRONMENT_SERVER_CONTAINER_0:8000,http://$ENVIRONMENT_SERVER_CONTAINER_1:8000" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -e HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  -e TASK_ID="$TASK_ID" \
  -e EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  -e LOCAL_FOLDER="$LOCAL_FOLDER" \
  -e MODEL="$MODEL" \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name "$TRAINER_CONTAINER" \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"