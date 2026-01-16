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
HUGGINGFACE_TOKEN="Your Huggingface Token"
WANDB_TOKEN=""
HUGGINGFACE_USERNAME="Your Huggingface Username"
EXPECTED_REPO_NAME="environment_test"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_CONFIG="${LAUNCHER_CONFIG:-$SCRIPT_DIR/environment_task_launcher.yml}"
BASE_ENV_CONFIG="${BASE_ENV_CONFIG:-$SCRIPT_DIR/../core/config/base_environment.yml}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python is required to parse $LAUNCHER_CONFIG." >&2
  exit 1
fi

if [[ ! -f "$LAUNCHER_CONFIG" ]]; then
  echo "Launcher config not found: $LAUNCHER_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$BASE_ENV_CONFIG" ]]; then
  echo "Base environment config not found: $BASE_ENV_CONFIG" >&2
  exit 1
fi

eval "$(
  LAUNCHER_CONFIG="$LAUNCHER_CONFIG" BASE_ENV_CONFIG="$BASE_ENV_CONFIG" "$PYTHON_BIN" - <<'PY'
import os
import re

def parse_yaml(path):
    data = {}
    stack = [(0, data)]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            line = raw.rstrip("\n")
            if "#" in line and line.count('"') % 2 == 0 and line.count("'") % 2 == 0:
                line = line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            if ":" not in text:
                continue
            key, val = text.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val == "":
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                parent = stack[-1][1] if stack else data
                parent[key] = {}
                stack.append((indent, parent[key]))
                continue

            if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                val = val[1:-1]
            elif val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try:
                    if re.match(r"^[0-9]+\\.[0-9]+$", val):
                        val = float(val)
                    elif re.match(r"^[0-9]+$", val):
                        val = int(val)
                except ValueError:
                    pass

            while stack and stack[-1][0] > indent:
                stack.pop()
            parent = stack[-1][1] if stack else data
            parent[key] = val
    return data

launcher_path = os.environ["LAUNCHER_CONFIG"]
base_env_path = os.environ["BASE_ENV_CONFIG"]

launcher = parse_yaml(launcher_path)
base_env = parse_yaml(base_env_path)

train_gpu_ids = str(launcher.get("train_gpu_ids", ""))
vllm_gpu_ids = str(launcher.get("vllm_gpu_ids", ""))
server = launcher.get("vllm_server", {}) or {}

base_trl = base_env.get("trl", {}) or {}
base_host = base_trl.get("vllm_server_host", "vllm-server")
base_port = base_trl.get("vllm_server_port", 8000)
group_port = base_trl.get("vllm_group_port", 51216)

server_host = server.get("host", base_host)
server_port = server.get("port", base_port)
tp_size = server.get("tensor_parallel_size", None)
gpu_mem = server.get("gpu_memory_utilization", 0.9)

def emit(name, value):
    print(f'{name}="{value}"')

emit("TRAIN_GPU_IDS", train_gpu_ids)
emit("VLLM_GPU_IDS", vllm_gpu_ids)
emit("VLLM_SERVER_HOST", server_host)
emit("VLLM_SERVER_PORT", server_port)
emit("VLLM_TENSOR_PARALLEL_SIZE", "" if tp_size is None else tp_size)
emit("VLLM_GPU_MEMORY_UTILIZATION", gpu_mem)
emit("BASE_TRL_VLLM_SERVER_HOST", base_host)
emit("BASE_TRL_VLLM_SERVER_PORT", base_port)
emit("BASE_TRL_VLLM_GROUP_PORT", group_port)
emit("LAUNCHER_GROUP_PORT", "" if server.get("group_port", None) is None else server.get("group_port"))
PY
)"

TRAIN_GPU_IDS="${TRAIN_GPU_IDS// /}"
VLLM_GPU_IDS="${VLLM_GPU_IDS// /}"

if [[ -z "$TRAIN_GPU_IDS" || -z "$VLLM_GPU_IDS" ]]; then
  echo "train_gpu_ids and vllm_gpu_ids must be set in $LAUNCHER_CONFIG." >&2
  exit 1
fi

IFS=',' read -ra TRAIN_GPUS <<< "$TRAIN_GPU_IDS"
IFS=',' read -ra VLLM_GPUS <<< "$VLLM_GPU_IDS"

declare -A TRAIN_SEEN
for gpu in "${TRAIN_GPUS[@]}"; do
  if [[ -z "$gpu" ]]; then
    continue
  fi
  if [[ -n "${TRAIN_SEEN[$gpu]:-}" ]]; then
    echo "Duplicate GPU id in train_gpu_ids: $gpu" >&2
    exit 1
  fi
  TRAIN_SEEN[$gpu]=1
done

declare -A VLLM_SEEN
for gpu in "${VLLM_GPUS[@]}"; do
  if [[ -z "$gpu" ]]; then
    continue
  fi
  if [[ -n "${VLLM_SEEN[$gpu]:-}" ]]; then
    echo "Duplicate GPU id in vllm_gpu_ids: $gpu" >&2
    exit 1
  fi
  VLLM_SEEN[$gpu]=1
done

for gpu in "${VLLM_GPUS[@]}"; do
  if [[ -n "${TRAIN_SEEN[$gpu]:-}" ]]; then
    echo "GPU id $gpu is in both train_gpu_ids and vllm_gpu_ids. These must be disjoint." >&2
    exit 1
  fi
done

if [[ "$BASE_TRL_VLLM_SERVER_HOST" != "$VLLM_SERVER_HOST" ]]; then
  echo "Mismatch: base_environment.yml vllm_server_host=$BASE_TRL_VLLM_SERVER_HOST but launcher host=$VLLM_SERVER_HOST" >&2
  exit 1
fi

if [[ "$BASE_TRL_VLLM_SERVER_PORT" != "$VLLM_SERVER_PORT" ]]; then
  echo "Mismatch: base_environment.yml vllm_server_port=$BASE_TRL_VLLM_SERVER_PORT but launcher port=$VLLM_SERVER_PORT" >&2
  exit 1
fi

if [[ -n "$LAUNCHER_GROUP_PORT" && "$LAUNCHER_GROUP_PORT" != "$BASE_TRL_VLLM_GROUP_PORT" ]]; then
  echo "Mismatch: base_environment.yml vllm_group_port=$BASE_TRL_VLLM_GROUP_PORT but launcher group_port=$LAUNCHER_GROUP_PORT" >&2
  exit 1
fi

if [[ -z "$VLLM_TENSOR_PARALLEL_SIZE" ]]; then
  VLLM_TENSOR_PARALLEL_SIZE="${#VLLM_GPUS[@]}"
fi

if [[ "$VLLM_TENSOR_PARALLEL_SIZE" -ne "${#VLLM_GPUS[@]}" ]]; then
  echo "vllm_server.tensor_parallel_size must match the number of vllm_gpu_ids." >&2
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 777 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 777 "$OUTPUTS_DIR"

VLLM_SERVER_CONTAINER="vllm-server"
TRAINER_CONTAINER="grpo-text-trainer-example"
NETWORK_NAME="envtask-net"
NETWORK_CREATED=0

cleanup() {
  docker rm -f "$TRAINER_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$VLLM_SERVER_CONTAINER" >/dev/null 2>&1 || true
  if [[ "$NETWORK_CREATED" -eq 1 ]]; then
    docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  docker network create "$NETWORK_NAME" >/dev/null
  NETWORK_CREATED=1
fi

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

MODEL_DIR_NAME="${MODEL//\//--}"
VLLM_MODEL_PATH="/cache/models/$MODEL_DIR_NAME"

echo "Starting vLLM server on GPUs [$VLLM_GPU_IDS]..."
docker run -d --rm \
  --gpus "device=$VLLM_GPU_IDS" \
  --network "$NETWORK_NAME" \
  --name "$VLLM_SERVER_CONTAINER" \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  standalone-text-trainer \
  trl vllm-serve \
  --model "$VLLM_MODEL_PATH" \
  --host "0.0.0.0" \
  --port "$VLLM_SERVER_PORT" \
  --tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"

echo "Starting training on GPUs [$TRAIN_GPU_IDS]..."
docker run --rm \
  --gpus "device=$TRAIN_GPU_IDS" \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network "$NETWORK_NAME" \
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