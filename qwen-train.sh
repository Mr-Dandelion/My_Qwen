#!/usr/bin/env bash
set -euo pipefail

#############################
# 1. CONFIGURABLE VARIABLES #
#############################
# Host paths – edit or export these before running
DATA_DIR=${DATA_DIR:-/home/lanfeng/Datasets/SpaceThinker}
MODEL_DIR=${MODEL_DIR:-/home/lanfeng/models/Qwen2.5-VL-7B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-/home/lanfeng/Checkpoints/Qwen2.5VL-7B-lora}
CACHE_DIR=${CACHE_DIR:-/home/lanfeng/hf_cache}

# Docker image/tag
IMAGE_NAME=${IMAGE_NAME:-qwen-runner:deepspeed}

# Path to Dockerfile (assumed to sit next to this script)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKERFILE_DIR=${DOCKERFILE_DIR:-"${SCRIPT_DIR}"}

##################################
# 2. BUILD IMAGE IF NOT PRESENT  #
##################################
if [[ -z $(docker images -q "${IMAGE_NAME}") ]]; then
  echo "[+] Building Docker image: ${IMAGE_NAME}"
  docker build -t "${IMAGE_NAME}" "${DOCKERFILE_DIR}"
else
  echo "[+] Docker image ${IMAGE_NAME} already exists. Skipping build."
fi

#############################
# 3. LAUNCH TRAINING RUN    #
#############################
CMD=(
  python /workspace/train.py \
    --model_id /mnt/models/Qwen2.5VL \
    --dataset_id /mnt/Datasets/SpaceThinker \
    --output_dir /mnt/Checkpoints/Qwen2.5VL-lora \
    --deepspeed_config /workspace/ds_config.json \
    "$@"
)

echo "[+] Starting container and running training:"
echo "    ${CMD[*]}"

docker run --gpus all --rm -it \
  -v "${MODEL_DIR}:/mnt/models/Qwen2.5VL" \
  -v "${DATA_DIR}:/mnt/Datasets/SpaceThinker" \
  -v "${OUTPUT_DIR}:/mnt/Checkpoints/Qwen2.5VL-lora" \
  -v "${CACHE_DIR}:/mnt/hf_cache" \
  -e HF_HOME="/mnt/hf_cache" \
  "${IMAGE_NAME}" \
  ${CMD[*]}