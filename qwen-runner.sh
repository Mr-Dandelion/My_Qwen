#!/usr/bin/env bash

set -e

MODE=$1

if [[ "$MODE" == "vllm" ]]; then
  echo "[启动] 以 vLLM 模式运行 Qwen3-8B..."
  docker run --rm -it --gpus all --shm-size=16g \
    -p 192.168.76.222:8000:8000 \
    -v /home/lanfeng/models/Qwen3-8B:/models \
    qwen-runner:vllm \
    --model /models \
    --tokenizer /models \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --dtype float16 \
    --served-model-name qwen3-8b \
    --no-enable-chunked-prefill \
    --max-model-len 8192

elif [[ "$MODE" == "raw" ]]; then
  echo "[启动] 以 raw 模式运行 Qwen3-32B..."
  docker run --rm -it --gpus all \
    -v /home/lanfeng/models/Qwen3-32B:/app/models/Qwen3-32B \
    qwen-runner:raw \
    --model_path /app/models/Qwen3-32B
# --enable-thinking --do-sample
else
  echo "用法: $0 [vllm|raw]"
  exit 1
fi

