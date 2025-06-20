FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 步骤 2: 更新 pip 并设置 python3.11 为默认的 python 和 python3
RUN pip3 install --no-cache-dir -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 验证版本
RUN python --version && pip --version

WORKDIR /workspace

# 步骤 3: 安装 requirements.txt 中的依赖
# 确保 requirements.txt 中已包含 deepspeed
COPY requirements.txt .
RUN pip install --no-cache-dir  --timeout 600 -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 步骤 4: 安装其他必要的Python包，例如 'av'
RUN pip install av -i https://mirrors.aliyun.com/pypi/simple

# 步骤 5: 安装您本地的 qwen_vl_utils
COPY qwen_vl_utils /tmp/qwen_vl_utils/
RUN pip install --no-cache-dir /tmp/qwen_vl_utils && \
    rm -rf /tmp/qwen_vl_utils

# 步骤 6: 复制DeepSpeed配置文件 (假设您在构建目录下有一个ds_config.json文件)
COPY ds_config.json /workspace/ds_config.json

# 步骤 7: 复制应用代码
COPY train.py .

# ENV 和 CMD
ENV MODEL_DIR=/mnt/models/Qwen2.5VL \
    DATA_DIR=/mnt/Datasets \
    OUTPUT_DIR=/mnt/Checkpoints/Qwen2.5VL-lora \
    WANDB_MODE=offline \
    DS_CONFIG_PATH=/workspace/ds_config.json

# 使用 shell 格式的 CMD 来确保环境变量能够展开
CMD ["sh", "-c", "deepspeed /workspace/train.py --model_id \"$MODEL_DIR\" --dataset_id \"$DATA_DIR\" --output_dir \"$OUTPUT_DIR\" --deepspeed_config \"$DS_CONFIG_PATH\""]