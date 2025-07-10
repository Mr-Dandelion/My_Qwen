FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# 安装 Python 和基础依赖
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# 更新 pip（防止 cuda 镜像自带版本过低）
RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install av -i https://mirrors.aliyun.com/pypi/simple
COPY qwen_vl_utils /tmp/qwen_vl_utils/
RUN pip install --no-cache-dir /tmp/qwen_vl_utils && \
    rm -rf /tmp/qwen_vl_utils

COPY build_api.py /app/api_server.py
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

