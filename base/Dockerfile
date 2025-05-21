# ---------- Stage 0: base ----------
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple


