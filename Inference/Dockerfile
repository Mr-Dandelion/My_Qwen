FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir  --timeout 600 -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

RUN pip install av -i https://mirrors.aliyun.com/pypi/simple
COPY qwen_vl_utils /tmp/qwen_vl_utils/
RUN pip install --no-cache-dir /tmp/qwen_vl_utils && \
    rm -rf /tmp/qwen_vl_utils

COPY main.py /app/main.py

ENTRYPOINT ["python", "main.py"]

CMD ["--base_model_path", "/app/models/base_model", \
     "--adapter_path", "/app/models/adapter", \
     "--image_path", "/app/default_image.jpg", \
     "--prompt", "Describe this image."]