FROM qwen-base:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 不再 COPY 模型和图像，让用户挂载指定路径
ENTRYPOINT ["python3", "main.py"]
CMD []


