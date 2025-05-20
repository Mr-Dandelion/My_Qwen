FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LANG=C.UTF-8
WORKDIR /app

COPY .. /app/

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


ENTRYPOINT ["python", "main.py"]



