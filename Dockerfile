FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN python -c 'from huggingface_hub import snapshot_download; snapshot_download("intfloat/multilingual-e5-base", allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"])'

COPY Task2/knowledge_base/ Task2/knowledge_base/
COPY Task3/qdrant_data/ Task3/qdrant_data/
COPY Task5/bot.py Task5/filters.py Task5/

ENTRYPOINT ["python", "Task5/bot.py"]
