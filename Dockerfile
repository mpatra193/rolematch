FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX embedding model into a writable, world-readable cache,
# so a non-root container (Render, HF, …) can read it and boot fast.
ENV FASTEMBED_CACHE_PATH=/app/.cache HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')" && \
    chmod -R 777 /app/.cache

COPY . .

# Listen on $PORT when the host sets one (Hugging Face Spaces, Render, …),
# defaulting to 8000 for local docker-compose. Shell form so $PORT expands.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
