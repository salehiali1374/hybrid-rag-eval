FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HRE_DATA_DIR=/data HF_HOME=/hf

# CPU-only PyTorch first, so that sentence-transformers does not pull the CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dense]"

# /data: dataset and embedding cache, /hf: downloaded models (both are volumes in compose)
RUN useradd --create-home app && mkdir /data /hf && chown app:app /data /hf
USER app

EXPOSE 8000
CMD ["uvicorn", "hybrid_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
