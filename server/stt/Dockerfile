FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "faster-whisper==1.2.1" "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"
COPY server.py wavdecode.py ./

# Whisper models are downloaded on first start into /models (a Docker volume, so only once)
ENV HF_HOME=/models
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
