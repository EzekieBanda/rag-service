# -----------------------------
# Stage 1: Builder for dependencies
# -----------------------------
FROM python:3 AS builder

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies in user-local path (avoids root issues)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --user --no-cache-dir -r requirements.txt

# -----------------------------
# Stage 2: Runtime image
# -----------------------------
FROM python:3

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY ./app ./app

# Set environment variables for RAG service
ENV DATA_PATH=/data
ENV INDEX_PATH=/index
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=gpt-oss:20b

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["gunicorn","app.main:app","-k", "uvicorn.workers.UvicornWorker","--bind", "0.0.0.0:8000","--workers", "2", "--timeout", "120"]