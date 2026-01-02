# FROM python:3

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app/ app/

# EXPOSE 8000

# Stage 1: Install dependencies
FROM python:3 AS builder
WORKDIR /app

COPY requirements.txt .

# Install dependencies in a user-local path
RUN pip install --upgrade pip setuptools wheel \
    && pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["gunicorn","app.main:app","-k", "uvicorn.workers.UvicornWorker","--bind", "0.0.0.0:8000","--workers", "2", "--timeout", "120"]