# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch CPU-only version for Python 3.11
RUN python -m pip install --prefix=/install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# Install the rest of your requirements
RUN python -m pip install --prefix=/install --no-cache-dir -r requirements.txt

# =========================
# Stage 2: Runtime
# =========================
FROM python:3.11-slim-bullseye

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy model and other files
COPY best_model.pth .
COPY mlb.pkl .
COPY . .

EXPOSE 5000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
