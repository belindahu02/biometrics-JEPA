FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory inside the container
WORKDIR /app

# Install Python, pip, and other basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    git \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install PyTorch and Python dependencies
RUN pip install --no-cache-dir \
    torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    -r requirements.txt

# Copy the rest of the codebase into the container
COPY . .

# Ensure .project-root exists (for rootutils)
RUN touch .project-root

# Set Python path to make src/ imports work
ENV PYTHONPATH=/app:$PYTHONPATH

# Optional: define envs used in training code
ENV LOG_DIR=/app/logs
ENV CHECKPOINT_DIR=/app/checkpoints
ENV OUTPUT_DIR=/app/outputs
ENV DATA_PATH=/app/data

# Set Dora as entrypoint
ENTRYPOINT ["dora"]
CMD ["run", "src/train.py"]
