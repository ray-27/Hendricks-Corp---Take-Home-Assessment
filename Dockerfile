# syntax=docker/dockerfile:1.4
# Reproducible CPU environment for the Hendricks retail analytics take-home.
# Pose (YOLO11x) and ReID (NVIDIA TAO ONNX) weights download on first run.
#
# Build:
#   docker build -t hendricks-retail .
#
# torch/torchvision are installed from the CPU-only wheel index below --
# plain PyPI wheels for torch bundle full CUDA runtime libs (multiple GB)
# that this CPU-only image never uses, and were the actual cause of a
# build that hangs for 40+ minutes and then drops with an EOF from the
# Docker build channel. The CPU wheels are a fraction of the size.
#
# Run (mount videos; outputs written back to ./outputs):
#   docker run --rm \
#     -v "$(pwd)/raw_videos:/app/raw_videos:ro" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     hendricks-retail
#
# This image is CPU-only by design (see torch install below). For GPU, swap
# the --index-url line for the matching CUDA wheel index
# (https://download.pytorch.org/whl/cuXXX) and run with:
#   docker run --rm --gpus all \
#     -v "$(pwd)/raw_videos:/app/raw_videos:ro" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     hendricks-retail

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_QPA_PLATFORM=offscreen \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch/torchvision first, in their own cached layer (~200MB total
# instead of several GB of CUDA wheels), with a pip cache mount + retries so
# a transient network drop doesn't restart the whole download from zero.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Everything else. pip sees torch/torchvision already satisfy the
# requirements.txt version constraints and won't re-download them.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries 5 -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --default-timeout=100 --retries 5 opencv-python-headless>=4.8

COPY configs ./configs
COPY pipelines ./pipelines
COPY src ./src
COPY run_all.py README.md ./

# Videos are mounted at runtime. Config JSON in configs/ is copied in.
CMD ["python3", "run_all.py"]
