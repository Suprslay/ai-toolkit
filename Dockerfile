FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set python3.10 as the default python (already available in this image)
RUN ln -sf $(which python3.10) /usr/local/bin/python && \
    ln -sf $(which python3.10) /usr/local/bin/python3

# Set working directory
WORKDIR /workspace

# Install system dependencies for image processing and S3 access
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN uv pip install --upgrade -r /workspace/requirements.txt --no-cache-dir --system

# Copy the entire toolkit and configuration
COPY . /workspace/

# Ensure the config directory exists
RUN mkdir -p /workspace/config

# Copy the config file specifically (adjust path as needed)
COPY config/train_flux.yaml /workspace/config/train_flux.yaml

# Copy handler file
COPY handler.py /workspace/handler.py

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV DISABLE_TELEMETRY=YES
ENV PYTHONPATH=/workspace

# Hugging Face authentication
# The HF_TOKEN will be set at runtime via RunPod template
ENV HF_TOKEN=${HF_TOKEN}

# Perform Hugging Face login if token is provided
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "Logging into Hugging Face..." && \
        huggingface-cli login --token $HF_TOKEN; \
    else \
        echo "No HF_TOKEN provided - skipping Hugging Face login"; \
    fi

# AWS S3 Configuration (optional - can be set at runtime)
# ENV AWS_ACCESS_KEY_ID=your_access_key_here
# ENV AWS_SECRET_ACCESS_KEY=your_secret_key_here
# ENV AWS_DEFAULT_REGION=us-east-1

# Create datasets directory
RUN mkdir -p /workspace/datasets/input

# Set permissions
RUN chmod +x /workspace/handler.py

# Run the handler
CMD ["python", "-u", "/workspace/handler.py"]