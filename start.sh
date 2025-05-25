#!/bin/bash
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face..."
    huggingface-cli login --token $HF_TOKEN
fi
exec python -u /workspace/handler.py