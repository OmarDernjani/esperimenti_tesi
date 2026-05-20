#!/bin/bash
podman build -t experiment .

podman run -d \
  --name experiment \
  -e OLLAMA_HOST=http://127.0.0.1:11435 \
  -v "$(pwd)":/workspace \
  -v "$(pwd)":/results \
  --network host \
  -w /workspace \
  experiment python resume_missing.py
