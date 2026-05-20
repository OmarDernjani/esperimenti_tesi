#!/bin/bash
podman run -d \
  --name ollama \
  --device nvidia.com/gpu=3 \
  -v ollama_models:/root/.ollama \
  -p 11435:11434 \
  docker.io/ollama/ollama:latest
