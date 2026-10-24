#!/usr/bin/env bash
# Linux server launcher. Spark defaults to GPU 0; h200.sh supplies GPU 3.
set -euo pipefail
: "${VLLM_IMAGE:?Set VLLM_IMAGE to a Spark/ARM64-compatible vLLM image tag or digest from the NVIDIA/vLLM recipe}"
model="${CODEGEN_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
name="${CODEGEN_CONTAINER:-cpo-vllm}"
gpu="${CODEGEN_GPU:-0}"
port="${CODEGEN_PORT:-8000}"
cache="${CODEGEN_CACHE:-cpo-vllm-hf}"
if ! podman image exists "$VLLM_IMAGE"; then
  podman pull "$VLLM_IMAGE"
fi
echo 'Host architecture:' "$(uname -m)"
podman image inspect --format 'Image architecture={{.Architecture}} ID={{.Id}}' "$VLLM_IMAGE"
if [[ "$(uname -m)" == aarch64 && "$(podman image inspect --format '{{.Architecture}}' "$VLLM_IMAGE")" != arm64 ]]; then
  echo 'Spark requires an ARM64 image; select the image from its supported recipe.' >&2; exit 1
fi
podman volume create "$cache" >/dev/null
extra=()
[[ -z "${CODEGEN_REVISION:-}" ]] || extra+=(--revision "$CODEGEN_REVISION")
[[ -z "${CODEGEN_API_KEY:-}" ]] || extra+=(--api-key "$CODEGEN_API_KEY")
podman run -d --name "$name" --device "nvidia.com/gpu=$gpu" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-driver=k8s-file --log-opt=max-size=50mb \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_TOKEN \
  -p "127.0.0.1:$port:8000" -v "$cache:/root/.cache/huggingface" \
  --entrypoint vllm "$VLLM_IMAGE" serve "$model" --served-model-name "$model" \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len "${CODEGEN_CONTEXT:-16384}" \
  --gpu-memory-utilization "${CODEGEN_GPU_UTIL:-0.70}" --max-num-seqs "${CODEGEN_MAX_SEQS:-4}" \
  --generation-config vllm "${extra[@]}" "$@"
echo "Follow startup: podman logs -f $name"
echo "API: http://127.0.0.1:$port/v1 (wait until the server reports ready)"
echo 'Only a code generator is loaded. Fine-tuned prompt optimizers are not needed.'
