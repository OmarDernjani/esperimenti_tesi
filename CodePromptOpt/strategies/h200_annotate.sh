#!/usr/bin/env bash
# Dedicated annotation service. Run one profile at a time on the assigned GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
action="${1:-help}"
profile="${2:-qwen}"
source CodePromptOpt/finetuning/podman.env
export CPO_CLIENT_IMAGE="${CPO_CLIENT_IMAGE:-localhost/cpo-codegen-client:1}"
export VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:latest}"
export CODEGEN_GPU="${ANNOTATION_GPU:-$CPO_GPU}"
export CODEGEN_PORT="${ANNOTATION_PORT:-8001}"
export CODEGEN_CONTEXT=16384 CODEGEN_GPU_UTIL=0.80 CODEGEN_MAX_SEQS=1
export CODEGEN_CACHE="${ANNOTATION_CACHE:-cpo-annotation-hf-h200}"
export CODEGEN_CONTAINER=cpo-annotator-h200
root="${ANNOTATION_ROOT:-CodePromptOpt/strategies/artifacts/annotation_pilot_v2}"
extra=()
case "$profile" in
  qwen) export CODEGEN_MODEL=Qwen/Qwen2.5-32B-Instruct ;;
  mistral)
    export CODEGEN_MODEL=mistralai/Mistral-Small-24B-Instruct-2501
    extra=(--tokenizer-mode mistral --config-format mistral --load-format mistral)
    ;;
  *) echo 'Profile must be qwen or mistral' >&2; exit 2 ;;
esac
config="$root/$profile/server-config.json"
cpu() {
  podman run --rm --network=host -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" "$CPO_CLIENT_IMAGE" "$@"
}
case "$action" in
  build) bash CodePromptOpt/finetuning/downstream/client.sh build ;;
  prepare)
    cpu python -m CodePromptOpt.strategies.pipeline dataset \
      --output CodePromptOpt/strategies/artifacts/candidates.jsonl
    cpu python -m CodePromptOpt.strategies.annotation_study pilot --output "$root"
    ;;
  start)
    if podman container exists "$CODEGEN_CONTAINER"; then
      echo 'Annotator container already exists; inspect logs or stop it first.' >&2; exit 1
    fi
    active="$(nvidia-smi -i "$CODEGEN_GPU" --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "$active" ]]; then
      echo "GPU $CODEGEN_GPU has active compute processes. Finish the other job before starting annotation." >&2; exit 1
    fi
    if ! podman image exists "$VLLM_IMAGE"; then podman pull "$VLLM_IMAGE"; fi
    image_id="$(podman image inspect --format '{{.Id}}' "$VLLM_IMAGE")"
    revision="$(podman run --rm --entrypoint python3 -e HF_TOKEN "$image_id" -c \
      'from huggingface_hub import HfApi; import sys; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)' \
      "$CODEGEN_MODEL" "${ANNOTATION_REVISION:-main}")"
    [[ "$revision" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Invalid resolved model revision' >&2; exit 1; }
    cpu python -c '
import json, sys
from pathlib import Path
from CodePromptOpt.finetuning.downstream.configure_h200 import make_config
from CodePromptOpt.finetuning.downstream.run import atomic_json
path, model, revision, image, profile, gpu, port, *extra = sys.argv[1:]
cfg = make_config(model, revision, image, 16384, .80, 1, "annotation-" + profile)
cfg["server_metadata"].update(startup_extra_args=extra, annotation_profile=profile, host_gpu=gpu, port=port)
p = Path(path)
if p.exists() and json.loads(p.read_text()) != cfg:
    raise ValueError("Server protocol changed; pin image/revision or use a new ANNOTATION_ROOT")
atomic_json(p, cfg)
' "$config" "$CODEGEN_MODEL" "$revision" "$image_id" "$profile" "$CODEGEN_GPU" "$CODEGEN_PORT" "${extra[@]}"
    VLLM_IMAGE="$image_id" CODEGEN_REVISION="$revision" \
      bash CodePromptOpt/finetuning/downstream/serve_vllm.sh "${extra[@]}"
    ;;
  logs) podman logs -f "$CODEGEN_CONTAINER" ;;
  ready)
    cpu python -c 'import sys; from CodePromptOpt.finetuning.downstream.run import request_json; print(request_json(sys.argv[1], timeout=10, retries=0))' \
      "http://127.0.0.1:$CODEGEN_PORT/v1/models"
    ;;
  annotate)
    cpu python -c '
import json, sys, subprocess
from CodePromptOpt.finetuning.downstream.run import request_json
root, profile, base = sys.argv[1:]
cfg = json.load(open(root + "/" + profile + "/server-config.json"))
served = request_json(base + "/models", timeout=10, retries=0)
if cfg["model"] not in {m["id"] for m in served["data"]}:
    raise ValueError("Served model does not match annotator profile")
subprocess.run([sys.executable, "-m", "CodePromptOpt.strategies.infer",
    "--input", root + "/requests.jsonl", "--output", root + "/" + profile + "/responses.jsonl",
    "--model", cfg["model"], "--model-revision", cfg["server_metadata"]["model_revision"],
    "--server-config", root + "/" + profile + "/server-config.json", "--base-url", base], check=True)
' "$root" "$profile" "http://127.0.0.1:$CODEGEN_PORT/v1"
    ;;
  stop)
    mkdir -p "$root/logs"
    podman stop "$CODEGEN_CONTAINER"
    podman logs "$CODEGEN_CONTAINER" > "$root/logs/server-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
    podman rm "$CODEGEN_CONTAINER"
    ;;
  compare)
    cpu python -m CodePromptOpt.strategies.annotation_study compare --records "$root/records.jsonl" \
      --a "$root/qwen/responses.jsonl" --b "$root/mistral/responses.jsonl" --output "$root/comparison"
    ;;
  *) echo 'Usage: bash CodePromptOpt/strategies/h200_annotate.sh {build|prepare|start|logs|ready|annotate|stop|compare} [qwen|mistral]' ;;
esac
