#!/usr/bin/env bash
# Local H200 serving + CPU client; no SSH tunnel or training models required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/podman.env
source CodePromptOpt/finetuning/downstream/h200.env
action="${1:-help}"
if (( $# )); then shift; fi
if [[ ! "$CPO_CODEGEN_EXPERIMENT" =~ ^[a-zA-Z0-9_-][a-zA-Z0-9_.-]*$ ]]; then
  echo 'Invalid CPO_CODEGEN_EXPERIMENT directory name' >&2; exit 2
fi
config="$CPO_CODEGEN_OUTPUT/$CPO_CODEGEN_EXPERIMENT/server-config.json"
case "$action" in
  build) bash CodePromptOpt/finetuning/downstream/client.sh build ;;
  config)
    printf 'MODEL=%s\nGPU=%s\nIMAGE=%s\nINPUT=%s\nOUTPUT=%s\nCONFIG=%s\n' \
      "$CODEGEN_MODEL" "$CODEGEN_GPU" "$VLLM_IMAGE" "$CPO_CODEGEN_INPUT" "$CPO_CODEGEN_OUTPUT/$CPO_CODEGEN_EXPERIMENT" "$config"
    ;;
  start)
    if (( $# )); then
      echo 'Configure server settings in h200.env; extra unrecorded vLLM arguments are not supported here.' >&2; exit 2
    fi
    # Catch overlap with the known training wrapper; also check nvidia-smi for
    # independent processes/containers before launching on this shared GPU.
    if podman container exists "$CPO_CONTAINER_NAME"; then
      if [[ "$(podman inspect --format '{{.State.Running}}' "$CPO_CONTAINER_NAME")" == true ]]; then
        echo "Training/optimizer container $CPO_CONTAINER_NAME is active. Wait before starting vLLM." >&2; exit 1
      fi
    fi
    if podman container exists "$CODEGEN_CONTAINER"; then
      echo "Container $CODEGEN_CONTAINER already exists: inspect logs/status or use h200.sh stop first." >&2; exit 1
    fi
    if ! podman image exists "$CPO_CLIENT_IMAGE"; then
      echo 'Build the CPU client first: bash CodePromptOpt/finetuning/downstream/h200.sh build' >&2; exit 1
    fi
    podman pull "$VLLM_IMAGE"
    image_id="$(podman image inspect --format '{{.Id}}' "$VLLM_IMAGE")"
    # Resolve the exact HF commit without downloading model weights to the host.
    revision="$(podman run --rm --pull=never --entrypoint python3 -e HF_TOKEN "$image_id" -c \
      'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)' \
      "$CODEGEN_MODEL" "$CODEGEN_REVISION")"
    [[ "$revision" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Could not resolve model commit' >&2; exit 1; }
    podman run --rm --pull=never -v "$PWD:/workspace:z" "$CPO_CLIENT_IMAGE" \
      python -m CodePromptOpt.finetuning.downstream.configure_h200 \
      --model "$CODEGEN_MODEL" --revision "$revision" --image "$image_id" \
      --context "$CODEGEN_CONTEXT" --gpu-util "$CODEGEN_GPU_UTIL" --max-seqs "$CODEGEN_MAX_SEQS" \
      --experiment "$CPO_CODEGEN_EXPERIMENT" --output "$config"
    # Serve the exact image and revision just recorded, even if a mutable tag moves.
    VLLM_IMAGE="$image_id" CODEGEN_REVISION="$revision" \
      bash CodePromptOpt/finetuning/downstream/serve_vllm.sh "$@"
    ;;
  logs) podman logs -f "$CODEGEN_CONTAINER" "$@" ;;
  status) podman inspect --format '{{.Name}}: {{.State.Status}} (exit={{.State.ExitCode}})' "$CODEGEN_CONTAINER" ;;
  ready)
    podman run --rm --pull=never --network=host -e CODEGEN_API_KEY "$CPO_CLIENT_IMAGE" python -c \
      'import sys; from CodePromptOpt.finetuning.downstream.run import request_json; print(request_json(sys.argv[1], timeout=10, retries=0))' \
      "http://127.0.0.1:$CODEGEN_PORT/v1/models"
    ;;
  plan|run)
    [[ -f "$config" ]] || { echo 'Start the server once to record its exact model/image config.' >&2; exit 1; }
    bash CodePromptOpt/finetuning/downstream/client.sh "$action" \
      --input-root "$CPO_CODEGEN_INPUT" --output-root "$CPO_CODEGEN_OUTPUT" --config "$config" \
      --base-url "http://127.0.0.1:$CODEGEN_PORT/v1" "$@"
    ;;
  stop)
    mkdir -p "$(dirname "$config")/logs"
    logfile="$(dirname "$config")/logs/server-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    podman stop "$CODEGEN_CONTAINER"
    podman logs "$CODEGEN_CONTAINER" > "$logfile" 2>&1
    podman rm "$CODEGEN_CONTAINER"
    echo "Server stopped; GPU memory released. Logs: $logfile. Model cache retained."
    ;;
  free-cache)
    associated="$(podman ps -aq --filter "volume=$CODEGEN_CACHE")"
    [[ -z "$associated" ]] || { echo 'Stop/remove containers using the code-generator cache first.' >&2; exit 1; }
    podman volume rm "$CODEGEN_CACHE"
    ;;
  *) echo 'Usage: bash CodePromptOpt/finetuning/downstream/h200.sh {build|config|start|logs|status|ready|plan|run|stop|free-cache} [options]' ;;
esac
