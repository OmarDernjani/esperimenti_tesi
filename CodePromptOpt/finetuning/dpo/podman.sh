#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/dpo/podman.env
action="${1:-help}"
if (( $# )); then shift; fi
cpu() {
  podman run --rm --pull=never --log-driver=k8s-file --log-opt=max-size=20mb \
    -v "$PWD:/workspace:z" -v "$CPO_DPO_CACHE:/cache" -e HF_TOKEN \
    -e PYTHONUNBUFFERED=1 "$CPO_DPO_IMAGE" "$@"
}
launch() {
  if podman container exists "$CPO_DPO_CONTAINER"; then
    echo "Container $CPO_DPO_CONTAINER exists. Use await-job after success; inspect failed runs before removing it." >&2
    exit 1
  fi
  # Prevent overlap with the existing project GPU jobs. Other users/processes
  # still need to be checked with nvidia-smi on the shared server.
  for other in "cpo-gpu-$CPO_DPO_GPU" cpo-vllm-h200 cpo-vllm-qwen3-30b-h200; do
    if podman container exists "$other"; then
      if [[ "$(podman inspect --format '{{.State.Running}}' "$other")" == true ]]; then
        echo "Project GPU container $other is active; stop it before DPO." >&2; exit 1
      fi
    fi
  done
  podman run -d --pull=never --name "$CPO_DPO_CONTAINER" \
    --label "cpo.dpo.root=$CPO_DPO_ROOT" --log-driver=k8s-file --log-opt=max-size=20mb \
    --device "nvidia.com/gpu=$CPO_DPO_GPU" --shm-size=8g \
    -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_TOKEN -e PYTHONUNBUFFERED=1 \
    -v "$PWD:/workspace:z" -v "$CPO_DPO_CACHE:/cache" "$CPO_DPO_IMAGE" "$@"
  echo "Detached. Monitor: bash CodePromptOpt/finetuning/dpo/podman.sh logs"
}
case "$action" in
  build)
    podman build -t "$CPO_DPO_IMAGE" -f CodePromptOpt/finetuning/dpo/Containerfile .
    if ! podman volume exists "$CPO_DPO_CACHE"; then podman volume create "$CPO_DPO_CACHE" >/dev/null; fi
    ;;
  config)
    printf 'GPU=%s\nIMAGE=%s\nCONTAINER=%s\nCACHE=%s\nROOT=%s\nCONFIG=%s\n' \
      "$CPO_DPO_GPU" "$CPO_DPO_IMAGE" "$CPO_DPO_CONTAINER" "$CPO_DPO_CACHE" "$CPO_DPO_ROOT" "$CPO_DPO_CONFIG"
    ;;
  check) cpu python -m CodePromptOpt.finetuning.dpo.check_install ;;
  prepare)
    cpu python -m CodePromptOpt.finetuning.dpo.prepare --config "$CPO_DPO_CONFIG" --output "$CPO_DPO_ROOT/data" "$@"
    ;;
  tokens|smoke|train|generate)
    launch python -m CodePromptOpt.finetuning.dpo.pipeline "$action" \
      --config "$CPO_DPO_CONFIG" --root "$CPO_DPO_ROOT" "$@"
    ;;
  logs) podman logs -f "$CPO_DPO_CONTAINER" "$@" ;;
  status) podman inspect --format '{{.Name}}: {{.State.Status}} (exit={{.State.ExitCode}})' "$CPO_DPO_CONTAINER" ;;
  wait|await-job)
    code="$(podman wait "$CPO_DPO_CONTAINER")"
    [[ "$code" =~ ^[0-9]+$ ]] || exit 1
    if [[ "$action" == await-job ]]; then
      saved_root="$(podman inspect --format '{{ index .Config.Labels "cpo.dpo.root" }}' "$CPO_DPO_CONTAINER")"
      mkdir -p "$saved_root/logs"
      podman logs "$CPO_DPO_CONTAINER" > "$saved_root/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log" 2>&1
      if [[ "$code" == 0 ]]; then podman rm "$CPO_DPO_CONTAINER"; fi
    fi
    exit "$code"
    ;;
  stop) podman stop "$CPO_DPO_CONTAINER" ;;
  *) echo 'Usage: bash CodePromptOpt/finetuning/dpo/podman.sh {build|check|config|prepare|tokens|smoke|train|generate|logs|status|wait|await-job|stop} [arguments]' ;;
esac
