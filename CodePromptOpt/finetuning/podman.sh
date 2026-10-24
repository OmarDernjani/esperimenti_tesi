#!/usr/bin/env bash
# Invoke with bash; Python and CUDA dependencies stay inside containers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMAGE="${CPO_IMAGE:-localhost/codepromptopt-ft:h200}"
# Host GPU: defaults to 3, so no external export is needed.
export CPO_GPU="${CPO_GPU:-3}"
GPU="$CPO_GPU"
CONTAINER="${CPO_CONTAINER_NAME:-cpo-gpu-${GPU//[^a-zA-Z0-9_.-]/-}}"
CACHE="${CPO_CACHE:-cpo-hf-cache}"
network_args=()
if [[ -n "${CPO_NETWORK:-}" ]]; then network_args=(--network "$CPO_NETWORK"); fi
action="${1:-help}"
if (( $# )); then shift; fi
run_cpu() {
  podman run --rm --pull=never "${network_args[@]}" --shm-size=8g -e PYTHONUNBUFFERED=1 \
    -e HF_TOKEN -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" -v "$CACHE:/cache" "$IMAGE" "$@"
}
run_gpu() {
  podman run --rm --pull=never "${network_args[@]}" --device "nvidia.com/gpu=$GPU" -e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTHONUNBUFFERED=1 --shm-size=8g \
    -e HF_TOKEN -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" -v "$CACHE:/cache" "$IMAGE" "$@"
}
run_gpu_detached() {
  if podman container exists "$CONTAINER"; then
    if [[ "$(podman inspect --format '{{.State.Running}}' "$CONTAINER")" == true ]]; then
      echo "Container $CONTAINER is already running. Wait for it before launching another GPU stage." >&2
      exit 1
    fi
    # Keep completed containers and their logs rather than removing them with --rm.
    archived="${CONTAINER}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    podman rename "$CONTAINER" "$archived"
    echo "Previous container retained as $archived"
  fi
  podman run -d --pull=never --name "$CONTAINER" "${network_args[@]}" \
    --device "nvidia.com/gpu=$GPU" -e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTHONUNBUFFERED=1 --shm-size=8g \
    -e HF_TOKEN -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" -v "$CACHE:/cache" "$IMAGE" "$@"
  echo "Container: $CONTAINER (detached; retained after exit)"
  echo "Logs: podman logs -f $CONTAINER"
  echo "Wait and check exit code: bash CodePromptOpt/finetuning/podman.sh wait"
}
case "$action" in
  build)
    podman build -t "$IMAGE" -f CodePromptOpt/finetuning/Containerfile .
    podman volume create "$CACHE" >/dev/null
    ;;
  doctor)
    run_gpu python -m CodePromptOpt.finetuning.doctor
    ;;
  plan) run_cpu python -m CodePromptOpt.finetuning.suite --stage plan "$@" ;;
  prepare)
    run_gpu python -m CodePromptOpt.finetuning.suite --stage "$action" "$@"
    ;;
  smoke|train|generate)
    run_gpu_detached python -m CodePromptOpt.finetuning.suite --stage "$action" "$@"
    ;;
  cpu) run_cpu "$@" ;;
  gpu) run_gpu_detached "$@" ;;
  logs) podman logs -f "$CONTAINER" "$@" ;;
  status) podman inspect --format '{{.Name}}: {{.State.Status}} (exit={{.State.ExitCode}})' "$CONTAINER" ;;
  wait)
    code="$(podman wait "$CONTAINER")"
    echo "$CONTAINER exited with code $code"
    [[ "$code" =~ ^[0-9]+$ ]] || exit 1
    exit "$code"
    ;;
  stop) podman stop "$CONTAINER" "$@" ;;
  evaluate)
    # Each test gets its own container; never expose the Podman socket to a workload.
    # Arguments: samples.jsonl tests.jsonl output.jsonl jobs-directory
    if (( $# != 4 )); then
      echo 'Usage: bash podman.sh evaluate SAMPLES TESTS OUTPUT JOBS_DIR' >&2; exit 2
    fi
    command -v timeout >/dev/null || { echo 'GNU timeout is required on the host' >&2; exit 2; }
    samples="$1"; tests="$2"; output="$3"; jobs="$4"
    TEST_IMAGE="${CPO_TEST_IMAGE:-docker.io/library/python:3.11-slim}"
    podman pull "$TEST_IMAGE"
    TEST_IMAGE_ID="$(podman image inspect --format '{{.Id}}' "$TEST_IMAGE")"
    run_cpu python -m CodePromptOpt.finetuning.container_evaluate prepare \
      --samples "$samples" --tests "$tests" --jobs "$jobs" --image "$TEST_IMAGE_ID"
    jobs_abs="$(realpath "$jobs")"
    while IFS= read -r job; do
      [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid job identifier' >&2; exit 2; }
      name="cpo-eval-$$-$job"
      rc=0
      timeout --kill-after=5s 20s podman run --rm --name "$name" --network=none --read-only \
        --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=64 --memory=512m --cpus=1 \
        --user=65534:65534 --tmpfs=/tmp:rw,noexec,nosuid,size=64m \
        -e PYTHONDONTWRITEBYTECODE=1 -v "$jobs_abs/$job:/work:ro,z" -w /work \
        "$TEST_IMAGE_ID" python -B check.py >/dev/null 2>&1 || rc=$?
      podman rm -f "$name" >/dev/null 2>&1 || true
      if [[ "$rc" == 125 || "$rc" == 126 || "$rc" == 127 ]]; then
        echo "Podman infrastructure failure for job $job (exit $rc)" >&2; exit 1
      fi
      printf '%s\n' "$rc" > "$jobs_abs/$job/exit_code"
    done < "$jobs_abs/jobs.txt"
    run_cpu python -m CodePromptOpt.finetuning.container_evaluate collect --jobs "$jobs" --output "$output"
    ;;
  *)
    echo 'Usage: bash CodePromptOpt/finetuning/podman.sh {build|doctor|plan|prepare|smoke|train|generate|cpu|gpu|evaluate|logs|status|wait|stop} [arguments]'
    ;;
esac
