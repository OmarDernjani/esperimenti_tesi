#!/usr/bin/env bash
# Invoke with bash; Python and CUDA dependencies stay inside containers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMAGE="${CPO_IMAGE:-localhost/codepromptopt-ft:h200}"
# Host GPU: defaults to 3, so no external export is needed.
export CPO_GPU="${CPO_GPU:-3}"
GPU="$CPO_GPU"
CACHE="${CPO_CACHE:-cpo-hf-cache}"
network_args=()
if [[ -n "${CPO_NETWORK:-}" ]]; then network_args=(--network "$CPO_NETWORK"); fi
action="${1:-help}"
if (( $# )); then shift; fi
run_cpu() {
  podman run --rm "${network_args[@]}" --shm-size=8g -e PYTHONUNBUFFERED=1 \
    -e HF_TOKEN -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" -v "$CACHE:/cache" "$IMAGE" "$@"
}
run_gpu() {
  podman run --rm "${network_args[@]}" --device "nvidia.com/gpu=$GPU" -e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTHONUNBUFFERED=1 --shm-size=8g \
    -e HF_TOKEN -e CODEGEN_API_KEY \
    -v "$PWD:/workspace:z" -v "$CACHE:/cache" "$IMAGE" "$@"
}
case "$action" in
  build)
    podman build -t "$IMAGE" -f CodePromptOpt/finetuning/Containerfile .
    podman volume create "$CACHE" >/dev/null
    ;;
  doctor)
    run_gpu python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0)); print("GiB:", torch.cuda.get_device_properties(0).total_memory/2**30)'
    ;;
  plan) run_cpu python -m CodePromptOpt.finetuning.suite --stage plan "$@" ;;
  prepare|smoke|train|generate)
    run_gpu python -m CodePromptOpt.finetuning.suite --stage "$action" "$@"
    ;;
  cpu) run_cpu "$@" ;;
  gpu) run_gpu "$@" ;;
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
    echo 'Usage: bash CodePromptOpt/finetuning/podman.sh {build|doctor|plan|prepare|smoke|train|generate|cpu|gpu|evaluate} [arguments]'
    ;;
esac
