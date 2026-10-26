#!/usr/bin/env bash
# Persistent input/output selection for downstream evaluation of DPO runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/dpo/podman.env
export CPO_CODEGEN_INPUT="$CPO_DPO_ROOT/runs"
export QWEN3_CODEGEN_EXPERIMENT="$CPO_DPO_CODEGEN_EXPERIMENT"
if [[ "${1:-}" == start ]]; then
  # Match the actual SFT generator, including the immutable local image ID.
  # Parsing runs in the CPU client, so host Python is not needed.
  settings="$(podman run --rm --pull=never -v "$PWD:/workspace:z" \
    "${CPO_CLIENT_IMAGE:-localhost/cpo-codegen-client:1}" python -c \
    'import json,sys; c=json.load(open(sys.argv[1])); m=c["server_metadata"]; assert m["dtype"]=="bfloat16"; print("\n".join(str(x) for x in [c["model"],m["model_revision"],m["image_id"],m["max_model_len"],m["gpu_memory_utilization"],m["max_num_seqs"]]))' \
    "$CPO_DPO_BASELINE_CONFIG")"
  mapfile -t pinned <<< "$settings"
  [[ ${#pinned[@]} == 6 ]] || { echo 'Invalid baseline server config' >&2; exit 1; }
  podman image exists "${pinned[2]}" || { echo "Baseline vLLM image ${pinned[2]} is not local; restore the same image before comparing." >&2; exit 1; }
  export QWEN3_CODEGEN_MODEL="${pinned[0]}" QWEN3_CODEGEN_REVISION="${pinned[1]}"
  export VLLM_IMAGE="${pinned[2]}" CODEGEN_CONTEXT="${pinned[3]}"
  export CODEGEN_GPU_UTIL="${pinned[4]}" CODEGEN_MAX_SEQS="${pinned[5]}"
fi
exec bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh "$@"
