#!/usr/bin/env bash
# Persistent input/output selection for downstream evaluation of DPO runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/dpo/podman.env
export CPO_CODEGEN_INPUT="$CPO_DPO_ROOT/runs"
export QWEN3_CODEGEN_EXPERIMENT="$CPO_DPO_CODEGEN_EXPERIMENT"
exec bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh "$@"
