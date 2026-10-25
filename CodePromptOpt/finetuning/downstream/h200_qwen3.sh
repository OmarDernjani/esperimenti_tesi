#!/usr/bin/env bash
# Use this entry point consistently for the Qwen3-Coder-30B experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/downstream/qwen3_coder_30b.env
export CODEGEN_MODEL="$QWEN3_CODEGEN_MODEL"
export CODEGEN_REVISION="$QWEN3_CODEGEN_REVISION"
export CODEGEN_CONTAINER="$QWEN3_CODEGEN_CONTAINER"
export CODEGEN_CACHE="$QWEN3_CODEGEN_CACHE"
export CODEGEN_PORT="$QWEN3_CODEGEN_PORT"
export CPO_CODEGEN_EXPERIMENT="$QWEN3_CODEGEN_EXPERIMENT"
exec bash CodePromptOpt/finetuning/downstream/h200.sh "$@"
