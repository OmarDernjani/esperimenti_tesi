#!/usr/bin/env bash
# Configuration only: no Podman, GPU, downloads or generated-code execution.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CODEGEN_MODEL=old-model CODEGEN_REVISION=old-commit CODEGEN_PORT=8000
export CODEGEN_CONTAINER=old-container CODEGEN_CACHE=old-cache CPO_CODEGEN_EXPERIMENT=old-experiment
export CPO_GPU=3
config="$(bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh config)"
[[ "$config" == *'MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct'* ]]
[[ "$config" == *'GPU=3'* && "$config" == *'API=http://127.0.0.1:8001/v1'* ]]
[[ "$config" == *'qwen3-coder-30b-h200-v1/server-config.json'* ]]
# A child wrapper must not change the invoking shell's original profile.
[[ "$CODEGEN_MODEL" == old-model && "$CODEGEN_CONTAINER" == old-container ]]
echo 'Separate model, experiment, port and parent environment: OK'
