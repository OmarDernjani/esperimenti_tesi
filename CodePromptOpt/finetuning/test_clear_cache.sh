#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test_root="$(mktemp -d)"
trap 'rm -rf -- "$test_root"' EXIT
export CPO_TEST_VOLUME="$test_root/cache"
export CPO_TEST_ACTIVE=''
mkdir -p "$CPO_TEST_VOLUME/huggingface/hub/models--Qwen--Qwen3-4B"
mkdir -p "$CPO_TEST_VOLUME/huggingface/hub/models--microsoft--Phi-4-mini-instruct"
mkdir -p "$test_root/outputs/adapter"
printf 'adapter\n' > "$test_root/outputs/adapter/adapter_model.safetensors"
podman() {
  case "$1" in
    volume) printf '%s\n' "$CPO_TEST_VOLUME" ;;
    ps) printf '%s' "$CPO_TEST_ACTIVE" ;;
    *) return 99 ;;
  esac
}
export -f podman
script=CodePromptOpt/finetuning/clear_cache.sh
hub="$CPO_TEST_VOLUME/huggingface/hub"
bash "$script" --all
[[ -d "$hub/models--Qwen--Qwen3-4B" ]]
export CPO_TEST_ACTIVE=running-container
if bash "$script" --all --delete; then exit 1; fi
[[ -d "$hub/models--Qwen--Qwen3-4B" ]]
export CPO_TEST_ACTIVE=''
bash "$script" --model qwen3_4b --delete
[[ ! -d "$hub/models--Qwen--Qwen3-4B" ]]
[[ -d "$hub/models--microsoft--Phi-4-mini-instruct" ]]
bash "$script" --all --delete
[[ ! -d "$hub/models--microsoft--Phi-4-mini-instruct" ]]
[[ -s "$test_root/outputs/adapter/adapter_model.safetensors" ]]
echo 'Preview, active-job guard, selective deletion and adapter preservation: OK'
