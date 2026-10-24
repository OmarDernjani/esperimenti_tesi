#!/usr/bin/env bash
# Exercise the H200 start command without any real Podman/GPU/network requests.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CPO_GPU=3 CODEGEN_GPU=3 CODEGEN_CONTAINER=cpo-vllm-h200
export VLLM_IMAGE=example/image:test
export CPO_MOCK_TRAINING=false
podman() {
  case "$1 $2" in
    'container exists') [[ "$CPO_MOCK_TRAINING" == true && "$3" == cpo-gpu-3 ]]; return ;;
    'image exists') return 0 ;;
    'image inspect')
      if [[ "$*" == *'{{.Id}}'* && "$*" != *Architecture* ]]; then
        printf 'sha256:mock-image\n'
      else
        printf 'amd64\n'
      fi
      ;;
    'volume create') printf 'mock-cache\n' ;;
    'inspect --format') printf 'true\n' ;;
    'run --rm')
      if [[ "$*" == *HfApi* ]]; then
        printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n'
      else
        [[ "$*" == *configure_h200* && "$*" == *'--revision aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'* ]] || return 92
      fi
      ;;
    'run -d')
      [[ "$*" == *'nvidia.com/gpu=3'* && "$*" == *'127.0.0.1:8000:8000'* && "$*" == *'max-size=50mb'* ]] || return 93
      [[ "$*" == *'sha256:mock-image'* && "$*" == *'--revision aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'* ]] || return 94
      printf 'mock-server-container-id\n'
      ;;
    'pull example/image:test') return 0 ;;
    *) echo "Unexpected mocked Podman call: $*" >&2; return 99 ;;
  esac
}
export -f podman
bash CodePromptOpt/finetuning/downstream/h200.sh start
export CPO_MOCK_TRAINING=true
if bash CodePromptOpt/finetuning/downstream/h200.sh start; then
  echo 'Expected the active optimizer/training to block serving' >&2; exit 1
fi
echo 'H200 device, localhost port, bounded logs, exact image/revision and active-training guard: OK'
