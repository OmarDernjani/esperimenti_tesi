#!/usr/bin/env bash
# Test command construction without requiring Podman or a GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CPO_GPU=3
export CPO_CONTAINER_NAME=cpo-gpu-3
export CPO_MOCK=launch
podman() {
  if [[ "$1 $2" == 'container exists' ]]; then
    [[ "$CPO_MOCK" == busy ]]
    return
  fi
  case "$1" in
    run)
      [[ "$2" == -d && "$*" == *'--name cpo-gpu-3'* && "$*" != *'--rm'* ]] || return 90
      [[ "$*" == *'--pull=never'* ]] || return 92
      printf 'mock-container-id\n'
      ;;
    inspect) printf 'true\n' ;;
    wait) printf '7\n' ;;
    *) return 91 ;;
  esac
}
export -f podman
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b
export CPO_MOCK=busy
if bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b; then
  echo 'Expected an active container to block a second launch' >&2
  exit 1
fi
code=0
bash CodePromptOpt/finetuning/podman.sh wait || code=$?
[[ "$code" == 7 ]]
echo 'Detached launch, active-container guard and exit-code propagation: OK'
