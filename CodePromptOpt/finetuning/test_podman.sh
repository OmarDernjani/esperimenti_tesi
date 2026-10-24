#!/usr/bin/env bash
# Test command construction without requiring Podman or a GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CPO_GPU=3
export CPO_CONTAINER_NAME=cpo-gpu-3
export CPO_MOCK=launch
export CPO_MOCK_CODE=7
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
    wait) printf '%s\n' "$CPO_MOCK_CODE" ;;
    logs) printf 'mock training log\n' ;;
    rm) printf 'mock successful container removal\n' ;;
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
logfile="CodePromptOpt/finetuning/outputs/test-await-job-$$.log"
trap 'rm -f -- "$logfile"' EXIT
code=0
result="$(bash CodePromptOpt/finetuning/podman.sh await-job "$logfile" 2>&1)" || code=$?
[[ "$code" == 7 && "$result" != *'mock successful container removal'* ]]
[[ "$(cat "$logfile")" == 'mock training log' ]]
export CPO_MOCK_CODE=0
result="$(bash CodePromptOpt/finetuning/podman.sh await-job "$logfile")"
[[ "$result" == *'mock successful container removal'* ]]
config="$(bash CodePromptOpt/finetuning/podman.sh config)"
[[ "$config" == *'GPU=3'* && "$config" == *'OUTPUT='* ]]
echo 'Detached launch, busy guard, persistent config, logs, success cleanup and failure retention: OK'
