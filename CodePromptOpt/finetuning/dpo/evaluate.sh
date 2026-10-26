#!/usr/bin/env bash
# CPU evaluation and paired comparison. Code generation uses codegen.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source CodePromptOpt/finetuning/podman.env
source CodePromptOpt/finetuning/dpo/podman.env
source CodePromptOpt/finetuning/downstream/h200.env
action="${1:-help}"
if (( $# )); then shift; fi
root="$CPO_CODEGEN_OUTPUT/$CPO_DPO_CODEGEN_EXPERIMENT/validation"
case "$action" in
  sandbox)
    bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh "$root" "${1:-verified_tests.jsonl}"
    ;;
  report)
    CPO_IMAGE="$CPO_CLIENT_IMAGE" bash CodePromptOpt/finetuning/podman.sh cpu \
      python -m CodePromptOpt.finetuning.dpo.compare_results --dpo-root "$root" "$@"
    ;;
  *) echo 'Usage: bash CodePromptOpt/finetuning/dpo/evaluate.sh {sandbox [TESTS]|report [--seeds 42 43 44]}' ;;
esac
