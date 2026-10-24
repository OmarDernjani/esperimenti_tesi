#!/usr/bin/env bash
# Host orchestrates sandbox containers; Python runs in the lightweight client image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
if (( $# != 2 )); then
  echo 'Usage: bash evaluate_suite.sh CODE_ROOT VERIFIED_TESTS_JSONL (paths relative to repository)' >&2; exit 2
fi
root="$1"
tests="$2"
[[ -d "$root" && -f "$tests" ]] || { echo 'Missing code root or verified tests' >&2; exit 2; }
export CPO_IMAGE="${CPO_CLIENT_IMAGE:-localhost/cpo-codegen-client:1}"
if [[ -z "${CPO_TEST_IMAGE:-}" ]]; then
  export CPO_TEST_IMAGE=localhost/cpo-benchmark-tests:1
  if ! podman image exists "$CPO_TEST_IMAGE"; then
    podman build -t "$CPO_TEST_IMAGE" -f CodePromptOpt/finetuning/downstream/Test.Containerfile .
  fi
fi
# Catch missing benchmark dependencies before turning them into model failures.
podman run --rm --network=none --entrypoint python "$CPO_TEST_IMAGE" -c 'import numpy; print("Benchmark NumPy runtime OK")'
found=0
while IFS= read -r -d '' sample; do
  found=1
  output="${sample%-code.jsonl}-results.jsonl"
  jobs="${sample%-code.jsonl}-jobs"
  if [[ -e "$output" || -e "$jobs" ]]; then
    echo "Existing evaluation: $sample; use a fresh directory to rerun." >&2; exit 1
  fi
  bash CodePromptOpt/finetuning/podman.sh evaluate "$sample" "$tests" "$output" "$jobs"
done < <(find "$root" -type f -name '*-code.jsonl' -print0)
[[ "$found" == 1 ]] || { echo 'No code files found' >&2; exit 1; }
echo 'Functional evaluations completed. Use finetuning.compare on paired *-results.jsonl files.'
