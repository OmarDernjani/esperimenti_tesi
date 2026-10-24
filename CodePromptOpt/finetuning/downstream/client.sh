#!/usr/bin/env bash
# CPU-only client. On Linux, host networking reaches a local server or SSH tunnel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
image="${CPO_CLIENT_IMAGE:-localhost/cpo-codegen-client:1}"
action="${1:-help}"
if (( $# )); then shift; fi
case "$action" in
  build) podman build -t "$image" -f CodePromptOpt/finetuning/downstream/Containerfile . ;;
  plan|run)
    flags=()
    [[ "$action" != plan ]] || flags+=(--plan-only)
    podman run --rm --network=host --log-driver=k8s-file --log-opt=max-size=50mb \
      -e CODEGEN_API_KEY -v "$PWD:/workspace:z" "$image" \
      python -m CodePromptOpt.finetuning.downstream.run "${flags[@]}" "$@"
    ;;
  *) echo 'Usage: bash CodePromptOpt/finetuning/downstream/client.sh {build|plan|run} [client options]' ;;
esac
