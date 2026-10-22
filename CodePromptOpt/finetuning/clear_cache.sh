#!/usr/bin/env bash
# Host-side cleanup: works even when storage is too full to launch a container.
# Only model directories in the named HF cache volume are eligible for deletion.
set -euo pipefail
volume="${CPO_CACHE:-cpo-hf-cache}"
model=''
all=0
delete=0
usage() {
  echo 'Usage: bash CodePromptOpt/finetuning/clear_cache.sh (--all | --model MODEL) [--delete]'
  echo 'Default: preview only. MODEL: qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b, phi4_mini, or OWNER/NAME.'
}
while (( $# )); do
  case "$1" in
    --all) all=1; shift ;;
    --model) [[ $# -ge 2 ]] || { usage; exit 2; }; model="$2"; shift 2 ;;
    --delete) delete=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
if [[ "$all" == 1 && -n "$model" ]] || [[ "$all" == 0 && -z "$model" ]]; then
  usage; exit 2
fi
case "$model" in
  qwen3_4b) model='Qwen/Qwen3-4B' ;;
  qwen3_8b) model='Qwen/Qwen3-8B' ;;
  qwen3_14b) model='Qwen/Qwen3-14B' ;;
  qwen3_32b) model='Qwen/Qwen3-32B' ;;
  phi4_mini) model='microsoft/Phi-4-mini-instruct' ;;
esac
if [[ -n "$model" && ! "$model" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
  echo 'Invalid Hugging Face model ID' >&2; exit 2
fi
mountpoint="$(podman volume inspect "$volume" --format '{{.Mountpoint}}')"
[[ -n "$mountpoint" && -d "$mountpoint" ]] || { echo 'Cache volume mountpoint not accessible' >&2; exit 1; }
mountpoint="$(realpath -e -- "$mountpoint")"
hub="$mountpoint/huggingface/hub"
if [[ ! -e "$hub" ]]; then
  echo 'Hugging Face cache already empty.'; exit 0
fi
[[ -d "$hub" && ! -L "$hub" && ! -L "$mountpoint/huggingface" ]] || {
  echo 'Refusing unexpected/symlinked cache layout' >&2; exit 1;
}
hub="$(realpath -e -- "$hub")"
[[ "$hub" == "$mountpoint/huggingface/hub" ]] || exit 1
targets=()
if [[ "$all" == 1 ]]; then
  shopt -s nullglob
  targets=("$hub"/models--*)
else
  target="$hub/models--${model/\//--}"
  [[ ! -e "$target" && ! -L "$target" ]] || targets=("$target")
fi
if (( ${#targets[@]} == 0 )); then
  echo 'No matching cached models.'; exit 0
fi
for target in "${targets[@]}"; do
  [[ -d "$target" && ! -L "$target" ]] || { echo "Unexpected cache entry: $target" >&2; exit 1; }
  resolved="$(realpath -e -- "$target")"
  [[ "$resolved" == "$target" && "$(dirname "$resolved")" == "$hub" ]] || exit 1
done
echo "Base-model cache in Podman volume: $volume"
du -sh -- "${targets[@]}"
if [[ "$delete" == 0 ]]; then
  echo 'Preview only. Add --delete to remove these model caches.'; exit 0
fi
# A running container might be training, generating or downloading. Never clear
# a shared model cache while any container has the volume mounted.
active="$(podman ps -q --filter "volume=$volume")"
if [[ -n "$active" ]]; then
  echo "Refusing cleanup: running containers use volume $volume:" >&2
  echo "$active" >&2
  exit 1
fi
rm -rf --one-file-system -- "${targets[@]}"
echo 'Base-model caches removed. Fine-tuned adapters and outputs were not touched.'
df -h -- "$mountpoint"
