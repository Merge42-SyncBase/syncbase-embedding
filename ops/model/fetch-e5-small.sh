#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model-directory>" >&2
  exit 64
fi

model_dir="$1"
model_sha="ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
tokenizer_sha="0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
model_url="https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx"
tokenizer_url="https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/tokenizer.json"

mkdir -p "$model_dir"
hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [[ -f "$model_dir/model.onnx" && -f "$model_dir/tokenizer.json" ]] \
  && [[ "$(hash_file "$model_dir/model.onnx")" == "$model_sha" ]] \
  && [[ "$(hash_file "$model_dir/tokenizer.json")" == "$tokenizer_sha" ]]; then
  printf 'model_path=%s\nmodel_sha256=%s\ntokenizer_path=%s\ntokenizer_sha256=%s\n' \
    "$model_dir/model.onnx" "$model_sha" "$model_dir/tokenizer.json" "$tokenizer_sha"
  exit 0
fi

temporary_dir="$(mktemp -d "${model_dir%/}/.download.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

curl --fail --location --retry 3 --output "$temporary_dir/model.onnx" "$model_url"
curl --fail --location --retry 3 --output "$temporary_dir/tokenizer.json" "$tokenizer_url"

[[ "$(hash_file "$temporary_dir/model.onnx")" == "$model_sha" ]] || {
  echo "model SHA-256 mismatch" >&2
  exit 1
}
[[ "$(hash_file "$temporary_dir/tokenizer.json")" == "$tokenizer_sha" ]] || {
  echo "tokenizer SHA-256 mismatch" >&2
  exit 1
}

mv "$temporary_dir/model.onnx" "$model_dir/model.onnx"
mv "$temporary_dir/tokenizer.json" "$model_dir/tokenizer.json"
printf 'model_path=%s\nmodel_sha256=%s\ntokenizer_path=%s\ntokenizer_sha256=%s\n' \
  "$model_dir/model.onnx" "$model_sha" "$model_dir/tokenizer.json" "$tokenizer_sha"
