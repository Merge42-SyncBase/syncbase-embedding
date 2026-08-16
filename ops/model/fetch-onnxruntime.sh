#!/usr/bin/env bash
set -euo pipefail

version="1.26.0"
output_dir="${1:?usage: fetch-onnxruntime.sh OUTPUT_DIR [darwin-arm64|linux-amd64|linux-arm64]}"
platform="${2:-}"
if [[ -z "$platform" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) platform="darwin-arm64" ;;
    Linux-x86_64) platform="linux-amd64" ;;
    Linux-aarch64|Linux-arm64) platform="linux-arm64" ;;
    *) echo "unsupported ONNX Runtime platform" >&2; exit 2 ;;
  esac
fi

case "$platform" in
  darwin-arm64)
    archive="onnxruntime-osx-arm64-${version}.tgz"
    expected="7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3"
    expected_library="cb0462c3fd35ad722e8772313030a33c182f3d4c6b33f4e5e1fcb2ce3199b86c"
    library="libonnxruntime.dylib"
    ;;
  linux-amd64)
    archive="onnxruntime-linux-x64-${version}.tgz"
    expected="1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57"
    expected_library="5bd5bedf736fc501692435d0ec4f6e8b2bdf48cd30af8e6d00d61b3ddc9a7ab8"
    library="libonnxruntime.so.${version}"
    ;;
  linux-arm64)
    archive="onnxruntime-linux-aarch64-${version}.tgz"
    expected="34ff1c2d0f12e2cf3d33a0c5f82e39792e1d581fbd6968fd7c30d173654be01a"
    expected_library="115ecb838e703d390262b8b4d07d5248e6693c67658d4c98c48f94905ab27af4"
    library="libonnxruntime.so.${version}"
    ;;
  *) echo "unsupported ONNX Runtime platform: $platform" >&2; exit 2 ;;
esac

temporary="$(mktemp -d "${TMPDIR:-/tmp}/syncbase-ort.XXXXXX")"
trap 'rm -rf "$temporary"' EXIT
url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}"
curl --fail --location --silent --show-error --retry 3 --retry-connrefused "$url" --output "$temporary/$archive"
actual="$(shasum -a 256 "$temporary/$archive" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "ONNX Runtime archive SHA-256 mismatch" >&2
  exit 1
fi
tar -xzf "$temporary/$archive" -C "$temporary"
source_path="$(find "$temporary" -type f -name "$library" -print -quit)"
if [[ -z "$source_path" ]]; then
  echo "ONNX Runtime shared library was not found in the archive" >&2
  exit 1
fi
library_actual="$(shasum -a 256 "$source_path" | awk '{print $1}')"
if [[ "$library_actual" != "$expected_library" ]]; then
  echo "ONNX Runtime library SHA-256 mismatch" >&2
  exit 1
fi
mkdir -p "$output_dir"
target="$output_dir/$library"
if [[ -e "$target" ]]; then
  echo "refusing to overwrite existing runtime: $target" >&2
  exit 1
fi
install -m 0555 "$source_path" "$target"
printf 'runtime_path=%s\narchive_sha256=%s\nruntime_sha256=%s\n' "$target" "$actual" "$library_actual"
