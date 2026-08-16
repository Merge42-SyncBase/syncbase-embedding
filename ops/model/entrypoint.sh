#!/bin/sh
set -eu

fetch-e5-small.sh /models

if ls /runtime/libonnxruntime.so.* >/dev/null 2>&1 || ls /runtime/libonnxruntime.dylib >/dev/null 2>&1; then
  echo "ONNX Runtime already present, skipping download"
else
  fetch-onnxruntime.sh /runtime
fi
