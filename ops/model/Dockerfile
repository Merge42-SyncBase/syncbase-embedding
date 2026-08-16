# syntax=docker/dockerfile:1.7
# One-shot init container: downloads the pinned E5 model + ONNX Runtime
# shared library into mounted /models and /runtime volumes. Idempotent —
# safe to run on every `docker compose up`.
ARG RUNTIME_IMAGE=debian:bookworm-slim@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241

FROM ${RUNTIME_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates perl \
    && rm -rf /var/lib/apt/lists/*
COPY ops/model/fetch-e5-small.sh ops/model/fetch-onnxruntime.sh ops/model/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/fetch-e5-small.sh /usr/local/bin/fetch-onnxruntime.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
