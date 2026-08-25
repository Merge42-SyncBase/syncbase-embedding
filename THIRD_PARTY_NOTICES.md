# Third-party notices

SyncBase vector embedding is licensed under Apache-2.0. The model, tokenizer,
runtime, and libraries below are not relicensed by SyncBase.

| Component | Pinned identity | Use | Upstream license | Source |
| --- | --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | Hugging Face revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; `onnx/model.onnx` and `onnx/tokenizer.json` are separately SHA-256 pinned by `ops/model/fetch-e5-small.sh` | Local 384-dimensional query/passage embeddings | MIT, as declared by the upstream model card/API | <https://huggingface.co/intfloat/multilingual-e5-small> |
| Microsoft ONNX Runtime | 1.26.0 release archives and shared libraries are platform-specific SHA-256 pins in `ops/model/fetch-onnxruntime.sh` | Native local ONNX inference runtime | MIT; copyright Microsoft Corporation | <https://github.com/microsoft/onnxruntime/tree/v1.26.0> |
| `github.com/yalue/onnxruntime_go` | v1.31.0 | Go binding for ONNX Runtime | MIT; copyright 2023 Nathan Otterness | <https://github.com/yalue/onnxruntime_go/tree/v1.31.0> |
| `github.com/sugarme/tokenizer` | v0.3.0 | Local tokenizer implementation | Apache-2.0; copyright 2020 Thang Tran | <https://github.com/sugarme/tokenizer/tree/v0.3.0> |

The E5 and ONNX Runtime fetchers download upstream artifacts directly into
mounted runtime volumes; the artifacts are not stored in this Git repository.
Their license identities, upstream revisions, and artifact hashes must also be
present in the final AI-model form and release SBOM.

`go.mod` and `go.sum` are the authoritative Go dependency pins. This file
highlights the runtime/model supply chain; the final CycloneDX SBOM must cover
all resolved transitive modules.
