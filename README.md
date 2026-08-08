# SyncBase vector embedding

고정된 multilingual-E5 ONNX 추론, tokenizer, artifact checksum, 동시성 제한과 모델 다운로드 도구를 소유한다. WAS 도메인 타입에 의존하지 않으며 최소 `Profile`만 입력받는다.

```sh
go test ./...
ops/model/fetch-e5-small.sh /absolute/model-dir
ops/model/fetch-onnxruntime.sh /absolute/runtime-dir linux-amd64
```
