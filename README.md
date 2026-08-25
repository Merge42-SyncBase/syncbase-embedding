# SyncBase vector embedding

고정된 multilingual-E5 ONNX 추론, tokenizer, artifact checksum, 동시성 제한과 모델 다운로드 도구를 소유한다. WAS 도메인 타입에 의존하지 않으며 최소 `Profile`과 `Provider` 계약만 공개한다. P0 구현체는 `E5` 하나이며, Worker는 passage embedding, MCP는 query embedding에 같은 immutable profile을 사용한다.

```sh
go test ./...
ops/model/fetch-e5-small.sh /absolute/model-dir
ops/model/fetch-onnxruntime.sh /absolute/runtime-dir linux-amd64
```

## License

SyncBase vector embedding의 자체 소스는 [Apache License 2.0](LICENSE)
(`Apache-2.0`)으로 배포합니다. `multilingual-e5-small`, ONNX Runtime,
tokenizer 구현체 등 외부
구성요소는 각자의 라이선스를 따르며 세부 출처는
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록합니다.
