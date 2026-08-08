package embedding

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestNewRejectsUnpinnedRuntimeLibrary(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	modelPath := directory + "/model.onnx"
	tokenizerPath := directory + "/tokenizer.json"
	runtimePath := directory + "/libonnxruntime.so"
	for path, content := range map[string][]byte{
		modelPath: []byte("model"), tokenizerPath: []byte("tokenizer"), runtimePath: []byte("runtime"),
	} {
		if err := os.WriteFile(path, content, 0o600); err != nil {
			t.Fatalf("WriteFile(%q): %v", path, err)
		}
	}
	_, err := New(Config{
		ModelPath: modelPath, ModelSHA256: fileSHA256(t, modelPath),
		TokenizerPath: tokenizerPath, TokenizerSHA256: fileSHA256(t, tokenizerPath),
		RuntimeLibraryPath: runtimePath, RuntimeSHA256: strings.Repeat("0", 64),
	})
	if err == nil || !strings.Contains(err.Error(), "verify ONNX Runtime library") {
		t.Fatalf("New error = %v, want runtime verification failure", err)
	}
}

func TestAcquireEmbeddingSlotHonorsCancellationAndBudget(t *testing.T) {
	semaphore := make(chan struct{}, 1)
	semaphore <- struct{}{}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := acquireEmbeddingSlot(canceled, semaphore); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled acquisition error = %v, want context.Canceled", err)
	}
	started := time.Now()
	if err := acquireEmbeddingSlot(context.Background(), semaphore); !errors.Is(err, ErrTemporarilyUnavailable) {
		t.Fatalf("budget acquisition error = %v, want ErrTemporarilyUnavailable", err)
	}
	if elapsed := time.Since(started); elapsed < 90*time.Millisecond || elapsed > 500*time.Millisecond {
		t.Fatalf("slot wait = %s, want approximately 100ms", elapsed)
	}
}

func TestPinnedE5ArtifactsProduceDeterministicSemanticVectors(t *testing.T) {
	model := os.Getenv("SYNCBASE_TEST_E5_MODEL_PATH")
	tokenizer := os.Getenv("SYNCBASE_TEST_E5_TOKENIZER_PATH")
	runtimeLibrary := os.Getenv("SYNCBASE_TEST_ORT_LIBRARY_PATH")
	if model == "" || tokenizer == "" || runtimeLibrary == "" {
		if os.Getenv("SYNCBASE_REQUIRE_E5_TEST") == "true" {
			t.Fatal("pinned E5 test artifacts are required but not configured")
		}
		t.Skip("pinned E5 test artifacts are not configured")
	}
	profile := Profile{
		EmbeddingModelID: "intfloat/multilingual-e5-small",
		VectorDimension:  VectorDimension,
		Distance:         "cosine",
	}
	service, err := New(Config{
		ModelPath:          model,
		ModelSHA256:        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
		TokenizerPath:      tokenizer,
		TokenizerSHA256:    "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
		RuntimeLibraryPath: runtimeLibrary,
		RuntimeSHA256:      fileSHA256(t, runtimeLibrary),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})

	query, err := service.EmbedQuery(context.Background(), "정보보안 정책에서 비밀번호 보호 근거", profile)
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	repeated, err := service.EmbedQuery(context.Background(), "정보보안 정책에서 비밀번호 보호 근거", profile)
	if err != nil {
		t.Fatalf("EmbedQuery repeated: %v", err)
	}
	passages, err := service.EmbedPassages(context.Background(), []string{
		"정보보안 정책은 비밀번호와 접근 토큰을 안전하게 보호해야 한다.",
		"사내 카페의 여름 음료 운영 시간 안내",
	}, profile)
	if err != nil {
		t.Fatalf("EmbedPassages: %v", err)
	}
	if len(query) != VectorDimension || len(passages) != 2 {
		t.Fatalf("dimensions: query=%d passages=%d", len(query), len(passages))
	}
	for index := range query {
		if query[index] != repeated[index] {
			t.Fatalf("query is nondeterministic at dimension %d", index)
		}
	}
	if math.Abs(norm(query)-1) > 1e-5 || math.Abs(norm(passages[0])-1) > 1e-5 {
		t.Fatalf("vectors are not normalized: query=%f passage=%f", norm(query), norm(passages[0]))
	}
	if cosineDistance(query, passages[0]) >= cosineDistance(query, passages[1]) {
		t.Fatalf("relevant passage did not rank first: relevant=%f unrelated=%f",
			cosineDistance(query, passages[0]), cosineDistance(query, passages[1]))
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := service.Ready(context.Background()); err == nil {
		t.Fatal("Ready after Close succeeded, want error")
	}
	if _, err := service.CountTokens("closed session"); err == nil {
		t.Fatal("CountTokens after Close succeeded, want error")
	}
	if _, err := service.EmbedQuery(context.Background(), "closed session", profile); err == nil {
		t.Fatal("EmbedQuery after Close succeeded, want error")
	}
}

func norm(values []float32) float64 {
	var squared float64
	for _, value := range values {
		squared += float64(value) * float64(value)
	}
	return math.Sqrt(squared)
}

func cosineDistance(left, right []float32) float64 {
	var dot float64
	for index := range left {
		dot += float64(left[index]) * float64(right[index])
	}
	return 1 - dot
}

func fileSHA256(t *testing.T, path string) string {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q): %v", path, err)
	}
	digest := sha256.Sum256(content)
	return hex.EncodeToString(digest[:])
}
