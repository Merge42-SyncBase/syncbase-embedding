// Package embedding adapts the pinned multilingual-E5 ONNX model to SyncBase.
package embedding

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	maxTokens       = 512
	batchSize       = 8
	// VectorDimension is the only vector size supported by the pinned E5 model.
	VectorDimension = 384
)

var (
	// ErrInvalidArgument reports invalid text or artifact configuration.
	ErrInvalidArgument = errors.New("invalid argument")
	// ErrProfileMismatch reports an incompatible model or vector contract.
	ErrProfileMismatch = errors.New("embedding profile mismatch")
	// ErrTemporarilyUnavailable reports bounded local inference saturation.
	ErrTemporarilyUnavailable = errors.New("embedding temporarily unavailable")
)

// Profile is the minimal processing contract required by the E5 engine.
type Profile struct {
	EmbeddingModelID string
	VectorDimension  int
	Distance         string
}

// Config identifies the pinned model, tokenizer, and ONNX Runtime artifacts.
type Config struct {
	ModelPath          string
	ModelSHA256        string
	TokenizerPath      string
	TokenizerSHA256    string
	RuntimeLibraryPath string
	RuntimeSHA256      string
}

// E5 owns one ONNX Runtime session that creates deterministic query and passage
// vectors. Worker and MCP processes use separate instances.
type E5 struct {
	profileModelID string
	tokenizer      *tokenizer.Tokenizer
	session        *ort.DynamicAdvancedSession
	inputNames     []string
	outputName     string
	semaphore      chan struct{}
	mu             sync.RWMutex
	closed         bool
}

// New verifies pinned artifacts and initializes the bounded ONNX runtime.
func New(config Config) (*E5, error) {
	if err := verifyArtifact(config.ModelPath, config.ModelSHA256); err != nil {
		return nil, fmt.Errorf("verify E5 model: %w", err)
	}
	if err := verifyArtifact(config.TokenizerPath, config.TokenizerSHA256); err != nil {
		return nil, fmt.Errorf("verify E5 tokenizer: %w", err)
	}
	if err := verifyArtifact(config.RuntimeLibraryPath, config.RuntimeSHA256); err != nil {
		return nil, fmt.Errorf("verify ONNX Runtime library: %w", err)
	}
	ort.SetSharedLibraryPath(config.RuntimeLibraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	initialized := true
	defer func() {
		if initialized {
			_ = ort.DestroyEnvironment()
		}
	}()

	inputs, outputs, err := ort.GetInputOutputInfo(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect E5 model contract: %w", err)
	}
	inputNames := make([]string, 0, len(inputs))
	hasIDs := false
	hasMask := false
	for _, input := range inputs {
		switch input.Name {
		case "input_ids":
			hasIDs = true
		case "attention_mask":
			hasMask = true
		case "token_type_ids":
		default:
			return nil, fmt.Errorf("unsupported E5 input %q: %w", input.Name, ErrProfileMismatch)
		}
		inputNames = append(inputNames, input.Name)
	}
	if !hasIDs || !hasMask || len(outputs) < 1 {
		return nil, fmt.Errorf("E5 model inputs or output are missing: %w", ErrProfileMismatch)
	}
	output := outputs[0]
	if len(output.Dimensions) != 3 || output.Dimensions[2] != VectorDimension {
		return nil, fmt.Errorf("E5 output dimensions %v: %w", output.Dimensions, ErrProfileMismatch)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create ONNX session options: %w", err)
	}
	defer options.Destroy()
	if err := options.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set ONNX inter-op threads: %w", err)
	}
	if err := options.SetIntraOpNumThreads(max(1, runtime.NumCPU()/2)); err != nil {
		return nil, fmt.Errorf("set ONNX intra-op threads: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(
		config.ModelPath, inputNames, []string{output.Name}, options,
	)
	if err != nil {
		return nil, fmt.Errorf("load E5 ONNX model: %w", err)
	}
	tk, err := pretrained.FromFile(config.TokenizerPath)
	if err != nil {
		_ = session.Destroy()
		return nil, fmt.Errorf("load E5 tokenizer: %w", err)
	}
	padToken := "<pad>"
	padID, found := tk.TokenToId(padToken)
	if !found {
		padToken = "[PAD]"
		padID, found = tk.TokenToId(padToken)
	}
	if !found {
		_ = session.Destroy()
		return nil, fmt.Errorf("E5 tokenizer has no pad token: %w", ErrProfileMismatch)
	}
	tk.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxTokens,
		Strategy:  tokenizer.LongestFirst,
	})
	tk.WithPadding(&tokenizer.PaddingParams{
		Strategy:  *tokenizer.NewPaddingStrategy(tokenizer.WithFixed(maxTokens)),
		Direction: tokenizer.Right,
		PadId:     padID,
		PadTypeId: 0,
		PadToken:  padToken,
	})
	initialized = false
	return &E5{
		profileModelID: "intfloat/multilingual-e5-small",
		tokenizer:      tk,
		session:        session,
		inputNames:     inputNames,
		outputName:     output.Name,
		semaphore:      make(chan struct{}, 2),
	}, nil
}

// EmbedQuery creates one normalized query vector for the active profile.
func (e *E5) EmbedQuery(ctx context.Context, query string, profile Profile) ([]float32, error) {
	vectors, err := e.embed(ctx, []string{withPrefix("query: ", query)}, profile)
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedPassages creates one normalized vector per passage for the active profile.
func (e *E5) EmbedPassages(ctx context.Context, passages []string, profile Profile) ([][]float32, error) {
	if len(passages) == 0 {
		return [][]float32{}, nil
	}
	prefixed := make([]string, len(passages))
	for index, passage := range passages {
		prefixed[index] = withPrefix("passage: ", passage)
		if prefixed[index] == "" {
			return nil, ErrInvalidArgument
		}
	}
	return e.embed(ctx, prefixed, profile)
}

// Ready reports whether the pinned tokenizer and ONNX session can accept work.
func (e *E5) Ready(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.tokenizer == nil || e.session == nil {
		return errors.New("E5 embedder is closed")
	}
	return nil
}

// CountTokens returns the pinned tokenizer's encoded length for one passage.
func (e *E5) CountTokens(text string) (int, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.tokenizer == nil {
		return 0, errors.New("E5 embedder is closed")
	}
	text = withPrefix("passage: ", text)
	if text == "" {
		return 0, ErrInvalidArgument
	}
	encoding, err := e.tokenizer.EncodeSingle(text, true)
	if err != nil {
		return 0, fmt.Errorf("tokenize chunk candidate: %w", err)
	}
	count := 0
	for _, present := range encoding.AttentionMask {
		count += int(present)
	}
	return count, nil
}

func (e *E5) embed(ctx context.Context, texts []string, profile Profile) ([][]float32, error) {
	if profile.EmbeddingModelID != e.profileModelID || profile.VectorDimension != VectorDimension ||
		profile.Distance != "cosine" {
		return nil, ErrProfileMismatch
	}
	if err := acquireEmbeddingSlot(ctx, e.semaphore); err != nil {
		return nil, err
	}
	defer func() { <-e.semaphore }()
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.tokenizer == nil || e.session == nil {
		return nil, errors.New("E5 embedder is closed")
	}
	result := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+batchSize, len(texts))
		vectors, err := e.runBatch(texts[start:end])
		if err != nil {
			return nil, err
		}
		result = append(result, vectors...)
	}
	return result, nil
}

func acquireEmbeddingSlot(ctx context.Context, semaphore chan struct{}) error {
	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()
	select {
	case semaphore <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return ErrTemporarilyUnavailable
	}
}

func (e *E5) runBatch(texts []string) ([][]float32, error) {
	ids := make([]int64, 0, len(texts)*maxTokens)
	masks := make([]int64, 0, len(texts)*maxTokens)
	types := make([]int64, 0, len(texts)*maxTokens)
	for _, text := range texts {
		encoding, err := e.tokenizer.EncodeSingle(text, true)
		if err != nil {
			return nil, fmt.Errorf("tokenize E5 input: %w", err)
		}
		if len(encoding.Ids) != maxTokens || len(encoding.AttentionMask) != maxTokens ||
			len(encoding.TypeIds) != maxTokens {
			return nil, fmt.Errorf("E5 tokenizer returned %d tokens: %w", len(encoding.Ids), ErrProfileMismatch)
		}
		for index := range encoding.Ids {
			ids = append(ids, int64(encoding.Ids[index]))
			masks = append(masks, int64(encoding.AttentionMask[index]))
			types = append(types, int64(encoding.TypeIds[index]))
		}
	}
	shape := ort.NewShape(int64(len(texts)), maxTokens)
	idTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		return nil, fmt.Errorf("create E5 input_ids: %w", err)
	}
	defer idTensor.Destroy()
	maskTensor, err := ort.NewTensor(shape, masks)
	if err != nil {
		return nil, fmt.Errorf("create E5 attention_mask: %w", err)
	}
	defer maskTensor.Destroy()
	typeTensor, err := ort.NewTensor(shape, types)
	if err != nil {
		return nil, fmt.Errorf("create E5 token_type_ids: %w", err)
	}
	defer typeTensor.Destroy()
	inputByName := map[string]ort.Value{
		"input_ids":      idTensor,
		"attention_mask": maskTensor,
		"token_type_ids": typeTensor,
	}
	inputs := make([]ort.Value, len(e.inputNames))
	for index, name := range e.inputNames {
		inputs[index] = inputByName[name]
	}
	output, err := ort.NewEmptyTensor[float32](ort.NewShape(
		int64(len(texts)), maxTokens, VectorDimension,
	))
	if err != nil {
		return nil, fmt.Errorf("create E5 output tensor %s: %w", e.outputName, err)
	}
	defer output.Destroy()
	if err := e.session.Run(inputs, []ort.Value{output}); err != nil {
		return nil, fmt.Errorf("run E5 model: %w", err)
	}
	return meanPool(output.GetData(), masks, len(texts)), nil
}

func meanPool(hidden []float32, masks []int64, batches int) [][]float32 {
	result := make([][]float32, batches)
	for batch := 0; batch < batches; batch++ {
		pooled := make([]float32, VectorDimension)
		tokens := 0
		for token := 0; token < maxTokens; token++ {
			if masks[batch*maxTokens+token] == 0 {
				continue
			}
			offset := (batch*maxTokens + token) * VectorDimension
			for dimension := range pooled {
				pooled[dimension] += hidden[offset+dimension]
			}
			tokens++
		}
		var squared float64
		for dimension := range pooled {
			pooled[dimension] /= float32(tokens)
			squared += float64(pooled[dimension]) * float64(pooled[dimension])
		}
		scale := float32(1 / math.Sqrt(squared))
		for dimension := range pooled {
			pooled[dimension] *= scale
		}
		result[batch] = pooled
	}
	return result
}

// Close releases the ONNX runtime and model resources.
func (e *E5) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	var result error
	if e.session != nil {
		result = errors.Join(result, e.session.Destroy())
		e.session = nil
	}
	e.tokenizer = nil
	result = errors.Join(result, ort.DestroyEnvironment())
	if result != nil {
		return fmt.Errorf("close E5 embedder: %w", result)
	}
	return nil
}

func withPrefix(prefix, value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return prefix + value
}

func verifyArtifact(path, expected string) error {
	if len(expected) != sha256.Size*2 {
		return ErrInvalidArgument
	}
	expectedBytes, err := hex.DecodeString(expected)
	if err != nil {
		return ErrInvalidArgument
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	actual := sha256.Sum256(content)
	if subtle.ConstantTimeCompare(actual[:], expectedBytes) != 1 {
		return errors.New("artifact SHA-256 mismatch")
	}
	return nil
}
