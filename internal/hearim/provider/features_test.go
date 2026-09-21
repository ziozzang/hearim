package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"hearim/internal/hearim/config"
)

func TestEndpointPoolFailover(t *testing.T) {
	var hitsA, hitsB int64
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hitsA, 1)
		w.WriteHeader(http.StatusServiceUnavailable) // dead replica
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hitsB, 1)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"logprobs": map[string]any{
					"top_logprobs": []any{map[string]any{"1": -0.5, "0": -1.0}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 1},
		})
	}))
	defer b.Close()

	cfg := config.ProviderConfig{
		ID: "pool", Engine: config.EngineVLLM,
		BaseURLs:   []string{a.URL, b.URL},
		MaxRetries: 2,
	}
	adpt := NewVLLM(cfg)
	// Several requests: rotation covers the dead replica, and every request
	// still succeeds because retries fail over to the healthy one.
	for i := 0; i < 4; i++ {
		res, err := adpt.ScoreNextToken(context.Background(), NextTokenScoreRequest{
			Model:               ModelIdentity{Model: "m"},
			CandidateTokenTexts: []string{"1", "0"},
		})
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		if lp := res.CandidateLogprobs[0]; lp != -0.5 {
			t.Errorf("logprob = %v", lp)
		}
	}
	if atomic.LoadInt64(&hitsA) == 0 {
		t.Errorf("dead replica never attempted: a=%d b=%d", hitsA, hitsB)
	}
	if atomic.LoadInt64(&hitsB) < 4 {
		t.Errorf("healthy replica underused: a=%d b=%d", hitsA, hitsB)
	}
}

func TestExtraParamsMergedUnprotected(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&seen)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"logprobs": map[string]any{
					"top_logprobs": []any{map[string]any{"1": -0.5, "0": -1.0}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 1},
		})
	}))
	defer srv.Close()

	cfg := config.ProviderConfig{
		ID: "p", Engine: config.EngineVLLM, BaseURL: srv.URL,
		ExtraParams: map[string]any{"seed": 7, "logprobs": 99, "max_tokens": 500},
		Models: config.ModelConfigs{
			{Name: "m", ExtraParams: map[string]any{"seed": 42, "top_k_seed": "x"}},
		},
	}
	adpt := NewVLLM(cfg)
	_, err := adpt.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		CandidateTokenTexts: []string{"1", "0"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen["seed"] != float64(42) { // model-level wins over provider-level
		t.Errorf("seed = %v (want 42, model overrides provider)", seen["seed"])
	}
	if seen["logprobs"] == float64(99) {
		t.Error("protected field logprobs must not be overridable")
	}
	if seen["max_tokens"] == float64(500) {
		t.Error("protected field max_tokens must not be overridable")
	}
	if seen["top_k_seed"] != "x" {
		t.Errorf("custom field dropped: %v", seen)
	}
}

func TestModelIsThinking(t *testing.T) {
	if !ModelIsThinking(&config.ModelConfig{Type: "thinking"}) {
		t.Error("thinking should be true")
	}
	if ModelIsThinking(&config.ModelConfig{Type: "vision"}) {
		t.Error("vision is not thinking")
	}
	if ModelIsThinking(nil) {
		t.Error("nil is not thinking")
	}
}
