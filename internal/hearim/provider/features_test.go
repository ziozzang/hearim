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

func TestPathOverrides(t *testing.T) {
	// A gateway that only exposes chat under a custom path: the adapter must
	// hit the configured path, not the engine default.
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"index":   0,
				"message": map[string]any{"role": "assistant", "content": "1"},
				"logprobs": map[string]any{
					"content": []any{map[string]any{
						"token": "1", "logprob": -0.4,
						"top_logprobs": []any{
							map[string]any{"token": "1", "bytes": []int{49}, "logprob": -0.4},
							map[string]any{"token": "0", "bytes": []int{48}, "logprob": -1.1},
						},
					}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 1},
		})
	}))
	defer srv.Close()

	chatPath := "/api/v2/chat"
	cfg := config.ProviderConfig{
		ID: "gw", Engine: config.EngineOllama, BaseURL: srv.URL,
		Paths: &config.PathsConfig{ChatCompletions: chatPath},
	}
	a := NewOllama(cfg)
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		PromptText:          "<state>x</state><question/><criteria>1 = a\n0 = b</criteria><answer-label>\n",
		CandidateTokenTexts: []string{"1", "0"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != chatPath {
		t.Errorf("path = %s, want %s", gotPath, chatPath)
	}
	if res.CandidateLogprobs[0] != -0.4 {
		t.Errorf("logprob = %v", res.CandidateLogprobs[0])
	}
}

func TestPathForDefaultsAndOverrides(t *testing.T) {
	// Engine defaults.
	if p := pathFor(config.ProviderConfig{Engine: config.EngineLlamaPP}, PathCompletion); p != "/completion" {
		t.Errorf("llama completion = %s", p)
	}
	if p := pathFor(config.ProviderConfig{Engine: config.EngineSGLang}, PathGenerate); p != "/generate" {
		t.Errorf("sglang generate = %s", p)
	}
	if p := pathFor(config.ProviderConfig{Engine: config.EngineOllama}, PathTokenize); p != "/api/tokenize" {
		t.Errorf("ollama tokenize = %s", p)
	}
	// Overrides win; health_path stays backward compatible.
	cfg := config.ProviderConfig{Engine: config.EngineVLLM,
		Paths:      &config.PathsConfig{Completions: "/custom/completions"},
		HealthPath: "/custom/health"}
	if p := pathFor(cfg, PathCompletions); p != "/custom/completions" {
		t.Errorf("override = %s", p)
	}
	if p := pathFor(cfg, PathTokenize); p != "/tokenize" {
		t.Errorf("default tokenize = %s", p)
	}
	if p := pathFor(cfg, PathHealth); p != "/custom/health" {
		t.Errorf("health_path = %s", p)
	}
}

func TestForcedEndpointPrecedence(t *testing.T) {
	chat := config.EndpointChatCompletion
	comp := config.EndpointCompletions
	pcfg := config.ProviderConfig{Endpoint: &comp}
	if f := ForcedEndpoint(pcfg, nil); f == nil || *f != comp {
		t.Error("provider force should apply")
	}
	if f := ForcedEndpoint(pcfg, &config.ModelConfig{Endpoint: &chat}); f == nil || *f != chat {
		t.Error("model force should win over provider force")
	}
	if f := ForcedEndpoint(config.ProviderConfig{}, nil); f != nil {
		t.Error("no force configured")
	}
}

func TestQueryParamsAndHeaders(t *testing.T) {
	var gotQuery, gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		gotHeader = r.Header.Get("X-Api-Key")
		w.Header().Set("X-Cost-Usd", "0.0012")
		w.Header().Set("X-Ratelimit-Remaining", "41")
		w.Header().Set("X-Ignored-Internal", "nope")
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
		ID: "gw", Engine: config.EngineVLLM, BaseURL: srv.URL,
		QueryParams: map[string]string{"detailed": "true", "version": "2"},
		Headers:     map[string]string{"X-Api-Key": "gateway-key-123"},
	}
	a := NewVLLM(cfg)
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		CandidateTokenTexts: []string{"1", "0"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotQuery != "detailed=true&version=2" {
		t.Errorf("query = %s", gotQuery)
	}
	if gotHeader != "gateway-key-123" {
		t.Errorf("header = %s", gotHeader)
	}
	if res.UpstreamHeaders["x-cost-usd"] != "0.0012" {
		t.Errorf("cost header not captured: %v", res.UpstreamHeaders)
	}
	if res.UpstreamHeaders["x-ratelimit-remaining"] != "41" {
		t.Errorf("ratelimit header not captured: %v", res.UpstreamHeaders)
	}
	if _, leaked := res.UpstreamHeaders["x-ignored-internal"]; leaked {
		t.Errorf("non-whitelisted header leaked: %v", res.UpstreamHeaders)
	}
}

func TestGenericChatEndpointForcing(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"index":   0,
				"message": map[string]any{"role": "assistant", "content": "1"},
				"logprobs": map[string]any{
					"content": []any{map[string]any{
						"token": "1", "logprob": -0.4,
						"top_logprobs": []any{
							map[string]any{"token": "1", "logprob": -0.4},
							map[string]any{"token": "0", "logprob": -1.4},
						},
					}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 1},
		})
	}))
	defer srv.Close()

	a := NewGeneric(config.ProviderConfig{ID: "g", Engine: config.EngineGeneric, BaseURL: srv.URL})
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		Endpoint:            config.EndpointChatCompletion,
		PromptText:          "<state>x</state><question/><criteria>1 = a\n0 = b</criteria><answer-label>\n",
		CandidateTokenTexts: []string{"1", "0"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" {
		t.Errorf("path = %s", path)
	}
	if res.CandidateLogprobs[0] != -0.4 {
		t.Errorf("logprob = %v", res.CandidateLogprobs[0])
	}
}
