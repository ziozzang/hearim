package provider

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"hearim/internal/hearim/config"
)

func testProvider(engine config.EngineKind, url string) config.ProviderConfig {
	return config.ProviderConfig{
		ID:          string(engine) + "-test",
		Engine:      engine,
		BaseURL:     url,
		Concurrency: 3,
	}
}

// --- vLLM ---

func TestVLLMScoreNextTokenSelectedIDs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["logprob_token_ids"] == nil {
			t.Error("logprob_token_ids not requested")
		}
		if body["max_tokens"] != float64(1) {
			t.Errorf("max_tokens = %v", body["max_tokens"])
		}
		if body["temperature"] != float64(1) || body["top_p"] != float64(1) {
			t.Errorf("sampler params = %v/%v", body["temperature"], body["top_p"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id": "req-1",
			"choices": []any{map[string]any{
				"text": "2",
				"logprobs": map[string]any{
					"tokens": []string{"2"},
					"top_logprobs": [][]any{[]any{
						map[string]any{"token": "1", "bytes": []int{49}, "logprob": -2.5, "top_logprobs": nil},
						map[string]any{"token": "2", "bytes": []int{50}, "logprob": -0.2, "top_logprobs": nil},
						map[string]any{"token": "3", "bytes": []int{51}, "logprob": -4.1, "top_logprobs": nil},
						map[string]any{"token": "4", "bytes": []int{52}, "logprob": -5.9, "top_logprobs": nil},
					}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 120, "completion_tokens": 1,
				"prompt_tokens_details": map[string]any{"cached_tokens": 80}},
		})
	}))
	defer srv.Close()

	a := NewVLLM(testProvider(config.EngineVLLM, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Provider: "vllm-test", Model: "m"},
		Endpoint:            config.EndpointCompletions,
		PromptText:          "PROMPT",
		CandidateTokenIDs:   []int{1001, 1002, 1003, 1004},
		CandidateTokenTexts: []string{"1", "2", "3", "4"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ScoringMethod != "selected-token-ids" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
	if !res.AllCandidatesPresent {
		t.Error("all candidates should be present")
	}
	if lp := res.CandidateLogprobs[1]; math.Abs(lp-(-0.2)) > 1e-9 {
		t.Errorf("logprob[1] = %v", lp)
	}
	if res.PromptTokens != 120 || res.CachedPromptTokens != 80 {
		t.Errorf("usage = %d/%d", res.PromptTokens, res.CachedPromptTokens)
	}
	if res.ProbabilitySpace != SpaceRaw {
		t.Errorf("space = %s", res.ProbabilitySpace)
	}
}

func TestVLLMScoreNextTokenLegacyMapShape(t *testing.T) {
	// OpenAI legacy top_logprobs: array of maps.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"logprobs": map[string]any{
					"top_logprobs": []any{
						map[string]any{"1": -0.69, "0": -0.7},
					},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 1},
		})
	}))
	defer srv.Close()

	a := NewVLLM(testProvider(config.EngineVLLM, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		CandidateTokenTexts: []string{"1", "0"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if lp := res.CandidateLogprobs[0]; math.Abs(lp-(-0.69)) > 1e-9 {
		t.Errorf("logprob = %v", lp)
	}
}

func TestVLLMConstrainedSpace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["allowed_token_ids"] == nil {
			t.Error("allowed_token_ids missing")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"logprobs": map[string]any{
					"top_logprobs": [][]any{[]any{
						map[string]any{"token": "1", "logprob": -1.1},
						map[string]any{"token": "0", "logprob": -2.2},
					}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 1},
		})
	}))
	defer srv.Close()

	a := NewVLLM(testProvider(config.EngineVLLM, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:                 ModelIdentity{Model: "m"},
		CandidateTokenIDs:     []int{1, 0},
		CandidateTokenTexts:   []string{"1", "0"},
		ConstrainToCandidates: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ProbabilitySpace != SpacePostMask {
		t.Errorf("space = %s", res.ProbabilitySpace)
	}
	if res.ScoringMethod != "constrained-vocab" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
}

// --- SGLang ---

func TestSGLangScoreNextTokenTokenIDsLogprob(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/generate" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["token_ids_logprob"] == nil {
			t.Error("token_ids_logprob missing")
		}
		sp := body["sampling_params"].(map[string]any)
		if sp["max_new_tokens"] != float64(1) {
			t.Errorf("max_new_tokens = %v", sp["max_new_tokens"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"text": "2",
			"logprobs": map[string]any{
				"output_token_ids_logprobs": []any{
					[]any{
						map[string]any{"token_id": 1001, "logprob": -1.5},
						map[string]any{"token_id": 1002, "logprob": -0.3},
					},
				},
			},
			"meta_info": map[string]any{"prompt_tokens": 200, "cached_tokens": 150, "req_id": "r9"},
		})
	}))
	defer srv.Close()

	a := NewSGLang(testProvider(config.EngineSGLang, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		Endpoint:            config.EndpointNativeGenerate,
		PromptTokenIDs:      []int{1, 2, 3},
		CandidateTokenIDs:   []int{1001, 1002},
		CandidateTokenTexts: []string{"1", "2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.AllCandidatesPresent {
		t.Error("candidates missing")
	}
	if lp := res.CandidateLogprobs[0]; math.Abs(lp-(-1.5)) > 1e-9 {
		t.Errorf("logprob[0] = %v", lp)
	}
	if res.CachedPromptTokens != 150 {
		t.Errorf("cached = %d", res.CachedPromptTokens)
	}
	if res.ScoringMethod != "selected-token-ids" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
}

func TestSGLangInputTokenIDsPreferred(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if _, ok := body["input_ids"]; !ok {
			t.Error("input_ids should be sent when available")
		}
		if body["text"] != nil {
			t.Error("text should be omitted when input_ids present")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"logprobs": map[string]any{
				"output_token_ids_logprobs": []any{
					[]any{map[string]any{"token_id": 5, "logprob": -0.1}},
				},
			},
			"meta_info": map[string]any{"prompt_tokens": 3},
		})
	}))
	defer srv.Close()

	a := NewSGLang(testProvider(config.EngineSGLang, srv.URL))
	_, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		PromptTokenIDs:      []int{1, 2, 3},
		CandidateTokenIDs:   []int{5},
		CandidateTokenTexts: []string{"1"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- llama.cpp ---

func TestLlamaCppScoreNextToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completion" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["n_predict"] != float64(1) {
			t.Errorf("n_predict = %v", body["n_predict"])
		}
		if body["cache_prompt"] != true {
			t.Errorf("cache_prompt = %v", body["cache_prompt"])
		}
		if _, ok := body["prompt"].([]any); !ok {
			t.Errorf("prompt should be token ids, got %T", body["prompt"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"content": "A",
			"completion_probabilities": []any{
				map[string]any{
					"content": "A",
					"id":      1234,
					"prob":    0.7,
					"top_logprobs": []any{
						map[string]any{"id": 1234, "content": "A", "prob": 0.7},
						map[string]any{"id": 1235, "content": "B", "prob": 0.2},
						map[string]any{"id": 1236, "content": "C", "prob": 0.08},
						map[string]any{"id": 1237, "content": "D", "prob": 0.02},
					},
				},
			},
			"tokens_cached":    90,
			"tokens_evaluated": 100,
		})
	}))
	defer srv.Close()

	a := NewLlamaCpp(testProvider(config.EngineLlamaPP, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		Endpoint:            config.EndpointNativeGenerate,
		PromptTokenIDs:      []int{1, 2, 3},
		CandidateTokenIDs:   []int{1234, 1235, 1236, 1237},
		CandidateTokenTexts: []string{"A", "B", "C", "D"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// llama.cpp returns probabilities; the adapter must convert to logprobs.
	if lp := res.CandidateLogprobs[0]; math.Abs(lp-math.Log(0.7)) > 1e-9 {
		t.Errorf("logprob[0] = %v, want ln(0.7)", lp)
	}
	if lp := res.CandidateLogprobs[3]; math.Abs(lp-math.Log(0.02)) > 1e-9 {
		t.Errorf("logprob[3] = %v", lp)
	}
	if !res.AllCandidatesPresent {
		t.Error("candidates missing")
	}
	if res.CachedPromptTokens != 90 || res.PromptTokens != 100 {
		t.Errorf("cache observability = %d/%d", res.CachedPromptTokens, res.PromptTokens)
	}
}

func TestLlamaCppGrammarConstraint(t *testing.T) {
	var gotGrammar string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		gotGrammar, _ = body["grammar"].(string)
		json.NewEncoder(w).Encode(map[string]any{
			"completion_probabilities": []any{
				map[string]any{
					"id":      1,
					"content": "1",
					"prob":    0.6,
					"top_logprobs": []any{
						map[string]any{"id": 1, "content": "1", "prob": 0.6},
						map[string]any{"id": 2, "content": "0", "prob": 0.4},
					},
				},
			},
			"tokens_evaluated": 10,
		})
	}))
	defer srv.Close()

	a := NewLlamaCpp(testProvider(config.EngineLlamaPP, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:                 ModelIdentity{Model: "m"},
		CandidateTokenTexts:   []string{"1", "0"},
		ConstrainToCandidates: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(gotGrammar, `"1"`) || !strings.Contains(gotGrammar, `"0"`) {
		t.Errorf("grammar = %q", gotGrammar)
	}
	if res.ProbabilitySpace != SpacePostMask {
		t.Errorf("space = %s", res.ProbabilitySpace)
	}
}

func TestLlamaCppTokenize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["add_special"] != true {
			t.Error("add_special expected")
		}
		json.NewEncoder(w).Encode(map[string]any{"tokens": []int{1, 22, 333}})
	}))
	defer srv.Close()

	a := NewLlamaCpp(testProvider(config.EngineLlamaPP, srv.URL))
	ids, err := a.Tokenize(context.Background(), "m", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 3 || ids[2] != 333 {
		t.Errorf("ids = %v", ids)
	}
}

// --- Ollama ---

func TestOllamaTopKScoring(t *testing.T) {
	// Ollama's exact logprob surface is chat completions (probed 2026-09-21):
	// choices[].logprobs.content[].top_logprobs.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["logprobs"] != true {
				t.Error("logprobs must be requested")
			}
			if body["max_tokens"] != float64(1) {
				t.Errorf("max_tokens = %v", body["max_tokens"])
			}
			msgs, _ := body["messages"].([]any)
			if len(msgs) == 0 {
				t.Error("chat messages missing")
			}
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{
					"index":   0,
					"message": map[string]any{"role": "assistant", "content": "2"},
					"logprobs": map[string]any{
						"content": []any{map[string]any{
							"token":   "2",
							"logprob": -0.1,
							"bytes":   []int{50},
							"top_logprobs": []any{
								map[string]any{"token": "1", "logprob": -0.9, "bytes": []int{49}},
								map[string]any{"token": "2", "logprob": -0.1, "bytes": []int{50}},
								map[string]any{"token": "3", "logprob": -3.0, "bytes": []int{51}},
								map[string]any{"token": "4", "logprob": -4.0, "bytes": []int{52}},
							},
						}},
					},
					"finish_reason": "length",
				}},
				"usage": map[string]any{"prompt_tokens": 42, "completion_tokens": 1,
					"prompt_tokens_details": map[string]any{"cached_tokens": 20}},
			})
		case "/api/version":
			json.NewEncoder(w).Encode(map[string]any{"version": "0.99"})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	a := NewOllama(testProvider(config.EngineOllama, srv.URL))
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "gemma4:31b"},
		PromptText:          "<state>x</state>\n<question/>\n<criteria>1 = a\n2 = b\n</criteria>\n<answer-label>\n",
		CandidateTokenTexts: []string{"1", "2", "3", "4"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ScoringMethod != "top-k" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
	if !res.AllCandidatesPresent {
		t.Error("all candidates should be found via text matching")
	}
	if lp := res.CandidateLogprobs[1]; math.Abs(lp-(-0.1)) > 1e-9 {
		t.Errorf("logprob = %v", lp)
	}
	if res.CachedPromptTokens != 20 {
		t.Errorf("cached = %d", res.CachedPromptTokens)
	}
	if !a.ChatRender() {
		t.Error("ollama exact route uses chat rendering")
	}
	// Continuation scoring must report unsupported.
	if _, err := a.ScoreContinuations(context.Background(), ContinuationScoreRequest{}); err == nil {
		t.Error("ollama continuation scoring should be unsupported")
	}
	// Tokenize must report probe-only absence (fake server lacks /api/tokenize).
	if _, err := a.Tokenize(context.Background(), "m", "x"); err == nil {
		t.Error("tokenize should fail on servers without the endpoint")
	}
}

// --- route resolution & scoring ---

func TestResolveExactRouteOrder(t *testing.T) {
	policy := config.BackendPolicyConfig{}
	caps := ProviderCapabilities{
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointChatCompletion, NextTokenLogprobsVerified: true,
				ReasoningCanBeFullyDisabled: true},
			{Kind: config.EndpointCompletions, NextTokenLogprobsVerified: true},
		},
	}
	e, err := ResolveExactRoute(caps, policy)
	if err != nil {
		t.Fatal(err)
	}
	if e.Kind != config.EndpointCompletions {
		t.Errorf("completions must outrank chat, got %s", e.Kind)
	}

	// Chat-only with reasoning disabled OK.
	caps2 := ProviderCapabilities{Endpoints: []EndpointProfile{
		{Kind: config.EndpointChatCompletion, NextTokenLogprobsVerified: true, ReasoningCanBeFullyDisabled: true},
	}}
	if e, err := ResolveExactRoute(caps2, policy); err != nil || e.Kind != config.EndpointChatCompletion {
		t.Errorf("verified no-reasoning chat should be usable: %v", err)
	}

	// Chat-only without full reasoning disable -> no exact route (§3.4).
	caps3 := ProviderCapabilities{Endpoints: []EndpointProfile{
		{Kind: config.EndpointChatCompletion, NextTokenLogprobsVerified: true, ReasoningCanBeFullyDisabled: false},
	}}
	if _, err := ResolveExactRoute(caps3, policy); err == nil {
		t.Error("chat-only without reasoning disable must be rejected")
	}
}

func TestRouteScoreOrdering(t *testing.T) {
	// §3.10: SGLang/vLLM > llama.cpp > Ollama > verified chat.
	vllm := NewVLLM(testProvider(config.EngineVLLM, "http://x")).Capabilities()
	llama := NewLlamaCpp(testProvider(config.EngineLlamaPP, "http://x")).Capabilities()
	ollama := NewOllama(testProvider(config.EngineOllama, "http://x")).Capabilities()
	if RouteScore(vllm) <= RouteScore(llama) {
		t.Errorf("vllm %d should outrank llama.cpp %d", RouteScore(vllm), RouteScore(llama))
	}
	if RouteScore(llama) <= RouteScore(ollama) {
		t.Errorf("llama.cpp %d should outrank ollama %d", RouteScore(llama), RouteScore(ollama))
	}
	chatOnly := ProviderCapabilities{
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointChatCompletion, NextTokenLogprobsVerified: true, ReasoningCanBeFullyDisabled: false},
		},
	}
	if RouteScore(chatOnly) >= 0 {
		t.Errorf("undisableable chat-only must score below 0, got %d", RouteScore(chatOnly))
	}
}

func TestUpstreamErrorClassification(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "slow down"}})
	}))
	defer srv.Close()

	a := NewVLLM(testProvider(config.EngineVLLM, srv.URL))
	a.cfg.MaxRetries = 0
	_, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{Model: ModelIdentity{Model: "m"}})
	if err == nil {
		t.Fatal("expected error")
	}
	if !IsRetriable(err) {
		t.Errorf("429 should be retriable: %v", err)
	}
	var ue *UpstreamError
	if !asUpstream(err, &ue) || ue.Status != 429 {
		t.Errorf("classification failed: %v", err)
	}
}

func asUpstream(err error, target **UpstreamError) bool {
	if e, ok := err.(*UpstreamError); ok {
		*target = e
		return true
	}
	return false
}
