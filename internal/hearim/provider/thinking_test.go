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

// waitCloseFake serves a chat surface that emits a reasoning block before
// the answer: content positions ["<think>", "reasoning", "</think>", "1",
// "2"] with label logprobs only meaningful at the position after the tag.
func waitCloseFake(t *testing.T, lastPrompts *[]string, lastBodies *[]map[string]any) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if lastBodies != nil {
			*lastBodies = append(*lastBodies, body)
		}
		if lastPrompts != nil {
			if msgs, ok := body["messages"].([]any); ok && len(msgs) > 0 {
				last := msgs[len(msgs)-1].(map[string]any)
				*lastPrompts = append(*lastPrompts, last["content"].(string))
			}
		}
		content := []map[string]any{
			{"token": "<think>", "logprob": -0.01},
			{"token": "the answer is ", "logprob": -0.1},
			{"token": "</think>", "logprob": -0.05},
			{"token": "1", "logprob": -0.3, "top_logprobs": []any{
				map[string]any{"token": "1", "bytes": []int{49}, "logprob": -0.3},
				map[string]any{"token": "0", "bytes": []int{48}, "logprob": -1.2},
			}},
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": "<think>the answer is </think>1"},
				"logprobs":      map[string]any{"content": content},
				"finish_reason": "length",
			}},
			"usage": map[string]any{"prompt_tokens": 30, "completion_tokens": len(content)},
		})
	})
	return mux
}

func TestWaitCloseScoresPositionAfterTag(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(waitCloseFake(t, nil, &bodies))
	defer srv.Close()

	a := NewOllama(config.ProviderConfig{ID: "x", Engine: config.EngineOllama, BaseURL: srv.URL})
	res, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "qwen3"},
		PromptText:          "PROMPT",
		CandidateTokenTexts: []string{"1", "0"},
		WaitClose:           true,
		CloseTag:            "</think>",
		MaxOutputTokens:     128,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ScoringMethod != "wait-close-tag" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
	// The distribution must come from the position AFTER </think>, not the
	// first token ("<think>" itself).
	if lp := res.CandidateLogprobs[0]; math.Abs(lp-(-0.3)) > 1e-9 {
		t.Errorf("logprob[1] = %v, want -0.3", lp)
	}
	if lp := res.CandidateLogprobs[1]; math.Abs(lp-(-1.2)) > 1e-9 {
		t.Errorf("logprob[0] = %v, want -1.2", lp)
	}
	// Generation budget widened for the reasoning block.
	if bodies[0]["max_tokens"] != float64(128) {
		t.Errorf("max_tokens = %v", bodies[0]["max_tokens"])
	}
}

func TestWaitCloseNeverClosedFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := []map[string]any{
			{"token": "<think>", "logprob": -0.01},
			{"token": "still thinking", "logprob": -0.1},
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":  map[string]any{"role": "assistant", "content": "<think>..."},
				"logprobs": map[string]any{"content": content},
			}},
		})
	}))
	defer srv.Close()

	a := NewOllama(config.ProviderConfig{ID: "x", Engine: config.EngineOllama, BaseURL: srv.URL})
	_, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:               ModelIdentity{Model: "m"},
		PromptText:          "P",
		CandidateTokenTexts: []string{"1", "0"},
		WaitClose:           true,
		CloseTag:            "</think>",
	})
	if err == nil || !strings.Contains(err.Error(), "never closed") {
		t.Errorf("expected never-closed error, got %v", err)
	}
}

func TestReasoningOverrideFieldWins(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(waitCloseFake(t, nil, &bodies))
	defer srv.Close()

	a := NewOllama(config.ProviderConfig{ID: "x", Engine: config.EngineOllama, BaseURL: srv.URL})
	_, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:          ModelIdentity{Model: "m"},
		PromptText:     "P",
		NoReasoning:    true,
		ReasoningField: "reasoning_effort",
		ReasoningValue: "none",
	})
	if err != nil {
		t.Fatal(err)
	}
	b := bodies[0]
	if b["reasoning_effort"] != "none" {
		t.Errorf("override field missing: %v", b)
	}
	if v, present := b["think"]; present {
		t.Errorf("engine default should not be set when override present: think=%v", v)
	}
}

func TestScanAfterCloseTag(t *testing.T) {
	toks := []string{"<th", "ink>", "abc ", "</th", "ink>", "1", "2"}
	if pos, err := scanAfterCloseTag(toks, "</think>"); err != nil || pos != 5 {
		t.Errorf("pos = %d %v, want 5", pos, err)
	}
	if _, err := scanAfterCloseTag([]string{"a", "b"}, "</think>"); err == nil {
		t.Error("unclosed should error")
	}
	if pos, err := scanAfterCloseTag([]string{"a"}, ""); err != nil || pos != 0 {
		t.Errorf("empty tag = first position, got %d %v", pos, err)
	}
}

func TestReasoningOverrideWithObjectValue(t *testing.T) {
	// z.ai's standard endpoint documents thinking: {"type": "disabled"} —
	// an OBJECT-valued control. It must marshal cleanly into the upstream
	// body (verified live: the standard endpoint honors it; the coding
	// endpoint ignores it).
	var bodies []map[string]any
	srv := httptest.NewServer(waitCloseFake(t, nil, &bodies))
	defer srv.Close()

	a := NewOllama(config.ProviderConfig{ID: "x", Engine: config.EngineOllama, BaseURL: srv.URL})
	_, err := a.ScoreNextToken(context.Background(), NextTokenScoreRequest{
		Model:          ModelIdentity{Model: "glm-4.5-air"},
		PromptText:     "P",
		ReasoningField: "thinking",
		ReasoningValue: map[string]any{"type": "disabled"},
	})
	if err != nil {
		t.Fatal(err)
	}
	th, ok := bodies[0]["thinking"].(map[string]any)
	if !ok || th["type"] != "disabled" {
		t.Errorf("thinking object control not sent: %v", bodies[0])
	}
}
