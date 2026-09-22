package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"hearim/internal/hearim/config"
)

func TestScoringProfileModelOverrides(t *testing.T) {
	yes, no := true, false
	cfg := testProvider(config.EngineVLLM, "http://unused")
	cfg.Scoring = &config.ProviderScoringConfig{SelectedTokenIDs: &no, Constraint: "none", MaxSelectedTokenIDs: 64, LogprobSpace: SpacePostMask}
	cfg.Models = config.ModelConfigs{{Name: "enabled", Scoring: &config.ProviderScoringConfig{SelectedTokenIDs: &yes, Constraint: "allowed_token_ids", LogprobSpace: SpaceRaw}}}
	a := NewVLLM(cfg)
	base, model := resolveScoring(cfg, "other"), resolveScoring(cfg, "enabled")
	if base.selected || base.constraint != "none" || base.space != SpacePostMask {
		t.Fatalf("base = %+v", base)
	}
	if !model.selected || model.maxSelected != 64 || model.constraint != "allowed_token_ids" || model.space != SpaceRaw {
		t.Fatalf("model = %+v", model)
	}
	if CapabilitiesForModel(a, "other").SupportsSelectedTokenIDs() || !CapabilitiesForModel(a, "enabled").SupportsSelectedTokenIDs() {
		t.Fatal("model capabilities not isolated")
	}
	if ScoringProfileKey(a, "other") == ScoringProfileKey(a, "enabled") {
		t.Fatal("profile changes must invalidate registry cache")
	}
}

func TestVLLMSelectedLimitAndTopKRetry(t *testing.T) {
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		last = nil
		json.NewDecoder(r.Body).Decode(&last)
		fmt.Fprint(w, `{"choices":[{"logprobs":{"top_logprobs":[{"A":-0.2,"B":-2}]}}]}`)
	}))
	defer srv.Close()
	a := NewVLLM(testProvider(config.EngineVLLM, srv.URL))
	for _, tc := range []struct {
		n        int
		force    bool
		selected bool
	}{{2, false, true}, {128, false, true}, {129, false, false}, {2, true, false}} {
		req := NextTokenScoreRequest{Model: ModelIdentity{Model: "m"}, TopK: 10, DisableSelectedTokenIDs: tc.force}
		for i := 0; i < tc.n; i++ {
			req.CandidateTokenIDs = append(req.CandidateTokenIDs, i)
			req.CandidateTokenTexts = append(req.CandidateTokenTexts, fmt.Sprint(i))
		}
		if _, err := a.ScoreNextToken(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if (last["logprob_token_ids"] != nil) != tc.selected {
			t.Fatalf("n=%d force=%v body=%v", tc.n, tc.force, last)
		}
		if last["allowed_token_ids"] != nil {
			t.Fatal("ordinary scoring must not add a mask")
		}
	}
}

func TestVLLMChatConstraintsAndConfiguredSpace(t *testing.T) {
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&last)
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path=%s", r.URL.Path)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"A"},"logprobs":{"content":[{"token":"A","logprob":-0.2,"top_logprobs":[{"token":"A","logprob":-0.2},{"token":"B","logprob":-2}]}]}}]}`)
	}))
	defer srv.Close()
	cfg := testProvider(config.EngineVLLM, srv.URL)
	cfg.Scoring = &config.ProviderScoringConfig{LogprobSpace: SpacePostMask}
	a := NewVLLM(cfg)
	req := NextTokenScoreRequest{Model: ModelIdentity{Model: "m"}, Endpoint: config.EndpointChatCompletion, PromptText: "Choose A or B", CandidateTokenIDs: []int{3, 4}, CandidateTokenTexts: []string{"A", "B"}, ConstrainToCandidates: true}
	res, err := a.ScoreNextToken(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if last["allowed_token_ids"] == nil || last["logprob_token_ids"] == nil || last["extra_body"] != nil {
		t.Fatalf("wire body=%v", last)
	}
	if res.ProbabilitySpace != SpacePostMask || !res.AllCandidatesPresent {
		t.Fatalf("result=%+v", res)
	}
	cfg.Models = config.ModelConfigs{{Name: "m", Scoring: &config.ProviderScoringConfig{Constraint: "none"}}}
	if _, err = NewVLLM(cfg).ScoreNextToken(context.Background(), req); err == nil {
		t.Fatal("disabled mask must fail before sending")
	}
	req.CandidateTokenIDs = []int{-1, 4}
	if _, err = a.ScoreNextToken(context.Background(), req); err == nil {
		t.Fatal("unresolved IDs must not be sent as a mask")
	}
}

func TestSGLangNativeTopKAndUnsupportedMask(t *testing.T) {
	no := false
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		sp := body["sampling_params"].(map[string]any)
		if sp["allowed_token_ids"] != nil || body["token_ids_logprob"] != nil || body["top_logprobs_num"] != float64(8) {
			t.Errorf("body=%v", body)
		}
		fmt.Fprint(w, `{"text":"A","meta_info":{"output_top_logprobs":[[[-0.2,3,"A"],[-2,4,"B"]]],"id":"r","prompt_tokens":9}}`)
	}))
	defer srv.Close()
	cfg := testProvider(config.EngineSGLang, srv.URL)
	cfg.Models = config.ModelConfigs{{Name: "m", Scoring: &config.ProviderScoringConfig{SelectedTokenIDs: &no}}}
	a := NewSGLang(cfg)
	req := NextTokenScoreRequest{Model: ModelIdentity{Model: "m"}, CandidateTokenIDs: []int{3, 4}, CandidateTokenTexts: []string{"A", "B"}, TopK: 8}
	res, err := a.ScoreNextToken(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !res.AllCandidatesPresent || res.ScoringMethod != "top-k" || res.BackendRequestID != "r" {
		t.Fatalf("result=%+v", res)
	}
	req.ConstrainToCandidates = true
	if _, err = a.ScoreNextToken(context.Background(), req); err == nil || calls != 1 {
		t.Fatal("unsupported SGLang mask must not be sent")
	}
}

func TestSGLangPromptTuplesAndZeroLogprob(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/tokenize" {
			fmt.Fprint(w, `{"input_ids":[1,2,3]}`)
			return
		}
		fmt.Fprint(w, `{"meta_info":{"input_token_logprobs":[[null,1,null],[-0.5,2,null],[0,3,null]]}}`)
	}))
	defer srv.Close()
	a := NewSGLang(testProvider(config.EngineSGLang, srv.URL))
	res, err := a.ScoreContinuations(context.Background(), ContinuationScoreRequest{Model: ModelIdentity{Model: "m"}, PrefixTokenIDs: []int{1, 2}, PrefixText: "p", Continuations: []string{"A"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Conditional) != 1 || res.Conditional[0].TokenLen != 1 || res.Conditional[0].LogProb != 0 {
		t.Fatalf("result=%+v", res)
	}
	for _, bad := range []string{`[null,3,null]`, `{"token_id":3}`, `["bad",3,null]`} {
		if _, err := parseSGLangEntry(json.RawMessage(bad)); err == nil {
			t.Errorf("accepted missing logprob: %s", bad)
		}
	}
}

func TestLlamaCppRejectsIgnoredPostSampling(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"completion_probabilities":[{"id":3,"logprob":-0.2,"token":"A","top_logprobs":[{"id":3,"logprob":-0.2,"token":"A"}]}]}`)
	}))
	defer srv.Close()
	a := NewLlamaCpp(testProvider(config.EngineLlamaPP, srv.URL))
	req := NextTokenScoreRequest{CandidateTokenIDs: []int{3}, CandidateTokenTexts: []string{"A"}}
	res, err := a.ScoreNextToken(context.Background(), req)
	if err != nil || math.Abs(res.CandidateLogprobs[0]+0.2) > 1e-9 {
		t.Fatalf("raw result=%+v err=%v", res, err)
	}
	req.ConstrainToCandidates = true
	if _, err := a.ScoreNextToken(context.Background(), req); err == nil {
		t.Fatal("raw response cannot be tagged post-mask")
	}
}
