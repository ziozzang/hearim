package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/scoring"
)

// VLLMAdapter targets vLLM /v1/completions with logprob_token_ids and
// allowed_token_ids (TODO.md §3.8), /tokenize, and APC.
type VLLMAdapter struct {
	cfg  config.ProviderConfig
	hc   *httpClient
	caps ProviderCapabilities
}

func NewVLLM(cfg config.ProviderConfig) *VLLMAdapter {
	a := &VLLMAdapter{cfg: cfg, hc: newHTTPClient(cfg)}
	maxTop := cfg.MaxTopLogprobs
	if maxTop == 0 {
		maxTop = 20
	}
	selectedField := cfg.SelectedTokenField
	if selectedField == "" {
		selectedField = "logprob_token_ids"
	}
	maxSelected := 256
	a.caps = ProviderCapabilities{
		Engine: config.EngineVLLM,
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointCompletions, MaxTopLogprobs: maxTop, MaxSelectedTokenIDs: maxSelected},
			{Kind: config.EndpointChatCompletion, MaxTopLogprobs: maxTop,
				ReasoningControl: "effort", ReasoningField: "reasoning_effort", ReasoningValue: "none",
				ReasoningCanBeFullyDisabled: false},
		},
		Tokenizer:           "remote",
		CandidateScoring:    []string{"selected-token-ids", "top-k", "teacher-forced"},
		PromptTokenLogprobs: true,
		MaxTopLogprobs:      maxTop,
		MaxSelectedTokenIDs: maxSelected,
		PrefixCache:         "automatic",
		ReportsCachedTokens: true,
		MaxConcurrency:      cfg.Concurrency,
	}
	return a
}

func (a *VLLMAdapter) ID() string                             { return a.cfg.ID }
func (a *VLLMAdapter) Engine() config.EngineKind              { return a.cfg.Engine }
func (a *VLLMAdapter) Capabilities() ProviderCapabilities     { return a.caps }
func (a *VLLMAdapter) SetCapabilities(c ProviderCapabilities) { a.caps = c }
func (a *VLLMAdapter) ChatRender() bool                       { return false }

func (a *VLLMAdapter) RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return plan.ChatMessages(plan.Questions[qi])
}

func (a *VLLMAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := a.hc.do(ctx, "POST", "/tokenize", map[string]any{"model": model, "prompt": text}, &out); err != nil {
		return nil, fmt.Errorf("provider: vllm tokenize: %w", err)
	}
	return out.Tokens, nil
}

func (a *VLLMAdapter) ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error) {
	field := a.cfg.SelectedTokenField
	if field == "" {
		field = "logprob_token_ids"
	}
	return scoreViaCompletions(ctx, a.hc, req, req.Model.Model, field)
}

// ScoreContinuations teacher-forces prefix+continuation with prompt_logprobs
// (TODO.md §3.11 strategy 3/5; vLLM supports input-token logprobs).
func (a *VLLMAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	prefixIDs := req.PrefixTokenIDs
	var err error
	if len(prefixIDs) == 0 {
		prefixIDs, err = a.Tokenize(ctx, req.Model.Model, req.PrefixText)
		if err != nil {
			return nil, err
		}
	}
	res := &ContinuationScoreResult{ScoringMethod: "teacher-forced-choice-text"}

	scoreOne := func(text string, ids []int, baseLen int, uncond bool) (ContinuationScore, error) {
		body := openaiCompletionsRequest{
			Model:       req.Model.Model,
			Prompt:      ids,
			MaxTokens:   1, // vLLM requires >= 1; output token is ignored
			Temperature: 1,
			TopP:        1,
			Stream:      false,
			Extra: map[string]any{
				"prompt_logprobs": 0,
			},
		}
		var out openaiResponse
		if err := a.hc.do(ctx, "POST", "/v1/completions", body, &out); err != nil {
			return ContinuationScore{}, err
		}
		if len(out.Choices) == 0 {
			return ContinuationScore{}, fmt.Errorf("provider: vllm: empty choices")
		}
		perToken, positions, err := parsePromptLogprobs(out.Choices[0].PromptLogprobs)
		if err != nil {
			return ContinuationScore{}, err
		}
		_ = positions
		// Healed suffix: everything past the stable prefix boundary.
		start := baseLen
		if start > len(perToken) {
			start = len(perToken)
		}
		var sum float64
		for i := start; i < len(perToken); i++ {
			sum += perToken[i]
		}
		return ContinuationScore{LogProb: sum, TokenLen: len(perToken) - start, HealedFrom: start}, nil
	}

	for _, cont := range req.Continuations {
		full, err := a.Tokenize(ctx, req.Model.Model, req.PrefixText+cont)
		if err != nil {
			return nil, err
		}
		heal := scoring.HealStart(prefixIDs, [][]int{full})
		cs, err := scoreOne(req.PrefixText+cont, full, heal, false)
		if err != nil {
			return nil, err
		}
		res.Conditional = append(res.Conditional, cs)
		res.PromptTokens = len(full)
	}

	if req.Unconditional {
		for _, cont := range req.Continuations {
			full, err := a.Tokenize(ctx, req.Model.Model, cont)
			if err != nil {
				return nil, err
			}
			// Without a prefix the whole continuation is scored.
			cs, err := scoreOne(cont, full, 0, true)
			if err != nil {
				return nil, err
			}
			res.Unconditional = append(res.Unconditional, cs)
		}
	}
	return res, nil
}

// parsePromptLogprobs accepts vLLM's prompt_logprobs list: null for the
// first token and per-token maps keyed by id, or single objects.
func parsePromptLogprobs(raw json.RawMessage) ([]float64, []int, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, fmt.Errorf("provider: no prompt_logprobs in response")
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, nil, fmt.Errorf("provider: parse prompt_logprobs: %w", err)
	}
	logps := make([]float64, 0, len(arr))
	ids := make([]int, 0, len(arr))
	for _, el := range arr {
		if len(el) == 0 || string(el) == "null" {
			logps = append(logps, 0)
			ids = append(ids, -1)
			continue
		}
		// Try map[string]object first.
		var m map[string]struct {
			Logprob float64 `json:"logprob"`
			Rank    int     `json:"rank"`
		}
		if err := json.Unmarshal(el, &m); err == nil && len(m) > 0 {
			// The realized token is the entry with rank 0 or the single key.
			best := ""
			bestRank := 1 << 30
			for k, v := range m {
				if v.Rank < bestRank {
					bestRank = v.Rank
					best = k
				}
			}
			logps = append(logps, m[best].Logprob)
			if id, err := parseInt(best); err == nil {
				ids = append(ids, id)
			} else {
				ids = append(ids, -1)
			}
			continue
		}
		var s struct {
			Logprob float64 `json:"logprob"`
			TokenID int     `json:"token_id"`
		}
		if err := json.Unmarshal(el, &s); err == nil {
			logps = append(logps, s.Logprob)
			ids = append(ids, s.TokenID)
			continue
		}
		return nil, nil, fmt.Errorf("provider: unrecognized prompt_logprob element")
	}
	return logps, ids, nil
}

func parseInt(s string) (int, error) {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

func (a *VLLMAdapter) Health(ctx context.Context) (Health, error) {
	var out struct {
		Version string `json:"version"`
	}
	if err := a.hc.do(ctx, "GET", "/version", nil, &out); err != nil {
		return Health{OK: false, Detail: err.Error()}, err
	}
	a.caps.EngineVersion = out.Version
	return Health{OK: true, Detail: out.Version}, nil
}
