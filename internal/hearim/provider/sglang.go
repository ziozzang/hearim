package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/scoring"
)

// SGLangAdapter targets the native /generate endpoint with token_ids_logprob
// (TODO.md §3.9) and RadixAttention prefix caching.
type SGLangAdapter struct {
	cfg  config.ProviderConfig
	hc   *httpClient
	caps ProviderCapabilities
}

func NewSGLang(cfg config.ProviderConfig) *SGLangAdapter {
	a := &SGLangAdapter{cfg: cfg, hc: newHTTPClient(cfg)}
	maxTop := cfg.MaxTopLogprobs
	if maxTop == 0 {
		maxTop = 20
	}
	maxSelected := 256
	a.caps = ProviderCapabilities{
		Engine: config.EngineSGLang,
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointNativeGenerate, MaxTopLogprobs: maxTop, MaxSelectedTokenIDs: maxSelected},
			{Kind: config.EndpointCompletions, MaxTopLogprobs: maxTop},
			{Kind: config.EndpointChatCompletion, MaxTopLogprobs: maxTop,
				ReasoningControl: "unsupported"},
		},
		Tokenizer:           "remote",
		CandidateScoring:    []string{"selected-token-ids", "top-k", "teacher-forced"},
		PromptTokenLogprobs: true,
		BatchedPrompts:      true,
		MaxTopLogprobs:      maxTop,
		MaxSelectedTokenIDs: maxSelected,
		PrefixCache:         "radix",
		ReportsCachedTokens: true,
		MaxConcurrency:      cfg.Concurrency,
		Vision:              true,
	}
	return a
}

func (a *SGLangAdapter) ID() string                             { return a.cfg.ID }
func (a *SGLangAdapter) Engine() config.EngineKind              { return a.cfg.Engine }
func (a *SGLangAdapter) Capabilities() ProviderCapabilities     { return a.caps }
func (a *SGLangAdapter) SetCapabilities(c ProviderCapabilities) { a.caps = c }
func (a *SGLangAdapter) ChatRender() bool                       { return false }

func (a *SGLangAdapter) RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return plan.ChatMessages(plan.Questions[qi])
}

func (a *SGLangAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	var out struct {
		InputIDs []int `json:"input_ids"`
		Tokens   []int `json:"tokens"`
	}
	// Model gateway exposes /v1/tokenize; single runtime may expose /tokenize.
	for _, path := range []string{pathFor(a.cfg, PathTokenize), "/v1/tokenize", "/tokenize"} {
		if err := a.hc.do(ctx, "POST", path, map[string]any{"text": text, "add_special_tokens": true}, &out); err == nil {
			if len(out.InputIDs) > 0 {
				return out.InputIDs, nil
			}
			if len(out.Tokens) > 0 {
				return out.Tokens, nil
			}
		}
	}
	return nil, fmt.Errorf("provider: sglang %s: no tokenizer endpoint", a.cfg.ID)
}

// ScoreNextToken calls /generate with token_ids_logprob for direct candidate
// ID scoring (TODO.md §3.9 request shape).
func (a *SGLangAdapter) ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error) {
	if req.WaitClose {
		return nil, fmt.Errorf("provider: sglang: thinking.wait_close unsupported; use thinking.close_tag preload")
	}
	input := any(req.PromptText)
	if len(req.PromptTokenIDs) > 0 {
		input = req.PromptTokenIDs
	}
	body := map[string]any{
		"sampling_params": map[string]any{
			"max_new_tokens": 1,
			"temperature":    tempOrOne(req.Temperature),
			"top_p":          topPOrOne(req.TopP),
			"top_k":          -1,
		},
		"return_logprob":   true,
		"top_logprobs_num": 0,
		"stream":           false,
	}
	if ids, ok := input.([]int); ok {
		body["input_ids"] = ids
	} else {
		body["text"] = input
	}
	if req.Model.Model != "" {
		body["model"] = req.Model.Model
	}
	method := "top-k"
	space := SpaceRaw
	if len(req.CandidateTokenIDs) > 0 {
		body["token_ids_logprob"] = req.CandidateTokenIDs
		method = "selected-token-ids"
		if req.ConstrainToCandidates {
			// Restrict sampling to the candidate set via allowed_token_ids.
			body["sampling_params"].(map[string]any)["allowed_token_ids"] = req.CandidateTokenIDs
			space = SpacePostMask
			method = "constrained-vocab"
		}
	} else if req.TopK > 0 {
		body["top_logprobs_num"] = req.TopK
	}
	MergeExtras(body, ModelExtras(a.cfg, req.Model.Model))

	var raw json.RawMessage
	if err := a.hc.do(ctx, "POST", pathFor(a.cfg, PathGenerate), body, &raw); err != nil {
		return nil, err
	}
	return parseSGLangScore(raw, req.CandidateTokenIDs, req.CandidateTokenTexts, method, space)
}

// parseSGLangScore handles the /generate response with flexible field names
// across SGLang versions: logprobs.output_token_ids_logprobs (per-position
// candidate lists) or output_top_logprobs (entries with token ids).
func parseSGLangScore(raw json.RawMessage, ids []int, texts []string, method, space string) (*NextTokenScoreResult, error) {
	var resp struct {
		LogProbs *struct {
			OutputTokenIDsLogprobs [][]json.RawMessage `json:"output_token_ids_logprobs"`
			OutputTopLogprobs      [][]json.RawMessage `json:"output_top_logprobs"`
			OutputTokenLogprobs    []float64           `json:"output_token_logprobs"`
		} `json:"logprobs"`
		MetaInfo *struct {
			PromptTokens     int    `json:"prompt_tokens"`
			CachedTokens     int    `json:"cached_tokens"`
			CompletionTokens int    `json:"completion_tokens"`
			ReqID            string `json:"req_id"`
		} `json:"meta_info"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("provider: sglang decode: %w", err)
	}
	if resp.LogProbs == nil {
		return nil, fmt.Errorf("provider: sglang: no logprobs in response")
	}

	// Prefer candidate-ID entries at the first decode position.
	var entries []openaiLogprobEntry
	if len(resp.LogProbs.OutputTokenIDsLogprobs) > 0 {
		pos := resp.LogProbs.OutputTokenIDsLogprobs[0]
		for _, el := range pos {
			var arr []openaiLogprobEntry
			if err := json.Unmarshal(el, &arr); err == nil {
				entries = append(entries, arr...)
				continue
			}
			var one openaiLogprobEntry
			if err := json.Unmarshal(el, &one); err == nil {
				entries = append(entries, one)
			}
		}
	}
	if len(entries) == 0 && len(resp.LogProbs.OutputTopLogprobs) > 0 {
		for _, el := range resp.LogProbs.OutputTopLogprobs[0] {
			var one openaiLogprobEntry
			if err := json.Unmarshal(el, &one); err == nil {
				entries = append(entries, one)
			}
		}
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("provider: sglang: no usable logprob entries")
	}
	logprobs, all := matchCandidates(entries, ids, texts)
	res := &NextTokenScoreResult{
		CandidateLogprobs:    logprobs,
		AllCandidatesPresent: all,
		Distribution:         "raw",
		ScoringMethod:        method,
		ProbabilitySpace:     space,
	}
	if resp.MetaInfo != nil {
		res.PromptTokens = resp.MetaInfo.PromptTokens
		res.CachedPromptTokens = resp.MetaInfo.CachedTokens
		res.BackendRequestID = resp.MetaInfo.ReqID
	}
	return res, nil
}

// ScoreContinuations teacher-forces full continuations via input-token
// logprobs (TODO.md §15: SGLang's select pattern — prefill the common prompt,
// then score suffixes).
func (a *SGLangAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	prefixIDs := req.PrefixTokenIDs
	var err error
	if len(prefixIDs) == 0 {
		prefixIDs, err = a.Tokenize(ctx, req.Model.Model, req.PrefixText)
		if err != nil {
			return nil, err
		}
	}
	res := &ContinuationScoreResult{ScoringMethod: "teacher-forced-choice-text"}

	scoreOne := func(text string, ids []int, start int) (ContinuationScore, error) {
		body := map[string]any{
			"input_ids":       ids,
			"sampling_params": map[string]any{"max_new_tokens": 1, "temperature": 1, "top_p": 1, "top_k": -1},
			"return_logprob":  true,
			// Return input-token logprobs from position 0.
			"logprob_start_len": 0,
			"top_logprobs_num":  0,
			"stream":            false,
		}
		if req.Model.Model != "" {
			body["model"] = req.Model.Model
		}
		MergeExtras(body, ModelExtras(a.cfg, req.Model.Model))
		var raw json.RawMessage
		if err := a.hc.do(ctx, "POST", pathFor(a.cfg, PathGenerate), body, &raw); err != nil {
			return ContinuationScore{}, err
		}
		var resp struct {
			LogProbs *struct {
				InputTokenLogprobs []json.RawMessage `json:"input_token_logprobs"`
			} `json:"logprobs"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return ContinuationScore{}, err
		}
		if resp.LogProbs == nil {
			return ContinuationScore{}, fmt.Errorf("provider: sglang: no input logprobs")
		}
		var sum float64
		n := 0
		for i := start; i < len(resp.LogProbs.InputTokenLogprobs); i++ {
			el := resp.LogProbs.InputTokenLogprobs[i]
			if len(el) == 0 || string(el) == "null" {
				continue
			}
			var f float64
			if err := json.Unmarshal(el, &f); err == nil {
				sum += f
				n++
				continue
			}
			var s struct {
				Logprob float64 `json:"logprob"`
			}
			if err := json.Unmarshal(el, &s); err == nil && s.Logprob != 0 {
				sum += s.Logprob
				n++
			}
		}
		return ContinuationScore{LogProb: sum, TokenLen: n, HealedFrom: start}, nil
	}

	for _, cont := range req.Continuations {
		full, err := a.Tokenize(ctx, req.Model.Model, req.PrefixText+cont)
		if err != nil {
			return nil, err
		}
		heal := scoring.HealStart(prefixIDs, [][]int{full})
		cs, err := scoreOne(req.PrefixText+cont, full, heal)
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
			cs, err := scoreOne(cont, full, 0)
			if err != nil {
				return nil, err
			}
			res.Unconditional = append(res.Unconditional, cs)
		}
	}
	return res, nil
}

func (a *SGLangAdapter) Health(ctx context.Context) (Health, error) {
	var out struct {
		Version []any  `json:"version"`
		Health  string `json:"health"`
	}
	if err := a.hc.do(ctx, "GET", pathFor(a.cfg, PathHealth), nil, &out); err != nil {
		return Health{OK: false, Detail: err.Error()}, err
	}
	return Health{OK: true}, nil
}
