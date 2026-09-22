package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

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
	var err error
	req, err = prepareScoring(a.cfg, req)
	if err != nil {
		return nil, err
	}
	if req.Endpoint == config.EndpointChatCompletion {
		// The OpenAI surface is top-k only here; native selected-ID fields
		// are not assumed to be portable across SGLang endpoint versions.
		req.SelectedTokenField = ""
		return scoreViaChat(ctx, a.hc, req, req.Model.Model, ModelExtras(a.cfg, req.Model.Model), pathFor(a.cfg, PathChatCompletions))
	}
	if req.Endpoint == config.EndpointCompletions {
		return scoreViaCompletions(ctx, a.hc, req, req.Model.Model, "", ModelExtras(a.cfg, req.Model.Model), pathFor(a.cfg, PathCompletions))
	}
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
		"return_logprob":          true,
		"return_text_in_logprobs": !validCandidateIDs(req),
		"top_logprobs_num":        0,
		"stream":                  false,
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
	if req.SelectedTokenField != "" {
		body["token_ids_logprob"] = req.CandidateTokenIDs
		method = "selected-token-ids"
	} else if req.TopK > 0 {
		body["top_logprobs_num"] = req.TopK
	} else {
		body["top_logprobs_num"] = a.caps.MaxTopLogprobs
	}
	MergeExtras(body, ModelExtras(a.cfg, req.Model.Model))

	var raw json.RawMessage
	if err := a.hc.do(ctx, "POST", pathFor(a.cfg, PathGenerate), body, &raw); err != nil {
		return nil, err
	}
	return parseSGLangScore(raw, req.CandidateTokenIDs, req.CandidateTokenTexts, method, space)
}

// Native /generate returns (logprob, token_id, text-or-null) tuples under
// meta_info. Retain the object envelope for older gateway integrations.
type sglangLogprobs struct {
	OutputTokenIDsLogprobs [][]json.RawMessage `json:"output_token_ids_logprobs"`
	OutputTopLogprobs      [][]json.RawMessage `json:"output_top_logprobs"`
	InputTokenLogprobs     []json.RawMessage   `json:"input_token_logprobs"`
}

type sglangResponse struct {
	Text     string          `json:"text"`
	LogProbs *sglangLogprobs `json:"logprobs"`
	MetaInfo struct {
		sglangLogprobs
		PromptTokens int    `json:"prompt_tokens"`
		CachedTokens int    `json:"cached_tokens"`
		ID           string `json:"id"`
		ReqID        string `json:"req_id"`
	} `json:"meta_info"`
}

func (r *sglangResponse) logprobs() *sglangLogprobs {
	if r.MetaInfo.OutputTokenIDsLogprobs != nil || r.MetaInfo.OutputTopLogprobs != nil || r.MetaInfo.InputTokenLogprobs != nil {
		return &r.MetaInfo.sglangLogprobs
	}
	return r.LogProbs
}

func parseSGLangEntry(raw json.RawMessage) (openaiLogprobEntry, error) {
	var tuple []json.RawMessage
	if json.Unmarshal(raw, &tuple) == nil && len(tuple) >= 2 {
		var lp *float64
		var id *int64
		if err := json.Unmarshal(tuple[0], &lp); err != nil {
			return openaiLogprobEntry{}, err
		}
		if err := json.Unmarshal(tuple[1], &id); err != nil {
			return openaiLogprobEntry{}, err
		}
		if lp == nil || id == nil || *id < 0 || math.IsNaN(*lp) || math.IsInf(*lp, 0) {
			return openaiLogprobEntry{}, fmt.Errorf("provider: invalid SGLang logprob tuple")
		}
		e := openaiLogprobEntry{Logprob: *lp, TokenID: id}
		if len(tuple) > 2 && string(tuple[2]) != "null" {
			if err := json.Unmarshal(tuple[2], &e.Token); err != nil {
				return e, err
			}
		}
		return e, nil
	}
	var e openaiLogprobEntry
	var fields struct {
		Logprob *float64 `json:"logprob"`
	}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return e, err
	}
	if fields.Logprob == nil {
		return e, fmt.Errorf("provider: missing SGLang logprob")
	}
	if err := json.Unmarshal(raw, &e); err != nil {
		return e, err
	}
	return e, nil
}

func parseSGLangScore(raw json.RawMessage, ids []int, texts []string, method, space string) (*NextTokenScoreResult, error) {
	var resp sglangResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("provider: sglang decode: %w", err)
	}
	lps := resp.logprobs()
	if lps == nil {
		return nil, fmt.Errorf("provider: sglang: no logprobs in response")
	}

	// Prefer candidate-ID entries at the first decode position.
	var entries []openaiLogprobEntry
	var pos []json.RawMessage
	if len(lps.OutputTokenIDsLogprobs) > 0 {
		pos = lps.OutputTokenIDsLogprobs[0]
	}
	if len(pos) == 0 && len(lps.OutputTopLogprobs) > 0 {
		pos = lps.OutputTopLogprobs[0]
	}
	for _, el := range pos {
		one, err := parseSGLangEntry(el)
		if err != nil {
			return nil, fmt.Errorf("provider: SGLang logprob entry: %w", err)
		}
		entries = append(entries, one)
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
	res.PromptTokens = resp.MetaInfo.PromptTokens
	res.CachedPromptTokens = resp.MetaInfo.CachedTokens
	res.BackendRequestID = firstNonEmpty(resp.MetaInfo.ID, resp.MetaInfo.ReqID)
	res.GeneratedText = resp.Text
	return res, nil
}

// ScoreContinuations teacher-forces full continuations via input-token
// logprobs (TODO.md §15: SGLang's select pattern — prefill the common prompt,
// then score suffixes).
func (a *SGLangAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	if !resolveScoring(a.cfg, req.Model.Model).promptLogprobs {
		return nil, fmt.Errorf("provider: prompt token logprobs disabled for %s", req.Model.Model)
	}
	if req.Endpoint == config.EndpointChatCompletion || req.Endpoint == config.EndpointCompletions {
		return nil, fmt.Errorf("provider: SGLang continuation scoring requires native_generate")
	}
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
		var resp sglangResponse
		if err := json.Unmarshal(raw, &resp); err != nil {
			return ContinuationScore{}, err
		}
		lps := resp.logprobs()
		if lps == nil || len(lps.InputTokenLogprobs) != len(ids) || start >= len(ids) {
			return ContinuationScore{}, fmt.Errorf("provider: sglang: no input logprobs")
		}
		var sum float64
		n := 0
		for i := start; i < len(lps.InputTokenLogprobs); i++ {
			// The first prompt position has no conditional logprob (BOS).
			if i == 0 {
				continue
			}
			e, err := parseSGLangEntry(lps.InputTokenLogprobs[i])
			if err != nil {
				return ContinuationScore{}, err
			}
			if e.TokenID != nil && int(*e.TokenID) != ids[i] {
				return ContinuationScore{}, fmt.Errorf("provider: SGLang prompt token ID mismatch at %d", i)
			}
			sum += e.Logprob
			n++
		}
		if n == 0 {
			return ContinuationScore{}, fmt.Errorf("provider: no scored continuation tokens")
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
