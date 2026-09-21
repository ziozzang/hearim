package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
)

// LlamaCppAdapter targets the native llama.cpp server /completion and
// /tokenize endpoints (TODO.md §3.7): token-ID prompts, n_probs, cache_prompt
// and slot-affine cache observability.
type LlamaCppAdapter struct {
	cfg  config.ProviderConfig
	hc   *httpClient
	caps ProviderCapabilities
}

func NewLlamaCpp(cfg config.ProviderConfig) *LlamaCppAdapter {
	a := &LlamaCppAdapter{cfg: cfg, hc: newHTTPClient(cfg)}
	maxTop := cfg.MaxTopLogprobs
	if maxTop == 0 {
		maxTop = 20
	}
	a.caps = ProviderCapabilities{
		Engine: config.EngineLlamaPP,
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointNativeGenerate, MaxTopLogprobs: maxTop},
			{Kind: config.EndpointCompletions, MaxTopLogprobs: maxTop},
		},
		Tokenizer: "remote",
		// raw-top-k and constrained-vocab via grammar (§3.7); llama.cpp does
		// not offer selected-ID logprobs, and prompt-token logprobs are not
		// part of the documented native contract.
		CandidateScoring:    []string{"top-k", "constrained-vocab"},
		MaxTopLogprobs:      maxTop,
		PrefixCache:         "request-flag",
		ReportsCachedTokens: true,
		MaxConcurrency:      cfg.Concurrency,
	}
	return a
}

func (a *LlamaCppAdapter) ID() string                             { return a.cfg.ID }
func (a *LlamaCppAdapter) Engine() config.EngineKind              { return a.cfg.Engine }
func (a *LlamaCppAdapter) Capabilities() ProviderCapabilities     { return a.caps }
func (a *LlamaCppAdapter) SetCapabilities(c ProviderCapabilities) { a.caps = c }
func (a *LlamaCppAdapter) ChatRender() bool                       { return false }

func (a *LlamaCppAdapter) RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return plan.ChatMessages(plan.Questions[qi])
}

func (a *LlamaCppAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	var out struct {
		Tokens []int `json:"tokens"`
	}
	body := map[string]any{
		"content":       text,
		"add_special":   true,
		"parse_special": true,
	}
	if err := a.hc.do(ctx, "POST", pathFor(a.cfg, PathTokenize), body, &out); err != nil {
		return nil, fmt.Errorf("provider: llama.cpp tokenize: %w", err)
	}
	return out.Tokens, nil
}

// llamaTopLogprob is llama.cpp's top_logprobs entry: id, content/token/text
// and prob (a probability, not a logprob).
type llamaTopLogprob struct {
	ID      *int            `json:"id"`
	TokenID *int            `json:"token_id"`
	Content string          `json:"content"`
	Token   string          `json:"token"`
	Text    string          `json:"text"`
	Bytes   json.RawMessage `json:"bytes"`
	Prob    *float64        `json:"prob"`
	Logprob *float64        `json:"logprob"`
}

func (e llamaTopLogprob) toOpenAI() openaiLogprobEntry {
	out := openaiLogprobEntry{
		Token: firstNonEmpty(e.Content, e.Token, e.Text),
		Bytes: e.Bytes,
	}
	if e.ID != nil {
		out.TokenID = int64ptr(int64(*e.ID))
	} else if e.TokenID != nil {
		out.TokenID = int64ptr(int64(*e.TokenID))
	}
	if e.Logprob != nil {
		out.Logprob = *e.Logprob
	} else if e.Prob != nil {
		out.Logprob = logSafe(*e.Prob)
	}
	return out
}

func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}

func (a *LlamaCppAdapter) ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error) {
	if req.WaitClose {
		return nil, fmt.Errorf("provider: llama.cpp: thinking.wait_close unsupported; use thinking.close_tag preload")
	}
	k := req.TopK
	if k <= 0 {
		k = 20
	}
	prompt := any(req.PromptText)
	if len(req.PromptTokenIDs) > 0 {
		prompt = req.PromptTokenIDs
	}
	method := "top-k"
	space := SpaceRaw
	body := map[string]any{
		"prompt":        prompt,
		"n_predict":     1,
		"temperature":   tempOrOne(req.Temperature),
		"top_p":         topPOrOne(req.TopP),
		"top_k":         0,
		"n_probs":       k,
		"min_keep":      k,
		"return_tokens": true,
		"cache_prompt":  cachePromptOn(a.cfg),
		"stream":        false,
	}
	if req.ConstrainToCandidates && len(req.CandidateTokenTexts) > 0 {
		// Grammar restricted to the label set: first token must be one of the
		// candidate labels. Post-sampling distribution is post-mask (§3.7).
		body["grammar"] = labelGrammar(req.CandidateTokenTexts)
		method = "constrained-vocab"
		space = SpacePostMask
	}
	MergeExtras(body, ModelExtras(a.cfg, req.Model.Model))

	var out struct {
		CompletionProbabilities []struct {
			ID          int               `json:"id"`
			Content     string            `json:"content"`
			Prob        float64           `json:"prob"`
			TopLogprobs []llamaTopLogprob `json:"top_logprobs"`
		} `json:"completion_probabilities"`
		TokensCached    int `json:"tokens_cached"`
		TokensEvaluated int `json:"tokens_evaluated"`
		TokensPredicted int `json:"n_predict"`
	}
	if err := a.hc.do(ctx, "POST", pathFor(a.cfg, PathCompletion), body, &out); err != nil {
		return nil, err
	}
	if len(out.CompletionProbabilities) == 0 {
		return nil, fmt.Errorf("provider: llama.cpp: no completion_probabilities")
	}
	entries := make([]openaiLogprobEntry, 0, len(out.CompletionProbabilities[0].TopLogprobs))
	for _, e := range out.CompletionProbabilities[0].TopLogprobs {
		entries = append(entries, e.toOpenAI())
	}
	// The sampled token itself counts as an entry if not in top_logprobs.
	sampled := out.CompletionProbabilities[0].ID
	if sampled != 0 {
		present := false
		for _, e := range entries {
			if e.TokenID != nil && int(*e.TokenID) == sampled {
				present = true
			}
		}
		if !present {
			e := openaiLogprobEntry{Token: out.CompletionProbabilities[0].Content, Logprob: logSafe(out.CompletionProbabilities[0].Prob), TokenID: int64ptr(int64(sampled))}
			entries = append(entries, e)
		}
	}
	logprobs, all := matchCandidates(entries, req.CandidateTokenIDs, req.CandidateTokenTexts)
	res := &NextTokenScoreResult{
		CandidateLogprobs:    logprobs,
		AllCandidatesPresent: all,
		Distribution:         "raw",
		PromptTokens:         out.TokensEvaluated,
		CachedPromptTokens:   out.TokensCached,
		ScoringMethod:        method,
		ProbabilitySpace:     space,
		SampledTokenID:       sampled,
	}
	return res, nil
}

func int64ptr(i int64) *int64 { return &i }

func cachePromptOn(cfg config.ProviderConfig) bool {
	return cfg.CachePrompt == nil || *cfg.CachePrompt
}

// labelGrammar builds a GBNF grammar accepting exactly one of the candidate
// label texts as the first token sequence (TODO.md §3.7 constrained-vocab).
func labelGrammar(labels []string) string {
	root := "root ::= "
	alts := make([]string, 0, len(labels))
	for i, l := range labels {
		alts = append(alts, fmt.Sprintf(`"%s"`, l))
		_ = i
	}
	return root + joinAlternatives(alts)
}

func joinAlternatives(alts []string) string {
	out := ""
	for i, a := range alts {
		if i > 0 {
			out += " | "
		}
		out += a
	}
	return out
}

// ScoreContinuations: llama.cpp's documented native API does not expose
// prompt-token logprobs, so teacher forcing is unavailable (§3.7).
func (a *LlamaCppAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	return nil, fmt.Errorf("provider: llama.cpp %s: prompt-token logprobs not in documented API", a.cfg.ID)
}

func (a *LlamaCppAdapter) Health(ctx context.Context) (Health, error) {
	var out struct {
		BuildInfo map[string]string `json:"build_info"`
		Status    string            `json:"status"`
	}
	if err := a.hc.do(ctx, "GET", pathFor(a.cfg, PathHealth), nil, &out); err != nil {
		return Health{OK: false, Detail: err.Error()}, err
	}
	return Health{OK: true, Detail: out.Status}, nil
}
