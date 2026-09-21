package provider

import (
	"context"
	"fmt"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
)

// OllamaAdapter targets Ollama's OpenAI-compatible surface (TODO.md §3.6:
// top-N logprobs, no tokenizer endpoint assumption, provider-managed cache).
type OllamaAdapter struct {
	cfg  config.ProviderConfig
	hc   *httpClient
	caps ProviderCapabilities
}

// NewOllama builds the adapter; caps may be replaced after boot probing.
//
// Capability reality (probed 2026-09-21, ollama cloud + local 0.24.0):
// /v1/completions accepts `logprobs` (int) but returns no logprobs at all,
// while /v1/chat/completions returns full
// message.logprobs.content[].top_logprobs. The chat endpoint is therefore
// the exact Ollama route; per §3.4 chat requires no-reasoning —
// non-thinking models qualify vacuously and `think: false` is applied for
// thinking models. GPT-OSS on Ollama accepts only low/medium/high effort
// and cannot fully disable reasoning, so it has no exact Ollama route.
func NewOllama(cfg config.ProviderConfig) *OllamaAdapter {
	a := &OllamaAdapter{cfg: cfg, hc: newHTTPClient(cfg)}
	maxTop := cfg.MaxTopLogprobs
	if maxTop == 0 {
		maxTop = 20
	}
	a.caps = ProviderCapabilities{
		Engine: config.EngineOllama,
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointCompletions, MaxTopLogprobs: maxTop,
				NextTokenLogprobsVerified: false},
			{Kind: config.EndpointChatCompletion, MaxTopLogprobs: maxTop,
				NextTokenLogprobsVerified:   true,
				ReasoningControl:            "boolean",
				ReasoningField:              "think",
				ReasoningValue:              "false",
				ReasoningCanBeFullyDisabled: true},
		},
		Tokenizer:           "probe-only",
		CandidateScoring:    []string{"top-k"},
		MaxTopLogprobs:      maxTop,
		PrefixCache:         "automatic",
		ReportsCachedTokens: true,
		MaxConcurrency:      cfg.Concurrency,
	}
	return a
}

func (a *OllamaAdapter) ID() string                             { return a.cfg.ID }
func (a *OllamaAdapter) Engine() config.EngineKind              { return a.cfg.Engine }
func (a *OllamaAdapter) Capabilities() ProviderCapabilities     { return a.caps }
func (a *OllamaAdapter) SetCapabilities(c ProviderCapabilities) { a.caps = c }
func (a *OllamaAdapter) ChatRender() bool                       { return true }

func (a *OllamaAdapter) RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return plan.ChatMessages(plan.Questions[qi])
}

// Tokenize: the OpenAI-compatible surface documents no tokenizer endpoint
// (TODO.md §6.1), so Ollama stays probe-only.
func (a *OllamaAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	// Try the native API opportunistically; ignore failures.
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := a.hc.do(ctx, "POST", "/api/tokenize", map[string]any{"model": model, "content": text}, &out); err == nil && len(out.Tokens) > 0 {
		return out.Tokens, nil
	}
	return nil, fmt.Errorf("provider: ollama %s: no tokenizer endpoint (probe-only)", a.cfg.ID)
}

func (a *OllamaAdapter) ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error) {
	// Chat is the verified logprob surface; raw completions returns none.
	return scoreViaChat(ctx, a.hc, req, req.Model.Model)
}

// ScoreContinuations: the OpenAI-compatible surface exposes no prompt-token
// logprobs, so teacher forcing is unavailable (TODO.md §3.6 row Ollama).
func (a *OllamaAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	return nil, fmt.Errorf("provider: ollama %s: prompt-token logprobs unsupported", a.cfg.ID)
}

func (a *OllamaAdapter) Health(ctx context.Context) (Health, error) {
	var out struct {
		Version string `json:"version"`
	}
	if err := a.hc.do(ctx, "GET", "/api/version", nil, &out); err != nil {
		return Health{OK: false, Detail: err.Error()}, err
	}
	a.caps.EngineVersion = out.Version
	return Health{OK: true, Detail: out.Version}, nil
}

// GenericAdapter targets any OpenAI-compatible server not otherwise known
// (TODO.md §3.6 last row): capability is used only as far as verified.
type GenericAdapter struct {
	cfg  config.ProviderConfig
	hc   *httpClient
	caps ProviderCapabilities
}

func NewGeneric(cfg config.ProviderConfig) *GenericAdapter {
	a := &GenericAdapter{cfg: cfg, hc: newHTTPClient(cfg)}
	maxTop := cfg.MaxTopLogprobs
	if maxTop == 0 {
		maxTop = 20
	}
	a.caps = ProviderCapabilities{
		Engine: config.EngineGeneric,
		Endpoints: []EndpointProfile{
			{Kind: config.EndpointCompletions, MaxTopLogprobs: maxTop},
			{Kind: config.EndpointChatCompletion, MaxTopLogprobs: maxTop, ReasoningControl: "unsupported"},
		},
		Tokenizer:        "probe-only",
		CandidateScoring: []string{"top-k"},
		MaxTopLogprobs:   maxTop,
		PrefixCache:      "unknown",
		MaxConcurrency:   cfg.Concurrency,
	}
	return a
}

func (a *GenericAdapter) ID() string                             { return a.cfg.ID }
func (a *GenericAdapter) Engine() config.EngineKind              { return a.cfg.Engine }
func (a *GenericAdapter) Capabilities() ProviderCapabilities     { return a.caps }
func (a *GenericAdapter) SetCapabilities(c ProviderCapabilities) { a.caps = c }
func (a *GenericAdapter) ChatRender() bool                       { return false }

func (a *GenericAdapter) RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return plan.ChatMessages(plan.Questions[qi])
}

func (a *GenericAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := a.hc.do(ctx, "POST", "/tokenize", map[string]any{"model": model, "prompt": text}, &out); err == nil && len(out.Tokens) > 0 {
		return out.Tokens, nil
	}
	return nil, fmt.Errorf("provider: generic %s: no tokenizer endpoint", a.cfg.ID)
}

func (a *GenericAdapter) ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error) {
	return scoreViaCompletions(ctx, a.hc, req, req.Model.Model, "")
}

func (a *GenericAdapter) ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error) {
	return nil, fmt.Errorf("provider: generic %s: continuation scoring unverified", a.cfg.ID)
}

func (a *GenericAdapter) Health(ctx context.Context) (Health, error) {
	if err := a.hc.do(ctx, "GET", "/models", nil, nil); err != nil {
		return Health{OK: false, Detail: err.Error()}, err
	}
	return Health{OK: true}, nil
}
