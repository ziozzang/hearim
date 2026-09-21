// Package provider defines the ProviderAdapter abstraction of TODO.md §3.5
// and per-engine adapters (Ollama, llama.cpp, vLLM, SGLang, generic OpenAI).
//
// The external API stays Jev/OpenAI compatible, but internally hearim talks
// to native provider endpoints first: arbitrary token-ID logprob requests,
// tokenizer endpoints, and cache control/observability live outside the
// standard OpenAI contract (§3.5).
package provider

import (
	"context"
	"errors"
	"fmt"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
)

// EndpointProfile describes one usable upstream endpoint (TODO.md §3.4).
type EndpointProfile struct {
	Kind config.EndpointKind
	// NextTokenLogprobsVerified is set by the boot capability probe: the
	// first decode position returns logprobs usable for label scoring.
	NextTokenLogprobsVerified bool
	// Reasoning control fields (§3.4); exact shapes are engine-specific.
	ReasoningControl            string // none-needed|boolean|effort|unsupported
	ReasoningField              string // e.g. "think", "reasoning_effort"
	ReasoningValue              string // e.g. "false", "none"
	ReasoningCanBeFullyDisabled bool
	HiddenReasoningDetected     bool
	MaxTopLogprobs              int
	MaxSelectedTokenIDs         int
}

// ProviderCapabilities is what an adapter reports after discovery/probing
// (TODO.md §3.5).
type ProviderCapabilities struct {
	Engine              config.EngineKind
	EngineVersion       string
	Endpoints           []EndpointProfile
	Tokenizer           string // remote|local-matched|probe-only
	CandidateScoring    []string
	PromptTokenLogprobs bool
	BatchedPrompts      bool
	MaxTopLogprobs      int
	MaxSelectedTokenIDs int
	PrefixCache         string
	ReportsCachedTokens bool
	MaxConcurrency      int
	// Vision: the endpoint accepts OpenAI-style image_url content parts,
	// so image-bearing states can be evaluated by VLM models.
	Vision bool
}

// SupportsSelectedTokenIDs reports the direct candidate-ID logprob path.
func (c ProviderCapabilities) SupportsSelectedTokenIDs() bool {
	for _, s := range c.CandidateScoring {
		if s == "selected-token-ids" {
			return true
		}
	}
	return false
}

// ModelIdentity pins (provider, engine, endpoint, model digest) — the route
// tuple of TODO.md §9.
type ModelIdentity struct {
	Provider string
	Model    string
	Digest   string // resolved model digest, "" when the engine reports none
}

func (m ModelIdentity) String() string {
	if m.Digest != "" {
		return fmt.Sprintf("%s/%s@%s", m.Provider, m.Model, m.Digest)
	}
	return fmt.Sprintf("%s/%s", m.Provider, m.Model)
}

// NextTokenScoreRequest scores candidate next tokens after a prompt
// (TODO.md §3.5). Exactly one decode position, no generated text is used.
type NextTokenScoreRequest struct {
	Model          ModelIdentity
	Endpoint       config.EndpointKind
	PromptText     string
	PromptTokenIDs []int
	// ChatMessages carries the §5.1 chat rendering for chat endpoints; nil
	// on raw completion routes. Chat adapters fall back to splitting
	// PromptText when nil.
	ChatMessages        []compile.ChatMessage
	CandidateTokenIDs   []int
	CandidateTokenTexts []string
	// TopK asks for top-N logprobs when selected-token-ids is unavailable.
	TopK int
	// Sampler parameters are fixed to identity by default (§7.1).
	Temperature float64
	TopP        float64
	// CacheKey groups requests that should hit the same prefix cache.
	CacheKey string
	// NoReasoning forces the model's reasoning off on chat routes.
	NoReasoning bool
	// ReasoningField/ReasoningValue override the engine-default reasoning
	// control with the model-card-documented one (e.g. field "reasoning_effort",
	// value "none"; field "think", value false).
	ReasoningField string
	ReasoningValue any
	// WaitClose scans a bounded generation for CloseTag and scores the first
	// position AFTER it (TODO.md §3.4 <think> technique, approximate: the
	// distribution is conditioned on the sampled reasoning text).
	WaitClose bool
	// CloseTag names the reasoning close marker to scan for (e.g. "</think>").
	CloseTag string
	// MaxOutputTokens bounds WaitClose generation (0 = adapter default).
	MaxOutputTokens int
	// ConstrainToCandidates applies a vocab restriction when supported;
	// the result is then marked post-mask (§7.3 step 5).
	ConstrainToCandidates bool
	// Retry marks an internal retry; adapters may widen parameters.
	Retry int
}

// ProbabilitySpace distinguishes raw vocabulary distributions from
// post-mask ones (§15.2 point 3).
const (
	SpaceRaw      = "raw"
	SpacePostMask = "post-mask"
)

// NextTokenScoreResult carries the restored candidate logprobs.
type NextTokenScoreResult struct {
	CandidateLogprobs    map[int]float64
	SampledTokenID       int
	AllCandidatesPresent bool
	Distribution         string
	PromptTokens         int
	CachedPromptTokens   int
	BackendRequestID     string
	ScoringMethod        string // selected-token-ids|top-k|constrained-vocab
	ProbabilitySpace     string
}

// ContinuationScoreRequest teacher-forces full choice texts (TODO.md §3.11
// strategy 5). No output tokens are generated; input-token logprobs are read.
type ContinuationScoreRequest struct {
	Model          ModelIdentity
	Endpoint       config.EndpointKind
	PrefixText     string
	PrefixTokenIDs []int
	Continuations  []string
	// Unconditional additionally scores each continuation without the
	// prefix, for PMI normalization (§7.3.1).
	Unconditional bool
}

// ContinuationScore is one continuation's healed sequence score.
type ContinuationScore struct {
	LogProb    float64
	TokenLen   int
	HealedFrom int
}

// ContinuationScoreResult returns per-continuation conditional and optional
// unconditional scores.
type ContinuationScoreResult struct {
	Conditional   []ContinuationScore
	Unconditional []ContinuationScore
	PromptTokens  int
	CachedTokens  int
	ScoringMethod string
}

// Health is a lightweight upstream liveness signal.
type Health struct {
	OK     bool
	Detail string
}

// ErrNoExactEvaluationRoute marks models with no endpoint that can deliver
// exact candidate probabilities (TODO.md §3.4).
var ErrNoExactEvaluationRoute = errors.New("provider: no exact evaluation route for model")

// Adapter is the native provider abstraction (TODO.md §3.5). One instance
// wraps one configured provider.
type Adapter interface {
	// ID returns the configured provider id.
	ID() string
	// Engine returns the engine kind.
	Engine() config.EngineKind
	// Capabilities reports probed capabilities.
	Capabilities() ProviderCapabilities
	// Tokenize converts text to backend token IDs. Returns an error when the
	// engine exposes no tokenizer endpoint (probe-only fallback applies).
	Tokenize(ctx context.Context, model, text string) ([]int, error)
	// ScoreNextToken evaluates one decode position for candidate tokens.
	ScoreNextToken(ctx context.Context, req NextTokenScoreRequest) (*NextTokenScoreResult, error)
	// ScoreContinuations teacher-forces full continuations. Returns an error
	// when the engine cannot report prompt-token logprobs.
	ScoreContinuations(ctx context.Context, req ContinuationScoreRequest) (*ContinuationScoreResult, error)
	// Health probes the upstream.
	Health(ctx context.Context) (Health, error)
	// ChatRender returns whether this adapter's exact route uses chat
	// rendering (affects registry probing, §5.1).
	ChatRender() bool
	// RenderChat builds chat messages for a compiled question.
	RenderChat(plan *compile.EvaluationPlan, qi int) []compile.ChatMessage
}

// ResolveExactRoute implements the route resolver of TODO.md §3.4:
//
//  1. verified raw completions
//  2. verified native generate
//  3. verified chat with fully-disableable reasoning and no hidden tokens
//  4. otherwise: no exact route
func ResolveExactRoute(caps ProviderCapabilities, policy config.BackendPolicyConfig) (EndpointProfile, error) {
	find := func(kind config.EndpointKind, needReasoningOff bool) (EndpointProfile, bool) {
		for _, e := range caps.Endpoints {
			if e.Kind != kind || !e.NextTokenLogprobsVerified {
				continue
			}
			if needReasoningOff {
				if !e.ReasoningCanBeFullyDisabled {
					continue
				}
				if e.HiddenReasoningDetected &&
					(policy.RejectHiddenReasoning == nil || *policy.RejectHiddenReasoning) {
					continue
				}
			}
			return e, true
		}
		return EndpointProfile{}, false
	}
	if e, ok := find(config.EndpointCompletions, false); ok {
		return e, nil
	}
	if e, ok := find(config.EndpointNativeGenerate, false); ok {
		return e, nil
	}
	// Chat is only an exact route when reasoning is fully disableable and no
	// hidden reasoning tokens exist (§3.4) — independent of config policy,
	// which can only tighten this further.
	if e, ok := find(config.EndpointChatCompletion, true); ok {
		return e, nil
	}
	return EndpointProfile{}, ErrNoExactEvaluationRoute
}

// RouteScore ranks adapters when the same model is served by several
// engines (TODO.md §3.10).
func RouteScore(caps ProviderCapabilities) int {
	score := 0
	if caps.SupportsSelectedTokenIDs() {
		score += 100
	}
	for _, e := range caps.Endpoints {
		if e.NextTokenLogprobsVerified &&
			(e.Kind == config.EndpointCompletions || e.Kind == config.EndpointNativeGenerate) {
			score += 50
			break
		}
	}
	switch caps.PrefixCache {
	case "automatic", "radix":
		score += 30
	case "request-flag":
		score += 15
	}
	if caps.Tokenizer == "remote" || caps.Tokenizer == "local-matched" {
		score += 20
	}
	if caps.ReportsCachedTokens {
		score += 10
	}
	chatOnly := true
	for _, e := range caps.Endpoints {
		if e.Kind == config.EndpointCompletions || e.Kind == config.EndpointNativeGenerate {
			chatOnly = false
		}
	}
	if chatOnly {
		// §3.10: chat-only -30; an undisableable reasoning chat endpoint
		// disqualifies only that endpoint, so it only kills the score when
		// no other endpoint exists.
		score -= 30
		for _, e := range caps.Endpoints {
			if e.Kind == config.EndpointChatCompletion && !e.ReasoningCanBeFullyDisabled {
				score -= 1000
				break
			}
		}
	}
	return score
}
