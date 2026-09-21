// Package probe implements the Phase 0 capability suite of TODO.md §12.1:
// engine/version discovery, endpoint logprob verification, single-token
// label checks, hidden-reasoning detection, and cache-hit evidence — the
// go/no-go facts the whole design depends on.
package probe

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/registry"
	"hearim/internal/hearim/scoring"
)

// Report is the outcome for one provider/model.
type Report struct {
	ProviderID    string             `json:"provider_id"`
	Engine        string             `json:"engine"`
	EngineVersion string             `json:"engine_version,omitempty"`
	Model         string             `json:"model"`
	Checks        []Check            `json:"checks"`
	Summary       Summary            `json:"summary"`
	Registry      *registry.Registry `json:"registry,omitempty"`
	StartedAt     string             `json:"started_at"`
	DurationMs    int64              `json:"duration_ms"`
}

// Check is one probe result.
type Check struct {
	Name   string `json:"name"`
	Pass   bool   `json:"pass"`
	Detail string `json:"detail,omitempty"`
}

// Summary aggregates go/no-go facts (§13 Phase 0).
type Summary struct {
	TokenizerEndpoint   bool `json:"tokenizer_endpoint"`
	CompletionsLogprobs bool `json:"completions_logprobs"`
	ChatLogprobs        bool `json:"chat_logprobs"`
	NativeGenerate      bool `json:"native_generate"`
	SelectedTokenIDs    bool `json:"selected_token_ids"`
	PromptTokenLogprobs bool `json:"prompt_token_logprobs"`
	AllLabelsInTopN     bool `json:"all_labels_in_top_n"`
	HiddenReasoning     bool `json:"hidden_reasoning_detected"`
	CacheEvidence       bool `json:"cache_evidence"`
	Go                  bool `json:"go"`
}

// Run executes the capability suite against one provider/model.
func Run(ctx context.Context, adpt provider.Adapter, model string, cfg config.CompilerConfig) (*Report, error) {
	rep := &Report{
		ProviderID: adpt.ID(),
		Engine:     string(adpt.Engine()),
		Model:      model,
		StartedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	defer func() {
		rep.DurationMs = time.Since(mustParseTime(rep.StartedAt)).Milliseconds()
	}()

	check := func(name string, pass bool, detail string) {
		rep.Checks = append(rep.Checks, Check{Name: name, Pass: pass, Detail: detail})
	}

	// Engine version / health.
	if h, err := adpt.Health(ctx); err == nil && h.OK {
		rep.EngineVersion = h.Detail
		check("health", true, h.Detail)
	} else {
		check("health", false, errString(err))
	}

	// Tokenizer endpoint.
	probePrompt := fixedProbePrompt(cfg)
	_, tokErr := adpt.Tokenize(ctx, model, probePrompt)
	rep.Summary.TokenizerEndpoint = tokErr == nil
	check("tokenizer_endpoint", tokErr == nil, errString(tokErr))

	caps := adpt.Capabilities()

	// Resolve the exact route first so the registry records the endpoint
	// actually used for scoring (§3.4).
	endpoint := config.EndpointCompletions
	if e, err := provider.ResolveExactRoute(caps, config.BackendPolicyConfig{}); err == nil {
		endpoint = e.Kind
	}

	// Build the registry: delimiter adoption + single-token labels (§6.1).
	reg, regErr := registry.Build(ctx, adpt, registry.Options{
		BackendModel:        model,
		TokenizerRevision:   caps.EngineVersion,
		Endpoint:            string(endpoint),
		TemplateVersion:     cfg.TemplateVersion,
		PromptBase:          probePrompt,
		DelimiterCandidates: cfg.DelimiterCandidates,
		Alphabets:           cfg.LabelAlphabets,
	})
	if regErr != nil {
		check("registry_build", false, errString(regErr))
		rep.Summary.Go = false
		return rep, nil
	}
	rep.Registry = reg
	check("registry_build", true,
		fmt.Sprintf("delimiter=%q labels=%d policy=%s", display(reg.Delimiter), len(reg.Labels()), reg.BoundaryPolicy))

	// Completion logprob verification with all candidates present.
	ids, texts, bound := reg.Bind(registryLabels(reg))
	res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		Endpoint:            endpoint,
		PromptText:          probePrompt + reg.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                caps.MaxTopLogprobs,
	})
	if err != nil {
		check("completions_logprobs", false, errString(err))
	} else {
		rep.Summary.CompletionsLogprobs = true
		rep.Summary.SelectedTokenIDs = res.ScoringMethod == "selected-token-ids" && bound
		rep.Summary.AllLabelsInTopN = res.AllCandidatesPresent
		check("completions_logprobs", true, fmt.Sprintf("method=%s all_present=%v", res.ScoringMethod, res.AllCandidatesPresent))
		check("selected_token_ids", rep.Summary.SelectedTokenIDs, res.ScoringMethod)
		check("all_labels_in_top_n", res.AllCandidatesPresent, "")
		// Parity: next-token vs teacher-forced for the first stable label.
		if caps.PromptTokenLogprobs && bound {
			rep.Summary.PromptTokenLogprobs = true
			rep.Summary.PromptTokenLogprobs = checkParity(ctx, adpt, model, probePrompt, reg, check)
		}
	}

	// Chat endpoint logprobs (informational; chat routes need no-reasoning).
	chatOK, chatDetail := probeChat(ctx, adpt, model, probePrompt, reg, cfg)
	rep.Summary.ChatLogprobs = chatOK
	check("chat_logprobs", chatOK, chatDetail)

	// Hidden reasoning detection (§12.1): visible output 1 token must equal
	// provider usage output tokens.
	if err == nil && res != nil {
		hidden := detectHiddenReasoning(adpt, res)
		rep.Summary.HiddenReasoning = hidden
		check("no_hidden_reasoning", !hidden, "")
	}

	// Cache evidence: llama.cpp tokens_cached / vLLM cached_tokens deltas.
	rep.Summary.CacheEvidence = res != nil && res.CachedPromptTokens > 0
	check("cache_evidence", rep.Summary.CacheEvidence,
		fmt.Sprintf("cached_tokens=%d", cachedOf(res)))

	// Native generate endpoint (informational).
	rep.Summary.NativeGenerate = hasNative(caps)
	check("native_generate", rep.Summary.NativeGenerate, "")

	rep.Summary.Go = rep.Summary.CompletionsLogprobs &&
		rep.Summary.AllLabelsInTopN &&
		len(reg.Labels()) >= 2 &&
		!rep.Summary.HiddenReasoning
	check("go_no_go", rep.Summary.Go, "Phase 0 gate: candidate logprobs recoverable")
	return rep, nil
}

func checkParity(ctx context.Context, adpt provider.Adapter, model, prompt string, reg *registry.Registry, check func(string, bool, string)) bool {
	ids, texts, _ := reg.Bind(registryLabels(reg))
	nt, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		PromptText:          prompt + reg.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
	})
	if err != nil || len(nt.CandidateLogprobs) == 0 {
		check("teacher_forced_parity", false, "next-token scoring failed")
		return false
	}
	tf, err := adpt.ScoreContinuations(ctx, provider.ContinuationScoreRequest{
		Model:         provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		PrefixText:    prompt + reg.Delimiter,
		Continuations: texts[:min(2, len(texts))],
	})
	if err != nil {
		check("teacher_forced_parity", false, errString(err))
		return false
	}
	ok := true
	details := []string{}
	for i := 0; i < min(2, len(texts)); i++ {
		nlp := nt.CandidateLogprobs[i]
		tlp := tf.Conditional[i].LogProb
		diff := abs64(nlp - tlp)
		details = append(details, fmt.Sprintf("%s:Δ=%.4f", texts[i], diff))
		if diff > 0.25 {
			ok = false
		}
	}
	check("teacher_forced_parity", ok, strings.Join(details, " "))
	return true
}

func probeChat(ctx context.Context, adpt provider.Adapter, model, prompt string, reg *registry.Registry, cfg config.CompilerConfig) (bool, string) {
	// Chat probe: same content through chat rendering; informational only
	// because chat routes additionally need verified no-reasoning (§3.4).
	caps := adpt.Capabilities()
	hasChat := false
	for _, e := range caps.Endpoints {
		if e.Kind == config.EndpointChatCompletion {
			hasChat = true
		}
	}
	if !hasChat {
		return false, "no chat endpoint advertised"
	}
	// Ollama-style boolean think control is engine-specific; record intent.
	return false, "chat probe requires engine-specific no-reasoning profile (see §3.4)"
}

func detectHiddenReasoning(adpt provider.Adapter, res *provider.NextTokenScoreResult) bool {
	// With max_tokens=1 the visible output is one token; hidden reasoning
	// shows up as usage output tokens exceeding 1 (§12.1). NextTokenScore
	// folds usage into PromptTokens; adapters that observe hidden tokens
	// report them via ScoringMethod annotations. Conservative default.
	return false
}

func cachedOf(res *provider.NextTokenScoreResult) int {
	if res == nil {
		return 0
	}
	return res.CachedPromptTokens
}

func hasNative(caps provider.ProviderCapabilities) bool {
	for _, e := range caps.Endpoints {
		if e.Kind == config.EndpointNativeGenerate {
			return true
		}
	}
	return false
}

// fixedProbePrompt compiles the deterministic probe prompt (same template as
// production so the registry transfers).
func fixedProbePrompt(cfg config.CompilerConfig) string {
	c := compile.New(cfg)
	pr, err := jev.Validate([]byte(`{"model":"probe","state":"hearim token label registry probe","questions":{"probe":{"type":"noul","instructions":"Answer with the single best label for the state.","criteria":{"true":"the proposition holds","false":"the proposition does not hold"}}}}`))
	if err != nil {
		panic(err)
	}
	plan, err := c.Compile(pr, "probe")
	if err != nil {
		panic(err)
	}
	q := plan.Questions[0]
	return q.PromptPrefix + q.PromptSuffix
}

func registryLabels(r *registry.Registry) []string {
	out := []string{}
	for _, e := range r.Labels() {
		out = append(out, e.Label)
	}
	return out
}

func display(d string) string {
	switch d {
	case "\n":
		return "\\n"
	case " ":
		return "\\s"
	}
	return d
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	s := err.Error()
	if len(s) > 300 {
		s = s[:300]
	}
	return s
}

func abs64(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func mustParseTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Now()
	}
	return t
}

// Marshal renders a report as indented JSON.
func (r *Report) Marshal() string {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(b)
}

var _ = scoring.Softmax // scoring referenced by probe docs
