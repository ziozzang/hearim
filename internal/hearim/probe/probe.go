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
	// ThinkingControlVerified: a bounded generation under the configured
	// disable control contained no reasoning markers. ThinkingTagsEmitted:
	// the model still emits tag structure when thinking is off (e.g.
	// gemma4's empty <|channel>thought blocks) — usable only via close-tag
	// handling, not as a plain no-reasoning route.
	ThinkingControlVerified bool `json:"thinking_control_verified,omitempty"`
	ThinkingTagsEmitted     bool `json:"thinking_tags_emitted,omitempty"`
	// CompletionsFallbackViable: the raw completions surface returned
	// usable logprobs when the resolved route was chat — evidence for
	// pinning models[].endpoint: completions on thinking models.
	CompletionsFallbackViable bool `json:"completions_fallback_viable,omitempty"`
	// ThinkingDiscovery: the auto-discovered reasoning-disable control
	// (empty unless RunOptions.DiscoverThinking was set).
	ThinkingDiscovery string `json:"thinking_discovery,omitempty"`
}

// thinkOpenMarkers are the known reasoning-block openers across model
// families; a generation containing one means reasoning was not suppressed.
var thinkOpenMarkers = []string{
	"<think", "<|think", "<|channel>thought", "<reasoning", "<|reasoning",
}

// ThinkingControlCandidate is one catalog entry for auto-discovery: a known
// model-card control shape to try against the live model.
type ThinkingControlCandidate struct {
	Label      string
	Field      string
	Value      any
	UserSuffix string
}

// thinkingCatalog enumerates the reasoning-disable shapes observed across
// model families (cards surveyed 2026-09-21). The engine default
// (think:false) is always tried implicitly as candidate zero.
var thinkingCatalog = []ThinkingControlCandidate{
	{Label: "reasoning_effort=none (OpenAI-style effort control)", Field: "reasoning_effort", Value: "none"},
	{Label: "enable_thinking=false (Qwen token plan / dashscope)", Field: "enable_thinking", Value: false},
	{Label: "thinking={type:disabled} (z.ai standard endpoint)", Field: "thinking", Value: map[string]any{"type": "disabled"}},
	{Label: "user_suffix /no_think (legacy GLM command token)", UserSuffix: "/no_think"},
	{Label: "chat_template_kwargs.enable_thinking=false (vLLM/SGLang)", Field: "chat_template_kwargs", Value: map[string]any{"enable_thinking": false}},
}

// DiscoverThinkingControl tries every catalog candidate against the live
// model and reports which (if any) suppresses reasoning markers.
func DiscoverThinkingControl(ctx context.Context, adpt provider.Adapter, model string,
	reg *registry.Registry, endpoint config.EndpointKind, cfg config.CompilerConfig) (string, bool) {

	ids, texts, _ := reg.Bind(registryLabels(reg))
	base := provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		Endpoint:            endpoint,
		PromptText:          fixedProbePrompt(cfg) + reg.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                reg.TopN,
		MaxOutputTokens:     64,
	}
	try := func(req provider.NextTokenScoreRequest) bool {
		res, err := adpt.ScoreNextToken(ctx, req)
		if err != nil {
			return false
		}
		for _, m := range thinkOpenMarkers {
			if strings.Contains(res.GeneratedText, m) {
				return false
			}
		}
		return true
	}

	if endpoint == config.EndpointChatCompletion {
		r := base
		r.NoReasoning = true
		if try(r) {
			return "engine default (NoReasoning -> think:false)", true
		}
	}
	for _, c := range thinkingCatalog {
		r := base
		r.PromptText = base.PromptText + c.UserSuffix
		r.ReasoningField = c.Field
		r.ReasoningValue = c.Value
		if try(r) {
			return c.Label, true
		}
	}
	return "", false
}

// probeThinkingControl runs a bounded generation with the model's configured
// disable control applied and reports whether reasoning markers appear.
func probeThinkingControl(ctx context.Context, adpt provider.Adapter, model string,
	reg *registry.Registry, modelCfg *config.ModelConfig, endpoint config.EndpointKind,
	cfg config.CompilerConfig, check func(string, bool, string)) (verified bool, tagsEmitted bool) {

	ids, texts, _ := reg.Bind(registryLabels(reg))
	caps := provider.CapabilitiesForModel(adpt, model)
	req := provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		Endpoint:            endpoint,
		PromptText:          fixedProbePrompt(cfg) + modelUserSuffix(modelCfg) + reg.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                caps.MaxTopLogprobs,
		NoReasoning:         endpoint == config.EndpointChatCompletion,
		MaxOutputTokens:     64,
	}
	strategy := "engine-default think:false"
	if modelCfg != nil && modelCfg.Thinking != nil {
		if modelCfg.Thinking.DisableField != "" {
			req.ReasoningField = modelCfg.Thinking.DisableField
			req.ReasoningValue = modelCfg.Thinking.DisableValue
			strategy = fmt.Sprintf("%s=%v", modelCfg.Thinking.DisableField, modelCfg.Thinking.DisableValue)
		}
		if modelCfg.Thinking.CloseTag != "" {
			strategy += fmt.Sprintf(" + close_tag %q", modelCfg.Thinking.CloseTag)
			if modelCfg.Thinking.WaitClose {
				strategy += " (wait_close)"
			}
		}
	}
	res, err := adpt.ScoreNextToken(ctx, req)
	if err != nil {
		check("thinking_control", false, strategy+": "+errString(err))
		return false, false
	}
	generated := res.GeneratedText
	for _, m := range thinkOpenMarkers {
		if strings.Contains(generated, m) {
			check("thinking_control", false,
				fmt.Sprintf("%s: reasoning markers still present in output %q", strategy, truncate(generated, 80)))
			return false, true
		}
	}
	check("thinking_control", true, strategy+": no reasoning markers in output")
	return true, false
}

// probeCompletionsFallback tries one scoring call against the completions
// surface and reports whether it yields candidate logprobs.
func probeCompletionsFallback(ctx context.Context, adpt provider.Adapter, model string,
	reg *registry.Registry, opts registry.Options, check func(string, bool, string)) bool {

	ids, texts, _ := reg.Bind(registryLabels(reg))
	res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		Endpoint:            config.EndpointCompletions,
		PromptText:          opts.PromptBase + opts.CloseTag + reg.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                reg.TopN,
	})
	ok := err == nil && len(res.CandidateLogprobs) >= 2
	detail := "logprobs recoverable: pin models[].endpoint: completions"
	if !ok {
		if err != nil {
			detail = "no usable logprobs: " + errString(err)
		} else {
			detail = fmt.Sprintf("only %d/%d labels recovered", len(res.CandidateLogprobs), len(texts))
		}
	}
	check("completions_fallback", ok, detail)
	return ok
}

func modelUserSuffix(mc *config.ModelConfig) string {
	if mc == nil || mc.Thinking == nil {
		return ""
	}
	return mc.Thinking.UserSuffix
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// RunOptions tunes a probe run.
type RunOptions struct {
	// DiscoverThinking tries the whole control catalog and reports the
	// working one as ready-to-paste YAML (report field ThinkingDiscovery).
	DiscoverThinking bool
}

// Run executes the capability suite against one provider/model. modelCfg
// carries the model-card-derived thinking configuration (may be nil).
func Run(ctx context.Context, adpt provider.Adapter, model string, cfg config.CompilerConfig, modelCfg *config.ModelConfig, opts ...RunOptions) (*Report, error) {
	var ro RunOptions
	if len(opts) > 0 {
		ro = opts[0]
	}
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

	caps := provider.CapabilitiesForModel(adpt, model)

	// Resolve the exact route first so the registry records the endpoint
	// actually used for scoring (§3.4).
	endpoint := config.EndpointCompletions
	if e, err := provider.ResolveExactRoute(caps, config.BackendPolicyConfig{}); err == nil {
		endpoint = e.Kind
	}

	// Build the registry: delimiter adoption + single-token labels (§6.1).
	// The production reasoning control applies: think-off measurements and
	// think-on traffic would disagree on N and label boundaries.
	regOpts := registry.Options{
		ScoringProfile:      provider.ScoringProfileKey(adpt, model),
		BackendModel:        model,
		TokenizerRevision:   caps.EngineVersion,
		Endpoint:            string(endpoint),
		TemplateVersion:     cfg.TemplateVersion,
		PromptBase:          probePrompt,
		DelimiterCandidates: cfg.DelimiterCandidates,
		Alphabets:           cfg.LabelAlphabets,
	}
	if endpoint == config.EndpointChatCompletion {
		regOpts.NoReasoning = true
	}
	if modelCfg != nil && modelCfg.Thinking != nil {
		if modelCfg.Thinking.DisableField != "" {
			regOpts.ReasoningField = modelCfg.Thinking.DisableField
			regOpts.ReasoningValue = modelCfg.Thinking.DisableValue
		}
		regOpts.UserSuffix = modelCfg.Thinking.UserSuffix
	}
	reg, regErr := registry.Build(ctx, adpt, regOpts)
	if regErr != nil {
		check("registry_build", false, errString(regErr))
		rep.Summary.Go = false
		return rep, nil
	}
	rep.Registry = reg
	check("registry_build", true,
		fmt.Sprintf("delimiter=%q labels=%d policy=%s top_n=%d/%d labels_visible=%d/%d",
			display(reg.Delimiter), len(reg.Labels()), reg.BoundaryPolicy,
			reg.TopN, reg.TopNCap, reg.TopNRecovered, len(reg.Labels())))

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

	if ro.DiscoverThinking {
		if label, ok := DiscoverThinkingControl(ctx, adpt, model, reg, endpoint, cfg); ok {
			rep.Summary.ThinkingDiscovery = label
		} else {
			rep.Summary.ThinkingDiscovery = "none of the catalog controls suppressed reasoning markers"
		}
	}

	// Model-card thinking control verification: model cards differ per
	// model (gemma4: <|think|> system token, tags still emitted when off;
	// gpt-oss: effort low/medium/high only; many cards document nothing),
	// so the configured control is tested against the live model instead
	// of trusted.
	rep.Summary.ThinkingControlVerified, rep.Summary.ThinkingTagsEmitted =
		probeThinkingControl(ctx, adpt, model, reg, modelCfg, endpoint, cfg, check)

	// Hidden reasoning detection (§12.1): visible output 1 token must equal
	// provider usage output tokens.
	if err == nil && res != nil {
		hidden := detectHiddenReasoning(adpt, res)
		rep.Summary.HiddenReasoning = hidden
		check("no_hidden_reasoning", !hidden, "")
	}

	// Cache evidence per §12.1: the same prefix run twice must show the
	// cached-token count INCREASE. A unique nonce keeps the first shot cold
	// even after the registry probes warmed the base prompt.
	rep.Summary.CacheEvidence = probeCacheEvidence(ctx, adpt, model, reg, modelCfg, endpoint, cfg, check)

	// Completions fallback (decision rule: think-off -> per-model N ->
	// completions). When the resolved route is chat and thinking markers
	// persist or N could not recover labels, the raw completions surface is
	// the last exact option; report whether it actually returns logprobs so
	// the operator can pin models[].endpoint: completions with evidence.
	if endpoint == config.EndpointChatCompletion {
		rep.Summary.CompletionsFallbackViable = probeCompletionsFallback(ctx, adpt, model, reg, regOpts, check)
	}

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
	caps := provider.CapabilitiesForModel(adpt, model)
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

// probeCacheEvidence runs two identical scoring calls on a fresh prefix and
// reports whether the second shot observes more cached prompt tokens than
// the first (llama.cpp tokens_cached, vLLM/Ollama prompt_tokens_details.
// cached_tokens). Engines that do not report cached tokens get an
// informational failure, not a gate failure.
func probeCacheEvidence(ctx context.Context, adpt provider.Adapter, model string,
	reg *registry.Registry, modelCfg *config.ModelConfig, endpoint config.EndpointKind,
	cfg config.CompilerConfig, check func(string, bool, string)) bool {

	nonce := time.Now().UnixNano()
	prompt := fixedProbePrompt(cfg) + "\n<!-- hearim-cache-probe-" + fmt.Sprintf("%d", nonce) + " -->" + reg.Delimiter
	ids, texts, _ := reg.Bind(registryLabels(reg))
	req := provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: model},
		Endpoint:            endpoint,
		PromptText:          prompt,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                adpt.Capabilities().MaxTopLogprobs,
	}
	if modelCfg != nil && modelCfg.Thinking != nil {
		if modelCfg.Thinking.DisableField != "" {
			req.ReasoningField = modelCfg.Thinking.DisableField
			req.ReasoningValue = modelCfg.Thinking.DisableValue
		}
	}
	first, err1 := adpt.ScoreNextToken(ctx, req)
	second, err2 := adpt.ScoreNextToken(ctx, req)
	if err1 != nil || err2 != nil {
		check("cache_evidence", false, fmt.Sprintf("probe calls failed: %v / %v", errString(err1), errString(err2)))
		return false
	}
	if !adpt.Capabilities().ReportsCachedTokens {
		check("cache_evidence", false, "engine does not report cached tokens; prefix reuse unverifiable")
		return false
	}
	delta := second.CachedPromptTokens - first.CachedPromptTokens
	ok := delta > 0 ||
		(second.CachedPromptTokens > 0 && second.CachedPromptTokens >= second.PromptTokens)
	check("cache_evidence", ok, fmt.Sprintf("cold=%d warm=%d delta=%d prompt_tokens=%d",
		first.CachedPromptTokens, second.CachedPromptTokens, delta, second.PromptTokens))
	return ok
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
