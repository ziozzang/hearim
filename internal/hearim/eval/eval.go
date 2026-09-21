// Package eval orchestrates one Jev request into per-question upstream
// scoring (TODO.md §3.11), probability restoration (§7), reductions (§7.4),
// usage aggregation (§8.5), and the diagnostic headers of §3.1.
package eval

import (
	"context"
	"fmt"
	"sort"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/registry"
	"hearim/internal/hearim/scoring"
)

// Route binds a public model to a concrete backend execution path.
type Route struct {
	Adapter      provider.Adapter
	ProviderID   string
	BackendModel string
	Engine       config.EngineKind
	Registry     *registry.Registry // label registry for this route
	Endpoint     config.EndpointKind
	Pricing      *config.RouteCandidate
	CachePlan    string
	// ModelCfg carries the configured upstream LLM type (chat|thinking|
	// vision) and per-model extra parameters.
	ModelCfg *config.ModelConfig
}

// ModelCfgThinking returns the model's thinking configuration, if any.
func (r Route) ModelCfgThinking() *config.ThinkingConfig {
	if r.ModelCfg == nil {
		return nil
	}
	return r.ModelCfg.Thinking
}

// QuestionResult carries one question's answer plus diagnostics.
type QuestionResult struct {
	ID               string
	Answer           any
	InputTokens      int64
	OutputTokens     int64
	CachedTokens     int64
	CandidateMass    float64
	ScoringMethod    string
	ProbabilitySpace string
	Attempts         []string
	ProfileID        string
	// UpstreamHeaders carries whitelisted cost/usage/rate-limit headers
	// from the winning upstream response.
	UpstreamHeaders map[string]string
}

// Meta summarizes a whole evaluation for headers and logs.
type Meta struct {
	Implementation      string
	BackendModel        string
	QuestionCalls       int
	CachePlan           string
	ConfidenceMethod    string
	ScoringMethods      []string
	ProbabilitySpaces   []string
	CalibrationProfiles []string
	TotalCandidateMass  float64
	// TotalCachedTokens is the sum of server-REPORTED cached input tokens
	// across questions (§8.5: reported only, never estimated).
	TotalCachedTokens int64
	// UpstreamHeaders merges per-question whitelisted upstream headers
	// (first value wins) for client passthrough.
	UpstreamHeaders map[string]string
}

// Evaluator executes compiled plans against a route.
type Evaluator struct {
	Compiler         *compile.Compiler
	ScoringCfg       config.ScoringConfig
	CompilerCfg      config.CompilerConfig
	Policy           config.BackendPolicyConfig
	Strict           bool
	MinCandidateMass float64
}

// New builds an evaluator from configuration.
func New(c *compile.Compiler, cfg *config.Config) *Evaluator {
	return &Evaluator{
		Compiler:         c,
		ScoringCfg:       cfg.Scoring,
		CompilerCfg:      cfg.Compiler,
		Policy:           cfg.BackendPolicy,
		Strict:           cfg.StrictMode(),
		MinCandidateMass: cfg.Scoring.MinCandidateMass,
	}
}

// EvaluateQuestion scores one compiled question through the strategy chain
// of TODO.md §3.11 and reduces it to the public answer.
func (e *Evaluator) EvaluateQuestion(ctx context.Context, plan *compile.EvaluationPlan, qi int, route Route) (*QuestionResult, error) {
	q := plan.Questions[qi]
	caps := route.Adapter.Capabilities()

	// Vision gate: image-bearing states need a route that accepts
	// image_url content parts. Raw completion paths cannot carry them, and
	// a model explicitly typed chat/thinking is not a VLM.
	if len(plan.Images) > 0 {
		modelType := ""
		if route.ModelCfg != nil {
			modelType = route.ModelCfg.Type
		}
		visionModel := modelType == "" || modelType == "vision"
		if !caps.Vision || route.Endpoint != config.EndpointChatCompletion || !visionModel {
			return nil, jev.NewError(jev.CodeValidationFailed,
				"state declares %d image(s); model %s (type %q) is not on a vision-capable chat route (endpoint %s)",
				len(plan.Images), route.BackendModel, orDefault(modelType, "chat"), route.Endpoint)
		}
	}

	if route.Registry == nil {
		return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID,
			"no token label registry for route %s", route.BackendModel)
	}
	ids, texts, bound := route.Registry.Bind(labelTexts(q))
	if len(q.Candidates) > route.Registry.MaxExactCandidates && e.ScoringCfg.ChoiceTextMode == "disabled" {
		return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID,
			"%d candidates exceed max exact candidates %d and choice-text mode is disabled",
			len(q.Candidates), route.Registry.MaxExactCandidates)
	}

	// <think> handling (TODO.md §3.4 technique, model-card driven):
	// preload mode appends the close tag to the prompt so the very next
	// token is the label; wait-close mode instead generates through the
	// reasoning block and scores the position after the tag (approximate).
	thinking := route.ModelCfgThinking()
	preloadTag := ""
	if thinking != nil && thinking.CloseTag != "" && !thinking.WaitClose {
		preloadTag = thinking.CloseTag
	}

	prompt := q.PromptPrefix + q.PromptSuffix + preloadTag + route.Registry.Delimiter
	res := &QuestionResult{ID: q.ID}
	var attempts []string

	// Strategy chain (§3.11): selected-token-ids -> top-k -> (retry) ->
	// teacher-forced-label -> constrained-vocab -> choice-text.
	var logprobs map[int]float64
	var method, space string
	var usage provider.NextTokenScoreResult

	chatMsgs := route.Adapter.RenderChat(plan, qi)
	if preloadTag != "" && len(chatMsgs) > 0 {
		// Append the close tag to the last message content (string concat,
		// or an extra text part for multimodal content).
		last := chatMsgs[len(chatMsgs)-1]
		if s, ok := last.Content.(string); ok {
			chatMsgs[len(chatMsgs)-1].Content = s + preloadTag
		} else if parts, ok := last.Content.([]map[string]any); ok {
			chatMsgs[len(chatMsgs)-1].Content = append(parts, map[string]any{"type": "text", "text": preloadTag})
		}
	}

	scoreReq := provider.NextTokenScoreRequest{
		Model: provider.ModelIdentity{
			Provider: route.ProviderID,
			Model:    route.BackendModel,
		},
		Endpoint:            route.Endpoint,
		PromptText:          prompt,
		ChatMessages:        chatMsgs,
		CandidateTokenTexts: texts,
		Temperature:         e.CompilerCfg.Temperature,
		TopP:                e.CompilerCfg.TopP,
		NoReasoning:         route.Endpoint == config.EndpointChatCompletion,
	}
	if thinking != nil {
		if thinking.DisableField != "" {
			scoreReq.ReasoningField = thinking.DisableField
			scoreReq.ReasoningValue = thinking.DisableValue
		}
		if thinking.WaitClose && thinking.CloseTag != "" {
			scoreReq.WaitClose = true
			scoreReq.CloseTag = thinking.CloseTag
			scoreReq.MaxOutputTokens = thinking.EffectiveMaxThinkTokens()
		}
	}

	tryScore := func(constrain bool) (*provider.NextTokenScoreResult, error) {
		req := scoreReq
		req.CandidateTokenIDs = ids
		req.ConstrainToCandidates = constrain
		req.TopK = caps.MaxTopLogprobs
		return route.Adapter.ScoreNextToken(ctx, req)
	}

	if bound && caps.SupportsSelectedTokenIDs() {
		out, err := tryScore(false)
		attempts = append(attempts, "selected-token-ids")
		if err == nil && out.AllCandidatesPresent {
			logprobs, method, space, usage = out.CandidateLogprobs, out.ScoringMethod, out.ProbabilitySpace, *out
		} else if err != nil {
			attempts = append(attempts, fmt.Sprintf("selected-token-ids-error(%v)", err))
		}
	}

	if logprobs == nil {
		// top-k pass.
		req := scoreReq
		req.CandidateTokenIDs = ids
		req.TopK = caps.MaxTopLogprobs
		out, err := route.Adapter.ScoreNextToken(ctx, req)
		attempts = append(attempts, "top-k")
		if err == nil {
			logprobs, method, space, usage = out.CandidateLogprobs, out.ScoringMethod, out.ProbabilitySpace, *out
			if !out.AllCandidatesPresent {
				// §7.3 step 3: one retry with a larger N.
				retry := req
				retry.TopK = req.TopK * e.CompilerCfg.RetryTopLogprobsFactor
				if retry.TopK > caps.MaxTopLogprobs*8 {
					retry.TopK = caps.MaxTopLogprobs * 8
				}
				if out2, err2 := route.Adapter.ScoreNextToken(ctx, retry); err2 == nil && out2.AllCandidatesPresent {
					logprobs, method, space, usage = out2.CandidateLogprobs, out2.ScoringMethod, out2.ProbabilitySpace, *out2
					attempts = append(attempts, "top-k-retry")
				}
			}
		} else {
			attempts = append(attempts, fmt.Sprintf("top-k-error(%v)", err))
		}
	}

	if logprobs == nil && caps.PromptTokenLogprobs && bound && !scoreReq.WaitClose {
		// §7.3 step 4 / §3.11 strategy 3: teacher-forced label scoring for
		// stable single-token labels.
		out, err := e.teacherForcedLabels(ctx, plan, qi, route, ids, texts)
		attempts = append(attempts, "teacher-forced-label")
		if err == nil {
			logprobs, method, space, usage = out.CandidateLogprobs, "teacher-forced-label", provider.SpaceRaw, *out
		} else {
			attempts = append(attempts, fmt.Sprintf("teacher-forced-label-error(%v)", err))
		}
	}

	if logprobs == nil && supportsConstrainedVocab(caps) {
		out, err := tryScore(true)
		attempts = append(attempts, "constrained-vocab")
		if err == nil && out.AllCandidatesPresent {
			logprobs, method, space, usage = out.CandidateLogprobs, out.ScoringMethod, out.ProbabilitySpace, *out
		} else if err != nil {
			attempts = append(attempts, fmt.Sprintf("constrained-vocab-error(%v)", err))
		}
	}

	if logprobs == nil && e.ScoringCfg.ChoiceTextMode != "disabled" && caps.PromptTokenLogprobs && !scoreReq.WaitClose {
		out, err := e.choiceText(ctx, plan, qi, route)
		attempts = append(attempts, "choice-text")
		if err == nil {
			logprobs, method, space, usage = out.CandidateLogprobs, out.ScoringMethod, out.ProbabilitySpace, *out
		} else {
			attempts = append(attempts, fmt.Sprintf("choice-text-error(%v)", err))
		}
	}

	if logprobs == nil || len(logprobs) < len(texts) {
		if e.Strict {
			return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID,
				"All candidate token logprobs were not returned (attempts: %v).", attempts)
		}
		// Approximation mode: renormalize over recovered candidates only.
		if logprobs == nil {
			return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID,
				"No candidate logprobs recovered (attempts: %v).", attempts)
		}
	}

	// Order-preserving logprob slice.
	ordered := make([]float64, len(texts))
	for i := range texts {
		if lp, ok := logprobs[i]; ok {
			ordered[i] = lp
		} else {
			ordered[i] = -1e18 // effectively zero mass in strict-missing cases
		}
	}

	mass := scoring.CandidateMass(ordered)
	probs := scoring.Softmax(ordered)

	res.CandidateMass = mass
	res.ScoringMethod = method
	res.ProbabilitySpace = space
	res.UpstreamHeaders = usage.UpstreamHeaders
	res.Attempts = attempts
	res.InputTokens = int64(usage.PromptTokens)
	res.OutputTokens = int64(1)
	res.CachedTokens = int64(usage.CachedPromptTokens)

	tau := e.tau(method)
	prof := scoring.CalibrationProfile{
		Model: route.BackendModel, Engine: string(route.Engine),
		TemplateVersion: plan.TemplateVersion, ScoringMethod: method, Tau: tau,
	}
	res.ProfileID = prof.ID()

	if mass < e.MinCandidateMass {
		return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID,
			"candidate mass %g below floor %g: prompt non-compliance or model mismatch", mass, e.MinCandidateMass)
	}

	answer, err := reduce(q, probs, tau)
	if err != nil {
		return nil, err
	}
	res.Answer = answer
	return res, nil
}

// teacherForcedLabels scores prefix+label with prompt-token logprobs. For a
// stable single-token label this must agree with the next-token logprob
// (TODO.md §12.2 parity requirement).
func (e *Evaluator) teacherForcedLabels(ctx context.Context, plan *compile.EvaluationPlan, qi int, route Route, ids []int, texts []string) (*provider.NextTokenScoreResult, error) {
	q := plan.Questions[qi]
	prefix := q.PromptPrefix + q.PromptSuffix + route.Registry.CloseTag + route.Registry.Delimiter
	out := &provider.NextTokenScoreResult{
		CandidateLogprobs: map[int]float64{},
		ScoringMethod:     "teacher-forced-label",
		ProbabilitySpace:  provider.SpaceRaw,
	}
	creq := provider.ContinuationScoreRequest{
		Model:      provider.ModelIdentity{Provider: route.ProviderID, Model: route.BackendModel},
		Endpoint:   route.Endpoint,
		PrefixText: prefix,
	}
	for i, text := range texts {
		// The teacher-forced label scores only the label token; a full
		// continuation call per label keeps adapter contracts simple.
		c, err := route.Adapter.ScoreContinuations(ctx, withContinuation(creq, text))
		if err != nil {
			return nil, err
		}
		if len(c.Conditional) != 1 {
			return nil, fmt.Errorf("eval: continuation result length %d", len(c.Conditional))
		}
		out.CandidateLogprobs[i] = c.Conditional[0].LogProb
		out.PromptTokens += c.PromptTokens
	}
	out.AllCandidatesPresent = len(out.CandidateLogprobs) == len(texts)
	return out, nil
}

func withContinuation(req provider.ContinuationScoreRequest, text string) provider.ContinuationScoreRequest {
	req.Continuations = []string{text}
	return req
}

// choiceText scores the full choice strings with the configured continuation
// normalization and calibration temperature (§7.3.1).
func (e *Evaluator) choiceText(ctx context.Context, plan *compile.EvaluationPlan, qi int, route Route) (*provider.NextTokenScoreResult, error) {
	q := plan.Questions[qi]
	mode := scoring.ContinuationMode(e.ScoringCfg.ContinuationNormalization)
	tau := e.tau("choice-text")
	creq := provider.ContinuationScoreRequest{
		Model:         provider.ModelIdentity{Provider: route.ProviderID, Model: route.BackendModel},
		Endpoint:      route.Endpoint,
		PrefixText:    q.PromptPrefix + q.PromptSuffix + route.Registry.Delimiter,
		Unconditional: mode == scoring.ModePMI,
	}
	conts := make([]string, 0, len(q.Candidates))
	for _, c := range q.Candidates {
		conts = append(conts, c.SequenceText)
	}
	creq.Continuations = conts
	c, err := route.Adapter.ScoreContinuations(ctx, creq)
	if err != nil {
		return nil, err
	}
	cond := make([]float64, len(c.Conditional))
	lens := make([]float64, len(c.Conditional))
	for i, cs := range c.Conditional {
		cond[i] = cs.LogProb
		lens[i] = float64(cs.TokenLen)
	}
	var uncond []float64
	if len(c.Unconditional) == len(cond) {
		uncond = make([]float64, len(cond))
		for i, cs := range c.Unconditional {
			uncond[i] = cs.LogProb
		}
	}
	normed, err := scoring.NormalizeContinuation(cond, lens, uncond, mode, tau)
	if err != nil {
		return nil, err
	}
	out := &provider.NextTokenScoreResult{
		CandidateLogprobs:    map[int]float64{},
		AllCandidatesPresent: true,
		ScoringMethod:        "choice-text-" + string(mode),
		ProbabilitySpace:     provider.SpaceRaw,
		PromptTokens:         c.PromptTokens,
		CachedPromptTokens:   c.CachedTokens,
	}
	for i, lp := range normed {
		out.CandidateLogprobs[i] = lp
	}
	return out, nil
}

func (e *Evaluator) tau(method string) float64 {
	if v, ok := e.ScoringCfg.CalibrationTau[method]; ok && v > 0 {
		return v
	}
	return 1
}

func supportsConstrainedVocab(caps provider.ProviderCapabilities) bool {
	for _, s := range caps.CandidateScoring {
		if s == "constrained-vocab" {
			return true
		}
	}
	return false
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func labelTexts(q *compile.CompiledQuestion) []string {
	out := make([]string, 0, len(q.Candidates))
	for _, c := range q.Candidates {
		out = append(out, c.TokenText)
	}
	return out
}

// reduce applies the §7.4 primitive reductions.
func reduce(q *compile.CompiledQuestion, probs []float64, tau float64) (any, error) {
	switch q.Reduction {
	case compile.ReductionCategorical:
		pm := map[string]float64{}
		for i, c := range q.Candidates {
			pm[c.PublicKey] = probs[i]
		}
		return &jev.ChoiceAnswer{
			Type:          "choice",
			Choice:        q.Candidates[scoring.Argmax(probs)].PublicKey,
			Probabilities: pm,
			Confidence:    scoring.Confidence(probs),
		}, nil
	case compile.ReductionOrdinalMean:
		pm := map[string]float64{}
		for i, c := range q.Candidates {
			pm[c.PublicKey] = probs[i]
		}
		legend := make([]string, 0, len(q.Candidates))
		for _, c := range q.Candidates {
			idx, _ := c.PublicValue.(int)
			for len(legend) <= idx {
				legend = append(legend, "")
			}
			legend[idx] = c.SequenceText
		}
		return &jev.ScoreAnswer{
			Type:          "score",
			Score:         scoring.WeightedMean(probs),
			Legend:        legend,
			Probabilities: pm,
			Confidence:    scoring.Confidence(probs),
		}, nil
	case compile.ReductionTrueProbability:
		return &jev.NoulAnswer{Type: "noul", Noul: probs[0]}, nil
	default:
		return nil, fmt.Errorf("eval: unknown reduction %q", q.Reduction)
	}
}

// MetaFrom assembles response metadata.
func MetaFrom(route Route, results []*QuestionResult, cfg *config.Config) Meta {
	m := Meta{
		Implementation:   fmt.Sprintf("%s-logprob-v1", route.Engine),
		BackendModel:     route.BackendModel,
		QuestionCalls:    len(results),
		CachePlan:        route.CachePlan,
		ConfidenceMethod: cfg.Gateway.ConfidenceMethod,
	}
	seenMethod := map[string]bool{}
	seenSpace := map[string]bool{}
	seenProf := map[string]bool{}
	for _, r := range results {
		if r == nil {
			continue
		}
		if !seenMethod[r.ScoringMethod] {
			m.ScoringMethods = append(m.ScoringMethods, r.ScoringMethod)
			seenMethod[r.ScoringMethod] = true
		}
		if !seenSpace[r.ProbabilitySpace] {
			m.ProbabilitySpaces = append(m.ProbabilitySpaces, r.ProbabilitySpace)
			seenSpace[r.ProbabilitySpace] = true
		}
		if !seenProf[r.ProfileID] {
			m.CalibrationProfiles = append(m.CalibrationProfiles, r.ProfileID)
			seenProf[r.ProfileID] = true
		}
		m.TotalCandidateMass += r.CandidateMass
		m.TotalCachedTokens += r.CachedTokens
		for k, v := range r.UpstreamHeaders {
			if m.UpstreamHeaders == nil {
				m.UpstreamHeaders = map[string]string{}
			}
			if _, exists := m.UpstreamHeaders[k]; !exists {
				m.UpstreamHeaders[k] = v
			}
		}
	}
	sort.Strings(m.ScoringMethods)
	return m
}
