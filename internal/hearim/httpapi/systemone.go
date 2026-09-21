package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"hearim/internal/hearim/eval"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/schedule"
	"hearim/internal/hearim/usage"
)

const maxSystemOneBody = 16 << 20

// handleSystemOne implements POST /v1/systemone (TODO.md §2, §3.1, §8.1):
//
//	Client -> schema validator -> canonicalizer -> model router ->
//	EvaluationPlan compiler -> prefix-aware scheduler -> ProviderAdapter ->
//	logprob reducer -> Jev-compatible response assembler
func (s *Server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxSystemOneBody))
	if err != nil {
		writeJevError(w, r, jev.NewError(jev.CodeInternal, "read body: %v", err))
		return
	}

	parsed, err := jev.Validate(body)
	if err != nil {
		var ve *jev.RequestValidationError
		if errors.As(err, &ve) {
			writeJevError(w, r, jev.FromValidationError(ve))
			return
		}
		writeJevError(w, r, jev.NewError(jev.CodeValidationFailed, "%v", err))
		return
	}

	route, err := s.Router.Resolve(parsed.Model)
	if err != nil {
		writeJevError(w, r, jev.NewError(jev.CodeModelNotFound, "%v", err))
		return
	}

	// Budget gate (§9.1): throttle -> 429, hard stop -> 529.
	switch st, _ := s.Budget.State(); st {
	case usage.BudgetThrottle:
		writeJevError(w, r, jev.NewError(jev.CodeRateLimited, "budget utilization above throttle threshold"))
		return
	case usage.BudgetHardStop:
		writeJevError(w, r, jev.NewError(jev.CodeBackendOverloaded, "budget exhausted (hard stop)"))
		return
	}

	plan, err := s.Compiler.Compile(parsed, route.BackendModel)
	if err != nil {
		writeJevError(w, r, jev.NewError(jev.CodeValidationFailed, "%v", err))
		return
	}

	// Fan-out: one upstream next-token request per question (§5.3), through
	// the provider's prefix-aware scheduler (§8.3).
	sch := s.Schedulers[route.ProviderID]
	if sch == nil {
		writeJevError(w, r, jev.NewError(jev.CodeInternal, "no scheduler for provider %s", route.ProviderID))
		return
	}
	prefixKey := plan.CommonPrefixHash("", tokenizerRevisionOf(route.Adapter))

	results := make([]*eval.QuestionResult, len(plan.Questions))
	qerrs := make([]error, len(plan.Questions))
	var wg sync.WaitGroup
	for i := range plan.Questions {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			err := sch.Submit(r.Context(), schedule.Task{
				PrefixKey: prefixKey,
				Run: func(ctx context.Context) {
					res, err := s.Evaluator.EvaluateQuestion(ctx, plan, i, route)
					results[i] = res
					qerrs[i] = err
				},
			})
			if err != nil {
				qerrs[i] = err
			}
		}(i)
	}
	wg.Wait()

	// Whole-request failure by default; partial mode is a non-standard
	// extension (§10).
	answers := map[string]any{}
	var agg usage.Aggregate
	for i, qerr := range qerrs {
		if qerr != nil {
			if s.Cfg.Gateway.PartialAnswers && results[i] == nil {
				continue
			}
			var je *jev.Error
			if errors.As(qerr, &je) {
				writeJevError(w, r, je)
				return
			}
			writeJevError(w, r, jev.NewQuestionError(jev.CodeInternal, plan.Questions[i].ID, "%v", qerr))
			return
		}
		res := results[i]
		answers[res.ID] = res.Answer
		agg.InputTokens += res.InputTokens
		agg.OutputTokens += res.OutputTokens
		agg.ReportedCachedTokens += res.CachedTokens
		agg.UpstreamCalls++
	}

	resp := &jev.Response{
		Model:   parsed.Model,
		Answers: answers,
		Usage: jev.Usage{
			InputTokens:  agg.InputTokens,
			OutputTokens: agg.OutputTokens,
		},
	}

	meta := eval.MetaFrom(route, results, s.Cfg)
	s.recordUsage(route, agg)

	w.Header().Set("Content-Type", "application/json")
	setDiagnostics(w, meta)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		s.Logger.Error("encode response", "err", err)
	}
}

// recordUsage feeds the cost ledger and budget with reported figures only
// (§8.5: estimates are never billed).
func (s *Server) recordUsage(route eval.Route, agg usage.Aggregate) {
	var usd float64
	if route.Pricing != nil && agg.InputTokens > 0 {
		h := float64(agg.ReportedCachedTokens) / float64(agg.InputTokens)
		input := route.Pricing.InputPer1M*(1-h) + route.Pricing.CachedInputPer1M*h
		usd = input * float64(agg.InputTokens) / 1e6
		usd += route.Pricing.OutputPer1M * float64(agg.OutputTokens) / 1e6
	}
	s.Ledger.Record(route.ProviderID+"/"+route.BackendModel, agg, 0, usd)
	s.Budget.Record(usd)
}

// setDiagnostics writes the §3.1 headers. Implementation details stay out of
// the response body; diagnostics ride on headers.
func setDiagnostics(w http.ResponseWriter, m eval.Meta) {
	h := w.Header()
	h.Set("x-jev-implementation", m.Implementation)
	h.Set("x-jev-backend-model", m.BackendModel)
	h.Set("x-jev-question-calls", strconv.Itoa(m.QuestionCalls))
	h.Set("x-jev-cache-plan", m.CachePlan)
	h.Set("x-jev-confidence-method", m.ConfidenceMethod)
	if len(m.ScoringMethods) > 0 {
		h.Set("x-jev-scoring-method", strings.Join(m.ScoringMethods, ","))
	}
	for _, sp := range m.ProbabilitySpaces {
		if sp == "post-mask" {
			h.Set("x-jev-probability-space", "post-mask")
			break
		}
	}
	if len(m.CalibrationProfiles) > 0 {
		h.Set("x-jev-calibration-profile", strings.Join(m.CalibrationProfiles, ","))
	}
}

func writeJevError(w http.ResponseWriter, r *http.Request, e *jev.Error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.HTTPStatus())
	_ = json.NewEncoder(w).Encode(map[string]any{"error": e})
}
