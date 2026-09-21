package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"hearim/internal/hearim/eval"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/scoring"
)

// handleProxy transparently forwards /v1/completions and /v1/chat/completions
// to the default route's provider (TODO.md §3.2). Plain requests stay
// untouched; hearim never guesses Jev evaluations out of chat traffic.
//
// The non-standard jev_candidates extension on /v1/completions turns the
// request into a label scoring call: candidates map label tokens to public
// values, and the response gains jev_choice/jev_probabilities fields.
func (s *Server) handleProxy(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		route, err := s.Router.Resolve("")
		if err != nil {
			writeJevError(w, r, jev.NewError(jev.CodeModelNotFound, "%v", err))
			return
		}
		pcfg := s.providerConfig(route.ProviderID)

		body, err := io.ReadAll(io.LimitReader(r.Body, maxSystemOneBody))
		if err != nil {
			writeJevError(w, r, jev.NewError(jev.CodeInternal, "read body: %v", err))
			return
		}

		// Non-standard extension: jev_candidates on completions.
		if kind == "completions" && s.tryJevCandidates(w, r, route, body) {
			return
		}

		upstreamPath := "/v1/completions"
		if kind == "chat" {
			upstreamPath = "/v1/chat/completions"
		}
		req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
			strings.TrimRight(pcfg.BaseURL, "/")+upstreamPath, bytes.NewReader(body))
		if err != nil {
			writeJevError(w, r, jev.NewError(jev.CodeInternal, "%v", err))
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		if pcfg.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+pcfg.APIKey)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			writeJevError(w, r, jev.NewError(jev.CodeBackendUnavailable, "upstream: %v", err))
			return
		}
		defer resp.Body.Close()
		copyHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		if strings.EqualFold(k, "Content-Length") || strings.EqualFold(k, "Transfer-Encoding") {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

// tryJevCandidates implements the optional extension of TODO.md §3.2:
//
//	{"model": "...", "prompt": "...\nANSWER:", "max_tokens": 1,
//	 "logprobs": 10, "jev_candidates": {"A": "billing", ...}}
//
// The prompt is forwarded byte-identically, the declared labels are scored at
// the next position, and probabilities are the conditional softmax over the
// candidate set.
func (s *Server) tryJevCandidates(w http.ResponseWriter, r *http.Request, route eval.Route, body []byte) bool {
	var probe struct {
		Prompt        string            `json:"prompt"`
		JevCandidates map[string]string `json:"jev_candidates"`
	}
	if err := json.Unmarshal(body, &probe); err != nil || len(probe.JevCandidates) == 0 {
		return false
	}
	// Preserve the request's JSON key order: candidate order determines the
	// label sequence and must not depend on Go map iteration.
	raw := struct {
		JevCandidates json.RawMessage `json:"jev_candidates"`
	}{}
	_ = json.Unmarshal(body, &raw)
	ordered, err := s.objectKeyOrder(raw.JevCandidates)
	if err != nil || len(ordered) == 0 {
		ordered = nil
		for l := range probe.JevCandidates {
			ordered = append(ordered, l)
		}
	}
	labels := make([]string, 0, len(ordered))
	values := make([]string, 0, len(ordered))
	for _, l := range ordered {
		labels = append(labels, l)
		values = append(values, probe.JevCandidates[l])
	}
	caps := route.Adapter.Capabilities()
	res, err := route.Adapter.ScoreNextToken(r.Context(), provider.NextTokenScoreRequest{
		Model: provider.ModelIdentity{
			Provider: route.ProviderID,
			Model:    route.BackendModel,
		},
		Endpoint:            route.Endpoint,
		PromptText:          probe.Prompt,
		CandidateTokenTexts: labels,
		Temperature:         s.Cfg.Compiler.Temperature,
		TopP:                s.Cfg.Compiler.TopP,
		TopK:                caps.MaxTopLogprobs,
	})
	if err != nil {
		writeJevError(w, r, jev.NewError(jev.CodeBackendUnavailable, "upstream: %v", err))
		return true
	}
	logps := make([]float64, len(labels))
	for i := range labels {
		if lp, ok := res.CandidateLogprobs[i]; ok {
			logps[i] = lp
		} else {
			logps[i] = -1e18
		}
	}
	probs := scoring.Softmax(logps)
	probsMap := map[string]float64{}
	best := 0
	for i, v := range values {
		probsMap[v] = probs[i]
		if probs[i] > probs[best] {
			best = i
		}
	}
	out := map[string]any{
		"jev_choice":            values[best],
		"jev_probabilities":     probsMap,
		"jev_scoring_method":    res.ScoringMethod,
		"jev_probability_space": res.ProbabilitySpace,
		"usage": map[string]int{
			"prompt_tokens":     res.PromptTokens,
			"completion_tokens": 1,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("x-jev-scoring-method", res.ScoringMethod)
	w.Header().Set("x-jev-backend-model", route.BackendModel)
	_ = json.NewEncoder(w).Encode(out)
	return true
}

// objectKeyOrder extracts a flat object's keys in JSON insertion order.
func (s *Server) objectKeyOrder(raw json.RawMessage) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("not an object")
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		k, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("key is not a string")
		}
		keys = append(keys, k)
		var skip any
		if err := dec.Decode(&skip); err != nil {
			return nil, err
		}
	}
	return keys, nil
}
