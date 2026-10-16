package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"hearim/internal/hearim/config"
	"hearim/internal/hearim/jev"
)

// DefaultSystemOnePath is the Jev endpoint path under the provider root.
const DefaultSystemOnePath = "/v1/systemone"

// JevUpstream forwards System One requests verbatim to a real Jev endpoint
// (engine "jev"): TypeSafe's api.typesafe.ai or OpenCode Zen's
// opencode.ai/inference/systemone. hearim does no token scoring on this
// route; the upstream's probabilities are passed through unchanged.
type JevUpstream struct {
	cfg  config.ProviderConfig
	http *httpClient
	path string
}

// NewJevUpstream builds the passthrough client for a jev provider.
func NewJevUpstream(cfg config.ProviderConfig) *JevUpstream {
	path := DefaultSystemOnePath
	if cfg.Paths != nil && cfg.Paths.SystemOne != "" {
		path = cfg.Paths.SystemOne
	}
	return &JevUpstream{cfg: cfg, http: newHTTPClient(cfg), path: path}
}

func (j *JevUpstream) ID() string { return j.cfg.ID }

// JevUpstreamResult is one upstream System One response.
type JevUpstreamResult struct {
	Model   string
	Answers map[string]json.RawMessage
	Usage   jev.Usage
	// Headers carries whitelisted upstream cost/usage/rate-limit headers.
	Headers map[string]string
}

// SystemOne sends a validated Jev request body upstream with its "model"
// replaced by the backend model name. state and questions are forwarded as
// raw JSON so criteria key order survives.
func (j *JevUpstream) SystemOne(ctx context.Context, body []byte, backendModel string) (*JevUpstreamResult, error) {
	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("provider: jev request: %w", err)
	}
	m, _ := json.Marshal(backendModel)
	req["model"] = m
	// ExtraParams merge like other engines, but never over the payload.
	for k, v := range j.cfg.ExtraParams {
		if k == "model" || k == "state" || k == "questions" {
			continue
		}
		if _, set := req[k]; set {
			continue
		}
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("provider: jev extra_params %s: %w", k, err)
		}
		req[k] = raw
	}

	var resp struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   jev.Usage                  `json:"usage"`
	}
	hdrs, err := j.http.doCapture(ctx, http.MethodPost, j.path, req, &resp)
	if err != nil {
		return nil, err
	}
	if resp.Answers == nil {
		return nil, fmt.Errorf("provider: jev response has no answers")
	}
	return &JevUpstreamResult{
		Model:   resp.Model,
		Answers: resp.Answers,
		Usage:   resp.Usage,
		Headers: CaptureHeaders(hdrs),
	}, nil
}

// DecodeJevAnswer turns one upstream answer into hearim's answer type
// (*jev.ChoiceAnswer, *jev.ScoreAnswer or *jev.NoulAnswer). The score
// legend is the Jev index-keyed object ({"0": "Calm", ...}); the array form
// hearim emitted before 0.6.5 is still accepted.
func DecodeJevAnswer(raw json.RawMessage) (any, error) {
	var head struct {
		Type jev.QuestionType `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("provider: jev answer: %w", err)
	}
	switch head.Type {
	case jev.TypeChoice:
		var a jev.ChoiceAnswer
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("provider: jev choice answer: %w", err)
		}
		return &a, nil
	case jev.TypeNoul:
		var a jev.NoulAnswer
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("provider: jev noul answer: %w", err)
		}
		return &a, nil
	case jev.TypeScore:
		var a struct {
			Type          string             `json:"type"`
			Score         float64            `json:"score"`
			Legend        json.RawMessage    `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    float64            `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("provider: jev score answer: %w", err)
		}
		legend, err := decodeLegend(a.Legend)
		if err != nil {
			return nil, err
		}
		return &jev.ScoreAnswer{Type: a.Type, Score: a.Score, Legend: legend,
			Probabilities: a.Probabilities, Confidence: a.Confidence}, nil
	default:
		return nil, fmt.Errorf("provider: jev answer has unknown type %q", head.Type)
	}
}

// decodeLegend reads a score legend. Structured-criteria legends carry
// objects ({"what": ..., "examples": [...]}) instead of strings; those
// values are kept as compact JSON text.
func decodeLegend(raw json.RawMessage) (map[string]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		out := make(map[string]string, len(arr))
		for i, v := range arr {
			out[strconv.Itoa(i)] = v
		}
		return out, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("provider: jev score legend is neither object nor array")
	}
	out := make(map[string]string, len(obj))
	for k, v := range obj {
		if i, err := strconv.Atoi(k); err != nil || i < 0 {
			return nil, fmt.Errorf("provider: jev score legend key %q is not a level index", k)
		}
		var text string
		if err := json.Unmarshal(v, &text); err != nil {
			text = string(v)
		}
		out[k] = text
	}
	return out, nil
}
