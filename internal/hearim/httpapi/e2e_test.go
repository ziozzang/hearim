package httpapi

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeUpstream is a vLLM-shaped backend: per-rune tokenizer, completions
// with top_logprobs for the requested labels, deterministic distribution.
type fakeUpstream struct {
	t               *testing.T
	completionsSeen int
	lastPrompts     []string
}

func (f *fakeUpstream) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /tokenize", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Prompt string `json:"prompt"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		ids := make([]int, 0, len(body.Prompt))
		for _, rr := range body.Prompt {
			ids = append(ids, int(rr)+1)
		}
		json.NewEncoder(w).Encode(map[string]any{"tokens": ids})
	})
	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) {
		f.completionsSeen++
		var body struct {
			Prompt          string `json:"prompt"`
			LogprobTokenIDs []int  `json:"logprob_token_ids"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		f.lastPrompts = append(f.lastPrompts, body.Prompt)

		// Find the trailing answer marker and the delimiter after it.
		// Distribution: label "2"/"B" favored.
		entries := []any{}
		// Extract labels from prompt criteria block.
		labels := []string{}
		for _, line := range strings.Split(body.Prompt, "\n") {
			if len(line) > 1 && (line[0] >= '1' && line[0] <= '9' || line[0] == '0' ||
				line[0] >= 'A' && line[0] <= 'Z') && line[1] == ' ' {
				labels = append(labels, string(line[0]))
			}
		}
		if len(labels) == 0 {
			// Default distribution for extension probes without criteria
			// lines in the prompt (A favored).
			labels = []string{"A", "B", "C", "D"}
		}
		for i, l := range labels {
			entries = append(entries, map[string]any{
				"token":   l,
				"bytes":   []int{int(l[0])},
				"logprob": -float64(i+1) - 0.5,
			})
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id": "breq-1",
			"choices": []any{map[string]any{
				"text": "",
				"logprobs": map[string]any{
					"tokens":       []string{},
					"top_logprobs": [][]any{entries},
				},
			}},
			"usage": map[string]any{
				"prompt_tokens":         len(body.Prompt),
				"completion_tokens":     1,
				"prompt_tokens_details": map[string]any{"cached_tokens": len(body.Prompt) / 2},
			},
		})
	})
	mux.HandleFunc("GET /version", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"version": "0.9.2"})
	})
	return mux
}

func newTestServer(t *testing.T, upstreamURL string, extraYAML string) (*Server, http.Handler) {
	t.Helper()
	yaml := fmt.Sprintf(`
gateway:
  default_model: jev-gemma4
  strict_candidate_probabilities: true
auth:
  api_keys: ["test-key"]
providers:
  - id: local
    engine: vllm
    base_url: %s
    models: ["gemma4:31b"]
    concurrency: 3
model_aliases:
  jev-gemma4: local:gemma4:31b
%s
`, upstreamURL, extraYAML)
	cfg, err := configParseForTest(yaml)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(&discardWriter{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	srv, err := New(cfg, logger, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return srv, srv.Handler()
}

type discardWriter struct{}

func (d *discardWriter) Write(p []byte) (int, error) { return len(p), nil }

func TestSystemOneEndToEnd(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()

	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	body := `{
	  "model": "jev-gemma4",
	  "state": {"ticket": "결제 후 다운로드 링크를 받지 못했습니다.", "account_age_days": 820},
	  "questions": {
	    "routing": {
	      "type": "choice",
	      "instructions": "가장 적절한 처리 큐를 선택하라.",
	      "criteria": {"billing": "결제 문제", "delivery": "전달 문제", "account": "계정 문제", "other": null}
	    },
	    "quality": {
	      "type": "score",
	      "criteria": ["부정확", "부분", "정확", "완벽"]
	    },
	    "refund": {
	      "type": "noul",
	      "criteria": {"true": "요청 있음", "false": "없음"}
	    }
	  }
	}`

	req, _ := http.NewRequest("POST", ts.URL+"/v1/systemone", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var raw map[string]any
		json.NewDecoder(resp.Body).Decode(&raw)
		t.Fatalf("status = %d: %v", resp.StatusCode, raw)
	}

	// Diagnostic headers (§3.1).
	if got := resp.Header.Get("x-jev-implementation"); got != "vllm-logprob-v1" {
		t.Errorf("implementation = %q", got)
	}
	if got := resp.Header.Get("x-jev-backend-model"); got != "gemma4:31b" {
		t.Errorf("backend model = %q", got)
	}
	if got := resp.Header.Get("x-jev-question-calls"); got != "3" {
		t.Errorf("question calls = %q", got)
	}
	if got := resp.Header.Get("x-jev-confidence-method"); got != "normalized-entropy-v1" {
		t.Errorf("confidence method = %q", got)
	}
	if resp.Header.Get("x-jev-scoring-method") == "" {
		t.Error("scoring method header missing")
	}
	if resp.Header.Get("x-request-id") == "" {
		t.Error("request id missing")
	}

	var out struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Model != "jev-gemma4" {
		t.Errorf("model = %q", out.Model)
	}
	if len(out.Answers) != 3 {
		t.Fatalf("answers = %d", len(out.Answers))
	}
	var choice struct {
		Type          string             `json:"type"`
		Choice        string             `json:"choice"`
		Probabilities map[string]float64 `json:"probabilities"`
		Confidence    float64            `json:"confidence"`
	}
	json.Unmarshal(out.Answers["routing"], &choice)
	if choice.Choice != "billing" {
		t.Errorf("argmax should be first label (logprob ordering), got %q", choice.Choice)
	}
	var sum float64
	for _, v := range choice.Probabilities {
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("probabilities sum = %v", sum)
	}
	var score struct {
		Score  float64  `json:"score"`
		Legend []string `json:"legend"`
	}
	json.Unmarshal(out.Answers["quality"], &score)
	if len(score.Legend) != 4 {
		t.Errorf("legend = %v", score.Legend)
	}
	var noul struct {
		Noul float64 `json:"noul"`
	}
	json.Unmarshal(out.Answers["refund"], &noul)
	if noul.Noul < 0 || noul.Noul > 1 {
		t.Errorf("noul = %v", noul.Noul)
	}
	if out.Usage.InputTokens == 0 || out.Usage.OutputTokens != 3 {
		t.Errorf("usage = %+v (output should be 1 per question)", out.Usage)
	}

	// Fan-out shape: three upstream completions, all sharing the same state
	// prefix (§1, §8.3).
	if up.completionsSeen < 3 {
		t.Errorf("upstream calls = %d, want >= 3", up.completionsSeen)
	}
	statePrefix := "<state encoding=\"canonical-json\""
	for i, p := range up.lastPrompts {
		if !strings.Contains(p, statePrefix) {
			t.Errorf("prompt %d lacks state block", i)
		}
		if !strings.HasSuffix(p, "<answer-label>\n\n") {
			t.Errorf("prompt %d must end with marker+delimiter, tail %q", i, p[len(p)-20:])
		}
		if strings.Contains(p, "routing") || strings.Contains(p, "refund") {
			t.Errorf("prompt %d leaks question ids", i)
		}
	}
}

func TestSystemOneErrors(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	post := func(body, auth string) *http.Response {
		req, _ := http.NewRequest("POST", ts.URL+"/v1/systemone", strings.NewReader(body))
		if auth != "" {
			req.Header.Set("Authorization", "Bearer "+auth)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// 401: bad key.
	if resp := post(`{"model":"jev-gemma4","state":"s","questions":{"q":{"type":"noul"}}}`, "wrong"); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("bad key status = %d", resp.StatusCode)
		resp.Body.Close()
	}

	// 422: invalid body with per-question detail.
	resp := post(`{"model":"jev-gemma4","state":42,"questions":{"q":{"type":"essay"}}}`, "test-key")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("validation status = %d", resp.StatusCode)
	}
	var e map[string]struct {
		Type    string `json:"type"`
		Details []struct {
			Path    string `json:"path"`
			Message string `json:"message"`
		} `json:"details"`
	}
	json.NewDecoder(resp.Body).Decode(&e)
	if e["error"].Type != "validation_failed" || len(e["error"].Details) == 0 {
		t.Errorf("error body = %v", e)
	}

	// 422: unknown model.
	resp2 := post(`{"model":"nope","state":"s","questions":{"q":{"type":"noul"}}}`, "test-key")
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("unknown model status = %d", resp2.StatusCode)
	}
}

func TestReadiness(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	resp, err := http.Get(ts.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("readyz = %d", resp.StatusCode)
	}

	req2, _ := http.NewRequest("GET", ts.URL+"/v1/routes", nil)
	req2.Header.Set("Authorization", "Bearer test-key")
	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var routes struct {
		Routes []map[string]any `json:"routes"`
	}
	json.NewDecoder(resp2.Body).Decode(&routes)
	if len(routes.Routes) != 1 {
		t.Fatalf("routes = %v", routes.Routes)
	}
	if routes.Routes[0]["model"] != "gemma4:31b" {
		t.Errorf("routes = %v", routes.Routes)
	}
	if routes.Routes[0]["ready"] != true {
		t.Errorf("route should be ready: %v", routes.Routes[0])
	}
}

func TestOpenAIProxyPassthrough(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	body := `{"model":"gemma4:31b","prompt":"hello","max_tokens":50}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("proxy status = %d", resp.StatusCode)
	}
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	if out["choices"] == nil {
		t.Error("upstream choices missing")
	}
}

func TestJevCandidatesExtension(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	body := `{"model":"gemma4:31b","prompt":"...\nANSWER:","max_tokens":1,"logprobs":10,
	          "jev_candidates": {"A": "billing", "B": "delivery", "C": "account", "D": "other"}}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var out struct {
		JevChoice        string             `json:"jev_choice"`
		JevProbabilities map[string]float64 `json:"jev_probabilities"`
		JevMethod        string             `json:"jev_scoring_method"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if out.JevChoice != "billing" {
		t.Errorf("choice = %q", out.JevChoice)
	}
	var sum float64
	for _, v := range out.JevProbabilities {
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("probabilities sum = %v", sum)
	}
	if out.JevMethod == "" {
		t.Error("method missing")
	}
	// The extension must forward the prompt bytes exactly.
	if len(up.lastPrompts) == 0 || up.lastPrompts[len(up.lastPrompts)-1] != "...\nANSWER:" {
		t.Errorf("prompt forwarded = %v", up.lastPrompts)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	// One request to seed counters.
	body := `{"model":"jev-gemma4","state":"s","questions":{"q":{"type":"noul"}}}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/systemone", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	if _, err := http.DefaultClient.Do(req); err != nil {
		t.Fatal(err)
	}

	req2, _ := http.NewRequest("GET", ts.URL+"/metrics", nil)
	req2.Header.Set("Authorization", "Bearer test-key")
	resp, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("metrics status = %d", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	out := string(data)
	for _, want := range []string{
		"# TYPE hearim_requests_total counter",
		`hearim_requests_total{path="/v1/systemone",status="200"} 1`,
		"# TYPE hearim_request_duration_seconds histogram",
		"# TYPE hearim_route_ready gauge",
		"hearim_routes_ready 1",
		"# TYPE hearim_budget_state gauge",
		"go_goroutines",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
}

func TestImageRequestGatedOnNonVisionRoute(t *testing.T) {
	up := &fakeUpstream{t: t}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	srv, handler := newTestServer(t, upSrv.URL, "")
	ts := httptest.NewServer(handler)
	defer ts.Close()
	defer srv.Shutdown(defaultCtx())

	// The e2e config types qwen-less vLLM route without vision model entry;
	// the adapter declares Vision true but the route is chat for Ollama...
	// Here the route is vLLM completions (non-chat), so images must be 422.
	body := `{"model":"jev-gemma4","state":{"note":"x","image":"https://x/a.png"},"questions":{"q":{"type":"noul"}}}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/systemone", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("image on non-vision route status = %d", resp.StatusCode)
	}
	var e struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&e)
	if e.Error.Type != "validation_failed" {
		t.Errorf("error type = %s", e.Error.Type)
	}
}
