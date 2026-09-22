package mockinfer_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"hearim/internal/hearim/config"
	"hearim/internal/hearim/httpapi"
	"hearim/internal/hearim/provider"
	"hearim/internal/mockinfer"
)

func start(t *testing.T, engine, scenario string) (*mockinfer.Server, *httptest.Server) {
	t.Helper()
	m, err := mockinfer.New(mockinfer.Options{Engine: engine, Scenario: scenario})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m)
	t.Cleanup(srv.Close)
	return m, srv
}

func post(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	client := &http.Client{Timeout: 5 * time.Second}
	r, err := client.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return r.StatusCode, out
}

func TestWireMaskScenarios(t *testing.T) {
	for _, tc := range []struct {
		engine, scenario, path, body, sample string
		status                               int
	}{
		{"vllm", "normal", "/v1/completions", `{"prompt":"test","max_tokens":1,"logprobs":2,"logprob_token_ids":[66,67],"allowed_token_ids":[66,67]}`, "A", 200},
		{"vllm", "mask-ignored", "/v1/completions", `{"prompt":"test","max_tokens":1,"logprobs":2,"logprob_token_ids":[66,67],"allowed_token_ids":[66,67]}`, "X", 200},
		{"vllm", "mask-rejected", "/v1/completions", `{"allowed_token_ids":[66,67]}`, "", 400},
		{"sglang", "normal", "/generate", `{"sampling_params":{"allowed_token_ids":[66,67]}}`, "", 400},
		{"sglang", "mask-ignored", "/generate", `{"sampling_params":{"allowed_token_ids":[66,67]},"return_logprob":true,"token_ids_logprob":[66,67]}`, "X", 200},
	} {
		t.Run(tc.engine+"/"+tc.scenario, func(t *testing.T) {
			_, srv := start(t, tc.engine, tc.scenario)
			status, out := post(t, srv.URL+tc.path, tc.body)
			if status != tc.status {
				t.Fatalf("status=%d body=%v", status, out)
			}
			if status != 200 {
				return
			}
			text, _ := out["text"].(string)
			if tc.engine == "vllm" {
				text = out["choices"].([]any)[0].(map[string]any)["text"].(string)
			}
			if text != tc.sample {
				t.Fatalf("sample=%q want=%q", text, tc.sample)
			}
		})
	}
}

func TestAdapterScenarios(t *testing.T) {
	for _, tc := range []struct {
		engine, scenario string
		constrain        bool
		space            string
		wantError        bool
		mass             float64
	}{
		{"vllm", "normal", false, "raw", false, 0.3},
		{"vllm", "normal", true, "raw", false, 0.3},
		{"vllm", "mask-ignored", true, "raw", false, 0.3},
		{"vllm", "mask-rejected", true, "", true, 0},
		{"vllm", "processed-logprobs", true, "post-mask", false, 1},
		{"sglang", "normal", false, "raw", false, 0.3},
		{"sglang", "normal", true, "", true, 0},
		{"llama.cpp", "normal", false, "raw", false, 0.3},
		{"llama.cpp", "normal", true, "post-mask", false, 1},
		{"llama.cpp", "post-sampling-ignored", true, "", true, 0},
	} {
		t.Run(fmt.Sprintf("%s/%s/mask=%v", tc.engine, tc.scenario, tc.constrain), func(t *testing.T) {
			_, srv := start(t, tc.engine, tc.scenario)
			cfg := config.ProviderConfig{ID: "mock", Engine: config.EngineKind(tc.engine), BaseURL: srv.URL, RequestTimeout: 5 * time.Second}
			if tc.scenario == "processed-logprobs" {
				cfg.Scoring = &config.ProviderScoringConfig{LogprobSpace: "post-mask"}
			}
			a, err := provider.NewAdapter(cfg)
			if err != nil {
				t.Fatal(err)
			}
			req := provider.NextTokenScoreRequest{Model: provider.ModelIdentity{Model: "mock"}, PromptText: "Choose A or B", CandidateTokenIDs: []int{66, 67}, CandidateTokenTexts: []string{"A", "B"}, TopK: 10, ConstrainToCandidates: tc.constrain}
			res, err := a.ScoreNextToken(context.Background(), req)
			if tc.wantError {
				if err == nil {
					t.Fatal("expected rejection")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !res.AllCandidatesPresent || res.ProbabilitySpace != tc.space {
				t.Fatalf("result=%+v", res)
			}
			mass := math.Exp(res.CandidateLogprobs[0]) + math.Exp(res.CandidateLogprobs[1])
			if math.Abs(mass-tc.mass) > 1e-9 {
				t.Fatalf("mass=%g want=%g", mass, tc.mass)
			}
			if math.Abs(math.Exp(res.CandidateLogprobs[0])/mass-2.0/3) > 1e-9 {
				t.Fatal("wrong conditional distribution")
			}
		})
	}
}

func TestGatewayScenarios(t *testing.T) {
	for _, tc := range []struct {
		engine, scenario, settings, method, space string
		status                                    int
	}{
		{"vllm", "normal", "{}", "selected-token-ids", "raw", 200},
		{"vllm", "selected-rejected", "{}", "top-k", "raw", 200},
		{"vllm", "selected-rejected", "{selected_token_ids: false, constraint: none}", "top-k", "raw", 200},
		{"sglang", "normal", "{}", "selected-token-ids", "raw", 200},
		{"sglang", "no-tokenizer", "{}", "top-k", "raw", 200},
		{"sglang", "topk-missing", "{selected_token_ids: false}", "teacher-forced-label", "raw", 200},
		{"sglang", "missing-candidates", "{prompt_token_logprobs: false}", "", "", 529},
		{"llama.cpp", "normal", "{}", "top-k", "raw", 200},
		{"llama.cpp", "topk-missing", "{}", "constrained-vocab", "post-mask", 200},
		{"llama.cpp", "missing-candidates", "{constraint: none}", "", "", 529},
	} {
		t.Run(tc.engine+"/"+tc.scenario+"/"+tc.settings, func(t *testing.T) {
			mock, upstream := start(t, tc.engine, tc.scenario)
			endpoint := "native_generate"
			if tc.engine == "vllm" {
				endpoint = "completions"
			}
			cfg, err := config.Parse([]byte(fmt.Sprintf(`
gateway: {default_model: test}
providers:
  - id: mock
    engine: %s
    base_url: %s
    endpoint: %s
    max_retries: 0
    models:
      - name: tiny-synthetic
        scoring: %s
model_aliases: {test: 'mock:tiny-synthetic'}
`, tc.engine, upstream.URL, endpoint, tc.settings)))
			if err != nil {
				t.Fatal(err)
			}
			gateway, err := httpapi.New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { gateway.Shutdown(context.Background()) })
			srv := httptest.NewServer(gateway.Handler())
			t.Cleanup(srv.Close)
			client := &http.Client{Timeout: 5 * time.Second}
			r, err := client.Post(srv.URL+"/v1/systemone", "application/json", strings.NewReader(`{"model":"test","state":"synthetic","questions":{"q":{"type":"choice","criteria":{"first":null,"second":null}}}}`))
			if err != nil {
				t.Fatal(err)
			}
			defer r.Body.Close()
			body, _ := io.ReadAll(r.Body)
			if r.StatusCode != tc.status {
				t.Fatalf("status=%d body=%s", r.StatusCode, body)
			}
			if tc.status != 200 {
				return
			}
			if ok, detail := gateway.Ready(); !ok {
				t.Fatalf("successful fallback route not ready: %s", detail)
			}
			if got := r.Header.Get("x-jev-scoring-method"); got != tc.method {
				t.Fatalf("method=%s want=%s body=%s", got, tc.method, body)
			}
			gotSpace := r.Header.Get("x-jev-probability-space")
			if gotSpace == "" {
				gotSpace = "raw"
			} // production omits the default
			if got := gotSpace; got != tc.space {
				t.Fatalf("space=%s want=%s", got, tc.space)
			}
			var result struct {
				Answers map[string]struct {
					Probabilities map[string]float64 `json:"probabilities"`
				} `json:"answers"`
			}
			if err := json.Unmarshal(body, &result); err != nil {
				t.Fatal(err)
			}
			// Numeric labels 1 and 2 have masses .15 and .05.
			if p := result.Answers["q"].Probabilities["first"]; math.Abs(p-0.75) > 1e-9 {
				t.Fatalf("P(first)=%g", p)
			}
			if tc.scenario == "selected-rejected" && tc.settings != "{}" {
				for _, req := range mock.Requests() {
					if req.Body["logprob_token_ids"] != nil {
						t.Fatal("disabled selected-ID field leaked onto wire")
					}
				}
			}
		})
	}
}

func TestModelSpecificScenariosAndSelectedLimit(t *testing.T) {
	m, err := mockinfer.New(mockinfer.Options{Engine: "vllm", ModelScenarios: map[string]string{"restricted": "selected-rejected"}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m)
	defer srv.Close()
	for model, status := range map[string]int{"standard": 200, "restricted": 400} {
		got, _ := post(t, srv.URL+"/v1/completions", fmt.Sprintf(`{"model":%q,"logprobs":1,"logprob_token_ids":[66,67]}`, model))
		if got != status {
			t.Fatalf("model=%s status=%d want=%d", model, got, status)
		}
	}
	ids := make([]int, 129)
	for i := range ids {
		ids[i] = i
	}
	body, _ := json.Marshal(map[string]any{"logprobs": 1, "logprob_token_ids": ids})
	if status, _ := post(t, srv.URL+"/v1/completions", string(body)); status != 400 {
		t.Fatal("129 selected IDs must be rejected")
	}
}
