package config

import (
	"os"
	"testing"
	"time"
)

const fullYAML = `
gateway:
  public_endpoint: /v1/systemone
  compatibility: typesafe-systemone-v1
  default_model: jev-local-auto
  strict_candidate_probabilities: false
  confidence_method: normalized-entropy-v1

server:
  addr: 127.0.0.1:9090

auth:
  api_keys: ["${TEST_KEY}", "static-key"]

log:
  level: debug
  format: json

ollama:
  base_url: https://ollama.com/v1
  concurrency: 3
  max_retries: 1

providers:
  - id: ollama-cloud
    engine: ollama
    base_url: https://ollama.com/v1
    endpoint_preference: [completions, chat_completions]
  - id: llama-cpp-local
    engine: llama.cpp
    base_url: http://llama-server:8080
    endpoint_preference: [native_generate, completions]
    cache_prompt: true
  - id: vllm-local
    engine: vllm
    base_url: http://vllm:8000
    selected_token_field: logprob_token_ids
    prefix_caching: automatic
  - id: sglang-local
    engine: sglang
    base_url: http://sglang:30000
    selected_token_field: token_ids_logprob
    prefix_caching: radix

model_aliases:
  jev-local-auto: ollama-cloud:gemma4:31b
  jev-gemma4: ollama-cloud:gemma4:31b
  jev-deepseek-flash: ollama-cloud:deepseek-v4.1-flash
  jev-gpt-oss-20b: ollama-cloud:gpt-oss:20b
  jev-vllm: vllm-local:google/gemma-4-31B-it

compiler:
  template_version: systemone-v1
  default_layout: state-major
  build_token_registry_on_startup: true
  label_alphabets:
    - ["1","2","3","4","5","6","7","8","9","0"]
    - ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

scoring:
  primary: selected_token_ids
  fallbacks: [top_k, teacher_forced_label, constrained_vocab]
  choice_text_mode: disabled
  continuation_normalization: pmi
  token_healing: exact_lcp
  calibration_scope: model-engine-template-mode

router:
  candidates:
    - model: gemma4:31b
      input_per_1m: 0.14
      cached_input_per_1m: 0.05
      output_per_1m: 0.40
    - model: deepseek-v4.1-flash
      input_per_1m: 0.15
      cached_input_per_1m: 0.003
      output_per_1m: 0.60
    - model: gpt-oss:20b
      input_per_1m: 0.07
      cached_input_per_1m: 0.035
      output_per_1m: 0.30
  cache_ratio_break_even_gemma_vs_deepseek: 0.1754
  require_quality_gate: true

budget:
  included_usd: 60
  warn_at: 0.70
  throttle_at: 0.90
  hard_stop_at: 1.00
`

func TestParseFullConfig(t *testing.T) {
	os.Setenv("TEST_KEY", "env-key")
	defer os.Unsetenv("TEST_KEY")

	c, err := Parse([]byte(fullYAML))
	if err != nil {
		t.Fatal(err)
	}
	if c.Gateway.DefaultModel != "jev-local-auto" {
		t.Errorf("default model = %q", c.Gateway.DefaultModel)
	}
	if c.StrictMode() {
		t.Error("strict_candidate_probabilities: false was overridden")
	}
	if got := c.Auth.APIKeys[0]; got != "env-key" {
		t.Errorf("env expansion failed: %q", got)
	}
	if len(c.Providers) != 4 {
		t.Fatalf("providers = %d", len(c.Providers))
	}
	if c.Providers[1].CachePrompt == nil || !*c.Providers[1].CachePrompt {
		t.Error("llama.cpp cache_prompt should persist true")
	}
	if c.Providers[2].SelectedTokenField != "logprob_token_ids" {
		t.Errorf("vllm selected field = %q", c.Providers[2].SelectedTokenField)
	}
	if c.Router.CacheRatioBreakEven != 0.1754 {
		t.Errorf("break-even = %v", c.Router.CacheRatioBreakEven)
	}
	if !*c.Router.RequireQualityGate {
		t.Error("require_quality_gate must default/persist true")
	}
}

func TestDefaults(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  - id: local
    engine: ollama
    base_url: http://localhost:11434/v1
model_aliases:
  jev-local: local:gemma4:31b
`))
	if err != nil {
		t.Fatal(err)
	}
	if !c.StrictMode() {
		t.Error("strict mode must default true")
	}
	if !*c.BackendPolicy.RequireNoReasoningForChat {
		t.Error("require_no_reasoning_for_chat must default true")
	}
	if !*c.BackendPolicy.RejectPromptOnlyReasoningSuppression {
		t.Error("reject_prompt_only_reasoning_suppression must default true")
	}
	if !*c.Compiler.BuildTokenRegistryOnStartup {
		t.Error("registry build must default true")
	}
	if c.Ollama.Concurrency != 3 {
		t.Errorf("ollama concurrency default = %d", c.Ollama.Concurrency)
	}
	if c.Server.ReadHeaderTimeout != 10*time.Second {
		t.Errorf("read header timeout = %v", c.Server.ReadHeaderTimeout)
	}
	if len(c.Compiler.LabelAlphabets) != 2 || len(c.Compiler.LabelAlphabets[0]) != 10 {
		t.Errorf("default alphabets = %v", c.Compiler.LabelAlphabets)
	}
	if c.Compiler.Temperature != 1 || c.Compiler.TopP != 1 || c.Compiler.MaxTokens != 1 {
		t.Errorf("sampling defaults = %v/%v/%v", c.Compiler.Temperature, c.Compiler.TopP, c.Compiler.MaxTokens)
	}
	if c.Scoring.Primary != ScoringSelectedTokenIDs {
		t.Errorf("primary = %v", c.Scoring.Primary)
	}
	if c.Budget.IncludedUSD != 60 || c.Budget.ThrottleAt != 0.90 {
		t.Errorf("budget defaults = %+v", c.Budget)
	}
	if c.Providers[0].Concurrency != 3 {
		t.Errorf("ollama provider concurrency = %d", c.Providers[0].Concurrency)
	}
}

func TestExplicitPolicyOff(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  - id: local
    engine: vllm
    base_url: http://vllm:8000
backend_policy:
  require_no_reasoning_for_chat: false
model_aliases:
  j: local:m
`))
	if err != nil {
		t.Fatal(err)
	}
	if *c.BackendPolicy.RequireNoReasoningForChat {
		t.Error("explicit false must survive")
	}
}

func TestValidationErrors(t *testing.T) {
	cases := []struct{ name, yaml string }{
		{"unknown alias provider", `
model_aliases:
  j: nowhere:m
`},
		{"alias without colon", `
providers:
  - id: p
    engine: ollama
    base_url: http://x
model_aliases:
  j: justname
`},
		{"bad engine", `
providers:
  - id: p
    engine: warp
    base_url: http://x
`},
		{"bad temp", `
providers:
  - id: p
    engine: ollama
    base_url: http://x
compiler:
  temperature: 0
`},
		{"bad budget order", `
providers:
  - id: p
    engine: ollama
    base_url: http://x
budget:
  warn_at: 0.9
  throttle_at: 0.5
`},
		{"empty config", ``},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse([]byte(tc.yaml)); err == nil {
				t.Errorf("expected error for %s", tc.name)
			}
		})
	}
}

func TestUnknownFieldRejected(t *testing.T) {
	if _, err := Parse([]byte(`
providers:
  - id: p
    engine: ollama
    base_url: http://x
not_a_real_section: true
`)); err == nil {
		t.Error("unknown top-level key not rejected")
	}
}

func TestBreakEvenDerivation(t *testing.T) {
	// From TODO.md §9: 0.14(1-h) + 0.05h = 0.15(1-h) + 0.003h -> h ≈ 0.1754.
	var gemmaIn, gemmaCached = 0.14, 0.05
	var flashIn, flashCached = 0.15, 0.003
	// 0.14(1-h) + 0.05h = 0.15(1-h) + 0.003h
	//  -> (0.15-0.003)h - (0.14-0.05)h = 0.15 - 0.14
	h := (flashIn - gemmaIn) / ((flashIn - flashCached) - (gemmaIn - gemmaCached))
	if h < 0.1753 || h > 0.1755 {
		t.Errorf("break-even = %v, want ~0.1754", h)
	}
	// Below break-even gemma is cheaper; above, flash is cheaper.
	cost := func(cachedFrac, in, cin float64) float64 {
		return in*(1-cachedFrac) + cin*cachedFrac
	}
	if cost(0.10, gemmaIn, gemmaCached) >= cost(0.10, flashIn, flashCached) {
		t.Error("gemma should be cheaper below break-even")
	}
	if cost(0.30, gemmaIn, gemmaCached) <= cost(0.30, flashIn, flashCached) {
		t.Error("flash should be cheaper above break-even")
	}
}
