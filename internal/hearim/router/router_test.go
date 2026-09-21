package router

import (
	"testing"

	"hearim/internal/hearim/config"
	"hearim/internal/hearim/eval"
)

func baseConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(`
providers:
  - id: p1
    engine: vllm
    base_url: http://p1
  - id: p2
    engine: sglang
    base_url: http://p2
model_aliases:
  jev-a: p1:gemma4:31b
  jev-auto: policy:auto-v1
router:
  candidates:
    - model: gemma4:31b
      provider: p1
      input_per_1m: 0.14
      cached_input_per_1m: 0.05
      output_per_1m: 0.40
      quality_gate_pass: true
    - model: deepseek-v4.1-flash
      provider: p2
      input_per_1m: 0.15
      cached_input_per_1m: 0.003
      output_per_1m: 0.60
      quality_gate_pass: true
    - model: gpt-oss:20b
      provider: p2
      input_per_1m: 0.07
      cached_input_per_1m: 0.035
      output_per_1m: 0.30
      quality_gate_pass: false
`))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestResolveAliasAndDirect(t *testing.T) {
	r := New(baseConfig(t))
	r.Bind("jev-a", eval.Route{ProviderID: "p1", BackendModel: "gemma4:31b"})
	route, err := r.Resolve("jev-a")
	if err != nil {
		t.Fatal(err)
	}
	if route.ProviderID != "p1" || route.BackendModel != "gemma4:31b" {
		t.Errorf("route = %+v", route)
	}
	if _, err := r.Resolve("nope"); err == nil {
		t.Error("unknown model should fail")
	}
}

func TestAutoPolicyCheapestGated(t *testing.T) {
	// Expected ratio 0.20: gemma 0.122, deepseek 0.1216, gpt-oss gated OFF.
	// deepseek wins on effective input cost.
	r := New(baseConfig(t))
	target := r.autoTarget()
	if target != "deepseek-v4.1-flash" {
		t.Errorf("auto target = %s, want deepseek-v4.1-flash", target)
	}
}

func TestAutoPolicyBelowBreakEvenPrefersHighCacheCandidate(t *testing.T) {
	cfg := baseConfig(t)
	cfg.Router.ExpectedCacheRatio = 0.0
	r := New(cfg)
	// At h=0 gemma (0.14) beats deepseek (0.15); gpt-oss is cheaper (0.07)
	// but gated off.
	if got := r.autoTarget(); got != "gemma4:31b" {
		t.Errorf("auto target = %s, want gemma4:31b", got)
	}
	// Ungate gpt-oss: cheapest wins.
	cfg.Router.Candidates[2].QualityGatePass = true
	r2 := New(cfg)
	if got := r2.autoTarget(); got != "gpt-oss:20b" {
		t.Errorf("auto target = %s, want gpt-oss:20b", got)
	}
}

func TestBreakEvenDocValue(t *testing.T) {
	cfg := baseConfig(t)
	gemma := cfg.Router.Candidates[0]
	flash := cfg.Router.Candidates[1]
	h, err := BreakEvenCacheRatio(gemma, flash)
	if err != nil {
		t.Fatal(err)
	}
	if h < 0.1753 || h > 0.1755 {
		t.Errorf("break-even = %v, want ~0.1754", h)
	}
	// §9: below break-even gemma cheaper, above flash cheaper.
	if EffectiveInputRate(gemma, 0.1) >= EffectiveInputRate(flash, 0.1) {
		t.Error("gemma should win below break-even")
	}
	if EffectiveInputRate(gemma, 0.3) <= EffectiveInputRate(flash, 0.3) {
		t.Error("flash should win above break-even")
	}
}

func TestBareModelNameResolution(t *testing.T) {
	// Backend model names contain colons ("gemma4:31b"); they must resolve
	// to the serving provider, not be misread as provider "gemma4".
	r := New(baseConfig(t))
	r.Bind("gemma4:31b", eval.Route{ProviderID: "p1", BackendModel: "gemma4:31b"})
	route, err := r.Resolve("gemma4:31b")
	if err != nil {
		t.Fatal(err)
	}
	if route.ProviderID != "p1" {
		t.Errorf("route = %+v", route)
	}

	// Ambiguity across providers is an explicit error.
	r2 := New(baseConfig(t))
	r2.Bind("a", eval.Route{ProviderID: "p1", BackendModel: "m"})
	r2.Bind("b", eval.Route{ProviderID: "p2", BackendModel: "m"})
	if _, err := r2.Resolve("m"); err == nil {
		t.Error("ambiguous bare model should error")
	}
}

func TestModelAliasChainFallback(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ModelAliasChains = map[string][]string{
		"resilient": {"p1:gemma4:31b", "p2:deepseek-v4.1-flash"},
	}
	r := New(cfg)
	// Only the second hop is bound: the first must be skipped.
	r.Bind("p2:deepseek-v4.1-flash", eval.Route{ProviderID: "p2", BackendModel: "deepseek-v4.1-flash"})
	route, err := r.Resolve("resilient")
	if err != nil {
		t.Fatal(err)
	}
	if route.ProviderID != "p2" {
		t.Errorf("chain did not fall through: %+v", route)
	}
	// Cached binding is stable afterward.
	route2, err := r.Resolve("resilient")
	if err != nil || route2.ProviderID != "p2" {
		t.Errorf("second resolve = %+v %v", route2, err)
	}
}
