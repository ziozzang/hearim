package config

import "testing"

func TestModelEntriesParsing(t *testing.T) {
	cfg, err := Parse([]byte(`
providers:
  - id: p
    engine: ollama
    base_url: http://x
    models:
      - gemma4:31b
      - name: qwen3:32b
        type: thinking
      - name: llava:13b
        type: vision
        extra_params:
          num_ctx: 16384
model_aliases:
  m: p:gemma4:31b
`))
	if err != nil {
		t.Fatal(err)
	}
	models := cfg.Providers[0].Models
	if len(models) != 3 {
		t.Fatalf("models = %+v", models)
	}
	if models[0].Name != "gemma4:31b" || models[0].Type != "" {
		t.Errorf("plain entry = %+v", models[0])
	}
	if models[1].Type != "thinking" {
		t.Errorf("thinking entry = %+v", models[1])
	}
	if models[2].Type != "vision" || models[2].ExtraParams["num_ctx"] != 16384 {
		t.Errorf("vision entry = %+v", models[2])
	}
	if got := cfg.Providers[0].Models.Names(); len(got) != 3 || got[0] != "gemma4:31b" {
		t.Errorf("names = %v", got)
	}
	if cfg.Providers[0].Models.Find("llava:13b") == nil {
		t.Error("Find failed")
	}
}

func TestUnknownModelTypeRejected(t *testing.T) {
	if _, err := Parse([]byte(`
providers:
  - id: p
    engine: ollama
    base_url: http://x
    models:
      - name: m
        type: psychic
model_aliases:
  a: p:m
`)); err == nil {
		t.Error("unknown type should be rejected")
	}
}

func TestProviderExtraParamsAndPool(t *testing.T) {
	cfg, err := Parse([]byte(`
providers:
  - id: pool
    engine: vllm
    base_urls: ["http://a:8000", "http://b:8000"]
    extra_params:
      seed: 42
    models: ["m"]
model_aliases:
  m: pool:m
`))
	if err != nil {
		t.Fatal(err)
	}
	p := cfg.Providers[0]
	if p.BaseURL != "" && len(p.BaseURLs) != 2 {
		t.Errorf("pool = %v", p.BaseURLs)
	}
	if p.ExtraParams["seed"] != 42 {
		t.Errorf("extras = %v", p.ExtraParams)
	}
}

func TestModelAliasChains(t *testing.T) {
	cfg, err := Parse([]byte(`
providers:
  - id: p1
    engine: ollama
    base_url: http://x
  - id: p2
    engine: vllm
    base_url: http://y
model_aliases:
  a: p1:gemma4:31b
model_alias_chains:
  resilient: ["p1:gemma4:31b", "p2:gemma-4-31b"]
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.ModelAliasChains["resilient"]) != 2 {
		t.Fatalf("chain = %v", cfg.ModelAliasChains)
	}
	// Unknown provider in a chain is rejected.
	if _, err := Parse([]byte(`
providers:
  - id: p
    engine: ollama
    base_url: http://x
model_aliases:
  a: p:m
model_alias_chains:
  bad: ["nowhere:m"]
`)); err == nil {
		t.Error("unknown chain provider not rejected")
	}
}
