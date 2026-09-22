package config

import (
	"fmt"
	"testing"
)

func TestProviderModelScoringConfig(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  - id: p
    engine: vllm
    base_url: http://x
    scoring:
      selected_token_ids: true
      max_selected_token_ids: 128
      constraint: allowed_token_ids
      logprob_space: raw
    models:
      - name: m
        scoring:
          selected_token_ids: false
          prompt_token_logprobs: false
          constraint: none
model_aliases: {a: 'p:m'}
`))
	if err != nil {
		t.Fatal(err)
	}
	p := c.Providers[0]
	if p.Scoring == nil || p.Scoring.SelectedTokenIDs == nil || !*p.Scoring.SelectedTokenIDs || p.Scoring.MaxSelectedTokenIDs != 128 {
		t.Fatalf("provider scoring=%+v", p.Scoring)
	}
	m := p.Models[0].Scoring
	if m == nil || m.SelectedTokenIDs == nil || *m.SelectedTokenIDs || m.PromptTokenLogprobs == nil || *m.PromptTokenLogprobs || m.Constraint != "none" {
		t.Fatalf("model scoring=%+v", m)
	}
}

func TestInvalidScoringProfiles(t *testing.T) {
	for _, tc := range []struct{ engine, scoring string }{
		{"sglang", "constraint: allowed_token_ids"},
		{"vllm", "constraint: grammar"},
		{"llama.cpp", "selected_token_ids: true"},
		{"vllm", "logprob_space: raw_logits"},
		{"vllm", "max_selected_token_ids: -1"},
		{"sglang", "logprob_space: post-mask"},
	} {
		for _, model := range []bool{false, true} {
			block := fmt.Sprintf("    scoring: {%s}\n", tc.scoring)
			if model {
				block = fmt.Sprintf("    models:\n      - name: m\n        scoring: {%s}\n", tc.scoring)
			}
			_, err := Parse([]byte(fmt.Sprintf("providers:\n  - id: p\n    engine: %s\n    base_url: http://x\n%smodel_aliases: {a: 'p:m'}\n", tc.engine, block)))
			if err == nil {
				t.Errorf("accepted engine=%s model=%v scoring=%s", tc.engine, model, tc.scoring)
			}
		}
	}
}
