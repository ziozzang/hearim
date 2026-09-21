package provider

import (
	"hearim/internal/hearim/config"
)

// protectedFields are correctness-critical request fields that extra_params
// cannot override. hearim's evaluation semantics depend on their exact
// values (§5.3, §7.1); a typo in configuration must not silently break
// probability restoration.
var protectedFields = map[string]bool{
	// request shape
	"model": true, "prompt": true, "messages": true, "input_ids": true,
	"stream": true, "grammar": true,
	// one-token evaluation budget
	"max_tokens": true, "max_completion_tokens": true, "n_predict": true,
	"max_new_tokens": true, "num_predict": true,
	// sampler identity (raw distribution)
	"temperature": true, "top_p": true, "top_k": true,
	// logprob plumbing
	"logprobs": true, "top_logprobs": true, "n_probs": true, "min_keep": true,
	"logprob_token_ids": true, "token_ids_logprob": true,
	"return_logprob": true, "top_logprobs_num": true, "logprob_start_len": true,
	"prompt_logprobs": true, "return_tokens": true, "return_text_in_logprobs": true,
	"allowed_token_ids": true, "sampling_params": true,
	// cache control is engine policy, not per-model tuning
	"cache_prompt": true,
}

// ModelExtras merges provider-level extra_params with the model entry's
// extra_params (model wins per key).
func ModelExtras(cfg config.ProviderConfig, model string) map[string]any {
	merged := map[string]any{}
	for k, v := range cfg.ExtraParams {
		merged[k] = v
	}
	if mc := cfg.Models.Find(model); mc != nil {
		for k, v := range mc.ExtraParams {
			merged[k] = v
		}
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

// MergeExtras applies extra JSON parameters into an upstream request body;
// protected fields keep hearim's values.
func MergeExtras(body map[string]any, extras map[string]any) {
	for k, v := range extras {
		if protectedFields[k] {
			continue
		}
		body[k] = v
	}
}

// ModelIsThinking reports whether a model entry declares an upstream type
// whose reasoning cannot be fully disabled (chat exact-route veto, §3.4).
func ModelIsThinking(mc *config.ModelConfig) bool {
	return mc != nil && (mc.Type == "thinking" || mc.Type == "reasoning")
}
