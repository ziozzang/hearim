# Scoring settings by engine and model

Settings are resolved in order: engine defaults → `providers[].scoring` →
`providers[].models[].scoring`. A model overrides only the fields it specifies.
Explicit `false` is preserved. The settings affect request generation, route
capabilities, fallback selection, and registry/calibration identity.

```yaml
providers:
  - id: inference
    engine: vllm
    base_url: http://localhost:8000
    scoring:
      selected_token_ids: true
      max_selected_token_ids: 128
      prompt_token_logprobs: true
      constraint: allowed_token_ids
      logprob_space: raw
    models:
      - name: standard-model
      - name: restricted-model
        scoring:
          selected_token_ids: false
          prompt_token_logprobs: false
          constraint: none
model_aliases:
  standard: inference:standard-model
  restricted: inference:restricted-model
```

The restricted route uses top-k scoring without selected-ID fields, prompt
logprob requests, or a candidate mask. If it cannot recover every candidate,
strict mode returns an error. Setting `constraint: none` alone leaves
selected-ID logprob retrieval and teacher forcing available: retrieving a
token's probability and restricting which token can be sampled are separate
features.

| Setting | Meaning |
|---|---|
| `selected_token_ids` | Enable/disable direct candidate-ID logprob requests. |
| `max_selected_token_ids` | Positive request limit for the deployed server; omitted/0 inherits. Larger sets skip direct retrieval and try other available scoring methods. This does not increase the server's limit. |
| `prompt_token_logprobs` | Enable/disable teacher-forced label/continuation scoring. Implemented for vLLM and native SGLang. |
| `constraint` | `none`; `allowed_token_ids` for vLLM or explicitly configured generic OpenAI servers; `grammar` for native llama.cpp. Unsupported engine combinations are configuration errors. |
| `logprob_space` | `raw` or `post-mask` for vLLM/generic routes. Describes the server's returned logprob mode; it does **not** change that server setting. Raw logits are not logprobs and are unsupported. |

## Engine defaults and wire formats

| Engine | Selected-ID defaults | Candidate constraint | Reported probability space |
|---|---|---|---|
| vLLM | enabled, up to 128 IDs | `allowed_token_ids` | `raw`, including constrained sampling with the default `raw_logprobs` mode |
| SGLang native | enabled, configurable request limit 256 | `none` | `raw` |
| llama.cpp native | disabled | `grammar` | `raw` normally; explicit `post_sampling_probs: true` and `top_probs` for constrained scoring |
| generic OpenAI | disabled unless `selected_token_field` is set or explicitly enabled | `none` | `raw` |
| Ollama | disabled | `none` | `raw` |

SGLang OpenAI-compatible routes use top-k; native selected-ID and prompt-logprob
support is not assumed on those endpoints. llama.cpp grammar scoring is native
only. Use `endpoint: native_generate` for the native SGLang/llama.cpp features.

OpenAI SDK `extra_body={"allowed_token_ids": [...]}` flattens the field into
the HTTP JSON body. It is not a cross-engine standard. hearim generates
engine-specific fields itself; protected fields cannot be overridden through
`extra_params`. For native SGLang, setting `allowed_token_ids` is unsupported
even inside `sampling_params`. Use selected-ID **retrieval** or top-k instead.

For vLLM, `--max-logprobs` and the selected-ID limit are separate. Increasing
one does not increase the other. If a server/model rejects or ignores the
mask, configure `constraint: none`. If it does not support selected retrieval,
set `selected_token_ids: false`. Defaults are based on the source revisions
in the [audit](PROVIDER-SOURCE-REVIEW.md), not a guarantee about every build.

llama.cpp grammar mode explicitly disables top-k/top-p/min-p truncation in the
sampling chain, uses temperature 1, and requires the post-sampling response
shape. A server returning only raw `top_logprobs` fails that attempt rather
than being mislabeled post-mask. Special-token labels still need a tokenizer
boundary and serving probe; a text grammar is not an arbitrary token-ID mask.

The registry persistence key includes the resolved scoring profile, so changes
trigger rebuilding instead of reusing measurements under different settings.
Calibration IDs include the profile and returned probability space as well.
Run `hearim probe` for each actual deployment. Regression tests validate
request routing, source-shaped responses, and fallback behavior; they do not
establish model accuracy or live server conformance.
