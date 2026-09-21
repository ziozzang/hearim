# Provider & model capability notes (measured)

> 영어가 기본 문서 언어이며, 전체 한국어 번역은 [PROVIDERS.ko.md](PROVIDERS.ko.md)에 있습니다.
>
> Everything here was **measured with `hearim probe` and direct API calls**,
> not copied from documentation. Capabilities drift — re-run `hearim probe`
> before trusting a route; the readiness gate keeps unverified routes out
> of rotation automatically. Last survey: **2026-09-21**.

## Summary matrix

| Engine / host | logprobs surface | tokenizer | vision | notes |
|---|---|---|---|---|
| Ollama local 0.24.0 (chat) | ✅ `choices[].logprobs.content[].top_logprobs` (cap 20, choice-level) | ❌ (probe-only) | ✅ | the exact Ollama route |
| Ollama local 0.24.0 (completions) | ❌ accepts `logprobs` int, returns none | — | — | |
| Ollama local 0.24.0 (native generate) | ❌ | ✅ `/api/tokenize`? see below | — | returns token ids in `context` |
| Ollama Cloud (all models tested) | ❌ chat & completions, text & vision | — | ✅ inference only | answers are correct; no logprobs → no exact eval today |
| vLLM | expected via `logprob_token_ids` (verify per version) | ✅ `/tokenize` | ✅ | direct candidate-ID path |
| SGLang | expected via `token_ids_logprob` | ✅ gateway `/v1/tokenize` | ✅ | prefix-affine routing recommended |
| llama.cpp | expected via `n_probs` on `/completion` | ✅ `/tokenize` | model-dependent | `cache_prompt`, `tokens_cached` observability |

## Ollama Cloud models (measured 2026-09-21)

Vision+tools+thinking+cloud tagged models: `gemma4` (e2b–31b), `qwen3.5`
(0.8b–122b), `glm-5.3-flash`, `deepseek-v4.1-flash`, `minimax-m3`,
`kimi-k2.6`, `kimi-k3`, `kimi-k2.7-code`.

| Model | image input | 1-token answer | logprobs |
|---|---|---|---|
| gemma4:31b | ✅ (296 prompt tokens for a 168-byte PNG) | `1` — correct red/blue discrimination | ❌ |
| deepseek-v4.1-flash | ✅ (234 tokens) | `` (thinking consumes the token) | ❌ |
| glm-5.3-flash | ✅ (190 tokens) | `` | ❌ |
| kimi-k3 | ✅ (190 tokens) | `` | ❌ |
| minimax-m3 | ✅ (217 tokens) | `` | ❌ |

Reading: **vision inference works on cloud; the logprob surface does not
exist.** hearim therefore keeps cloud routes out of exact rotation until a
probe sees logprobs. Thinking VLMs return empty visible content at
`max_tokens: 1` — generation-based fallbacks would need `thinking.wait_close`
with a real token budget, and still have no logprobs to read.

## GLM via z.ai (measured 2026-09-21)

Endpoint: `https://api.z.ai/api/coding/paas/v4` (coding subscription
surface, OpenAI-compatible). Models: glm-4.5 … glm-5.3-flash(x).

- Answers are **correct** (even/odd → `1`), but all models are always-on
  reasoners: output lands in `reasoning_content` while `content` stays
  empty until reasoning finishes — `max_tokens: 1` is consumed by
  reasoning (the §3.4 hidden-reasoning hazard, with
  `usage.completion_tokens_details.reasoning_tokens` to observe it).
- `thinking: {"type": "disabled"}` is ignored on this surface.
- **`logprobs` is silently ignored** — requested and dropped, on every
  model, chat and full generations alike.

Follow-up measurements (same day, both endpoints):

- The **standard endpoint** (`https://api.z.ai/api/paas/v4`) accepts the
  coding key and — unlike the coding endpoint — **honors
  `thinking: {"type": "disabled"}`**: glm-4.5-air answers immediately with
  visible content. glm-5.3-flash rejects it outright ("always engages in
  thinking and cannot be disabled"). Prefer the standard endpoint for
  hearim routes.
- `logprobs` remains absent on both endpoints and on every model — no
  client-side parameter shape changes that (also confirmed against
  Ollama's own compatibility matrix: logprobs and `echo` are marked
  unsupported on both chat and completions; only local 0.24.0's chat
  surface implements it ahead of the docs).

Verdict: same class as Ollama Cloud — inference fine, no logprob surface,
so no exact evaluation today. The moment z.ai or Ollama Cloud exposes
logprobs, `engine: generic-openai` + `endpoint: chat_completions` +
`thinking: {disable_field: thinking, disable_value: {type: disabled}}`
attaches with zero code changes. The object-valued control is covered by
a regression test.

## OpenRouter (measured 2026-09-21) — works

`https://openrouter.ai/api/v1`, OpenAI-compatible. Per-model passthrough:

| Model | logprobs | notes |
|---|---|---|
| `openai/gpt-4o-mini` | ✅ top-10, sharp (`1` −0.0 / `0` −11.25) | full pipeline verified through `/v1/systemone` |
| `meta-llama/llama-3.3-70b-instruct` | ❌ (provider-side) | check per model |

Verified end-to-end through hearim as a generic provider with custom
headers (`HTTP-Referer`, `X-Title` — our `headers:` config), forced
`chat_completions` endpoint, probe-only registry (OpenRouter has no
tokenizer endpoint): noul even/odd → 0.9999999998; choice routing →
billing/delivery 0.5/0.5 (honest ambiguity). Recommended config:

```yaml
providers:
  - id: openrouter
    engine: generic-openai
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    endpoint: chat_completions
    headers: {HTTP-Referer: "https://github.com/ziozzang/hearim", X-Title: hearim}
    models: [{name: openai/gpt-4o-mini}]
```

## Model-card survey: thinking control differs per model

- **gemma4** — thinking toggles via a `<|think|>` token at the start of the
  **system prompt**. Non-edge models *still emit the tag structure when
  disabled* (`<|channel>thought … <channel|>`, empty block). Configure:
  ```yaml
  models:
    - name: gemma4:31b
      prompt: {system_prefix: "<|think|>"}   # only to force thinking ON
      thinking: {close_tag: "<channel|>"}
  ```
- **gpt-oss:20b** — `reasoning_effort` low/medium/high only; cannot be fully
  disabled → no exact chat route (§3.4).
- **deepseek-v4.1-flash** — card documents no control at all.
- **qwen3-family** — `<think>…</think>` blocks; `think: false` on Ollama
  native, or preload `close_tag: "</think>"`.

`hearim probe` verifies each configured control against the live model and
reports `thinking_control_verified` / `thinking_tags_emitted`.

## Vision findings (local gemma3:4b, Ollama 0.24.0)

- Full pipeline verified end-to-end through `/v1/systemone`: red/blue
  discrimination `noul` = 0.755 / 0.012 / 0.012 (red-is-red, blue-is-red,
  red-is-blue) with conditional label logprobs.
- **State-text redaction matters**: with the base64 data URL left in the
  state text block, label logprobs flipped to the wrong direction
  (0.001–0.003 on all three questions). hearim now redacts image values to
  `<image:N>` placeholders in the rendered text and carries the image only
  as a content part (519 → 375 prompt tokens for the test image).
- Small VLMs answer free text (`red`) more readily than labels; that is
  fine — hearim scores the conditional distribution over declared labels,
  never the sampled token. Budget for low `candidate_mass` on vision chat
  routes (`scoring.min_candidate_mass`).
- **Text models accept images silently** (observed with qwen2.5:0.5b) and
  return confident garbage — which is why hearim requires an explicit
  `models[].type: vision` for image-bearing states (422 otherwise).
- moondream on Ollama 0.24.0 returns empty output even on native `/api/chat`
  — model/server incompatibility, not a hearim issue.

## What to configure per provider

```yaml
providers:
  - id: ollama-local
    engine: ollama
    base_url: http://localhost:11434
    request_timeout: 180s        # vision prefill + model load can be slow
    models:
      - name: gemma3:4b
        type: vision
```

See README "Arbitrary providers" for endpoint forcing, path remapping,
query params, headers, and cost-header passthrough.
