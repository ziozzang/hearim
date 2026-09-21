# hearim (헤아림)

**A Jev-compatible multi-backend System One gateway, written in Go.**

hearim exposes the TypeSafe AI **Jev** `POST /v1/systemone` contract over
ordinary autoregressive backends — Ollama Cloud, llama.cpp, vLLM, SGLang, and
generic OpenAI-compatible servers — by compiling `Choice` / `Score` / `Noul`
questions into **single-token multiple-choice problems** and restoring a
probability distribution from the candidate labels' logprobs.

It never uses generated text as the basis of an answer: one question maps to
exactly one next-token scoring call, and when the label fast path is not
possible it teacher-forces the label or the full choice continuation instead.

### The name

**헤아림** is the nominal form of the Korean verb **헤아리다**, which carries
two intertwined meanings: *to count (things) one by one*, and *to grasp or
comprehend something by thinking it through*. That double sense is exactly
what this gateway does — it turns the act of understanding ("does this state
mean the customer wants a refund?") into an act of counting: label logprobs
measured at one decode position, then normalized. hearim counts, so callers
can understand — with honest numbers attached.

> 영어가 기본 문서 언어이며, 전체 한국어 번역은 [README.ko.md](README.ko.md)에 있습니다.

---

## How it works

```
Client
  -> SystemOne schema validator          (422 with per-question detail)
  -> canonicalizer                       (RFC 8785 JCS: stable prompt bytes)
  -> model router                        (alias -> provider/engine/model)
  -> EvaluationPlan compiler             (state-major / rubric-major prompt)
  -> prefix-aware scheduler              (per-provider lanes, prefix affinity)
  -> ProviderAdapter                     (Ollama / llama.cpp / vLLM / SGLang)
  -> logprob reducer                     (conditional softmax + entropy confidence)
  -> Jev-compatible response
```

### Probability restoration

For a four-option choice, the model sees:

```text
A = approve
B = review further
C = reject
D = cannot decide

<answer-label>
```

hearim requests `max_tokens: 1` with `logprobs`, reads the label logprobs
`l_A … l_D`, and publishes the conditional distribution:

```text
p(A) = exp(l_A) / (exp(l_A) + exp(l_B) + exp(l_C) + exp(l_D))
```

`confidence = 1 − H(p)/ln(K)` (normalized entropy). Both are documented
implementation choices — wire-compatible with Jev, not guaranteed numerically
identical to TypeSafe's model. Diagnostics ride on response headers
(`x-jev-implementation`, `x-jev-backend-model`, `x-jev-question-calls`,
`x-jev-cache-plan`, `x-jev-confidence-method`, `x-jev-scoring-method`,
`x-jev-probability-space`, `x-jev-calibration-profile`), never in the body.

### Scoring strategy chain

Per question, adapters try in order (TODO.md §3.11 / §7.3):

1. **selected-token-ids** — direct candidate token-ID logprob request
   (vLLM `logprob_token_ids`, SGLang `token_ids_logprob`)
2. **top-k** — top-N logprobs with text/bytes matching, one retry with a
   larger N
3. **teacher-forced-label** — input-token logprob of the label token
   (must agree with the next-token logprob on stable labels)
4. **constrained-vocab** — grammar / `allowed_token_ids` mask; the result is
   marked `x-jev-probability-space: post-mask`
5. **choice-text continuation** — full choice string scoring with
   `sum` / `token-mean` / `pmi` normalization and a per-mode calibration
   temperature τ

In strict mode (default), unrecoverable candidates fail the request with
`529 backend_probability_unavailable`. Missing candidates are never assumed
to have probability 0.

### Token label registry

At boot, per model and endpoint, hearim verifies which labels are stable
single tokens in the exact answer-marker context: it tokenizes
`prefix + delimiter + label`, requires `tokenize(prefix)` to be an exact
prefix with exactly one extra token, tries delimiter candidates until the
whole alphabet qualifies, and confirms with a real 1-token completion probe.
Registries are persisted under `data/registries/` keyed by
model/digest/tokenizer/template/endpoint/delimiter, and invalidated when any
of those change. Boundary re-segmentation is handled with exact-LCP healing
(min over candidates of the longest common prefix), not fixed backtracking.

### Fan-out and caching

One Jev request fans out to one upstream scoring call per question, grouped
by shared prefix on bounded lanes (Ollama Pro: 3). The scheduler prefers
currently-running prefixes, then recently warm prefixes, then FIFO, so the
shared state prefill stays temporally adjacent and cache-friendly. The prefix
key is `SHA256(model_digest ‖ tokenizer_revision ‖ template_version ‖
layout ‖ prefix_bytes)` — aliases, timestamps, request ids, and trace ids
never enter prompts.

---

## Quick start

```bash
# build
go build -o hearim ./cmd/hearim

# configure (never put API keys in the file; env expansion is supported)
cp hearim.example.yaml hearim.yaml
export OLLAMA_API_KEY=...        # from your secret store, e.g. ~/env
export HEARIM_API_KEY=...        # key clients must present

# probe the Phase 0 capability gate before trusting a backend
./hearim probe -config hearim.yaml -provider ollama-cloud -model gemma4:31b

# serve
./hearim serve -config hearim.yaml -addr :8080
```

### Evaluate

```bash
curl -s localhost:8080/v1/systemone \
  -H "Authorization: Bearer $HEARIM_API_KEY" \
  -H "Content-Type: application/json" -d '{
    "model": "jev-gemma4",
    "state": {"ticket": "paid but never received the download link", "account_age_days": 820},
    "questions": {
      "routing": {
        "type": "choice",
        "instructions": "Pick the most appropriate handling queue.",
        "criteria": {
          "billing": "payment, refund, or invoice issues",
          "delivery": "product or digital content delivery issues",
          "account": "login or account issues",
          "other": null
        }
      }
    }
  }'
```

Response (shape per the Jev contract; probabilities are conditional over the
declared candidates):

```json
{
  "model": "jev-gemma4",
  "answers": {
    "routing": {
      "type": "choice",
      "choice": "delivery",
      "probabilities": {"billing": 0.08, "delivery": 0.86, "account": 0.02, "other": 0.04},
      "confidence": 0.66
    }
  },
  "usage": {"input_tokens": 243, "output_tokens": 1}
}
```

All three primitives are supported:

| type | semantics | response |
|---|---|---|
| `choice` | one of 2–255 named options | `choice`, full `probabilities`, `confidence` |
| `score` | ordinal scale of 2–10 levels | probability-weighted `score`, `legend`, `probabilities`, `confidence` |
| `noul` | true/false proposition | `noul` = P(true) |

### OpenAI-compatible surface

`POST /v1/completions` and `POST /v1/chat/completions` are transparent
proxies to the default route's provider. hearim deliberately does not guess
Jev evaluations out of chat traffic. A non-standard `jev_candidates`
extension exists on `/v1/completions` for advanced use:

```json
{
  "model": "jev-gemma4",
  "prompt": "...\nANSWER:",
  "max_tokens": 1,
  "logprobs": 10,
  "jev_candidates": {"A": "billing", "B": "delivery", "C": "account", "D": "other"}
}
```

The public product API is `/v1/systemone`; the extension is not part of the
OpenAI standard.

### Operations endpoints

- `GET /healthz` — liveness
- `GET /readyz` — fails when no default route has a ready registry
- `GET /v1/routes` — routes, endpoints, registries, cache plans
- `GET /metrics` — Prometheus text format (auth required)

---

## CLI

| command | purpose |
|---|---|
| `hearim serve` | run the gateway |
| `hearim probe` | Phase 0 capability suite: tokenizer endpoint, logprob shapes, single-token labels, teacher-forced parity, hidden-reasoning and cache evidence; prints a JSON report and exits non-zero when every target fails the go/no-go gate |
| `hearim bench` | run a labeled JSONL corpus: accuracy, macro F1, NLL, Brier, ECE, candidate mass, p50/p95 latency, and label-order permutation variance (`-permutations N`) |
| `hearim update` | self-update from GitHub releases with SHA256SUMS verification |

Bench corpus format:

```jsonl
{"state": "...", "question": {"type": "choice", "criteria": {"a": "...", "b": "..."}}, "expected": "a"}
{"state": "...", "question": {"type": "noul", "criteria": {"true": "...", "false": "..."}}, "expected": true}
{"state": "...", "question": {"type": "score", "criteria": ["low", "high"]}, "expected": 1}
```

## Model routing and cost

The router picks a `(provider, engine, endpoint, model)` tuple, not just a
model name. `policy:auto-v1` aliases resolve through the §9 cost rule: at or
above the break-even cached-input fraction, the high-cache-rate candidate
wins on input cost (gemma4:31b vs deepseek-v4.1-flash breaks even at
h ≈ 0.1754); below it, the cheapest quality-gated candidate. Quality gates
are explicit (`quality_gate_pass`); price alone never decides. The budget
guard throttles at 70%/90% and hard-stops at 100% of the included allowance.

### Model name resolution

A request's `model` field resolves in this order:

1. configured alias (`model_aliases`) or fallback chain
   (`model_alias_chains`, first bound route wins)
2. explicit `provider:model` (when the prefix names a configured provider)
3. **bare backend model name** — `"gemma4:31b"` resolves to the provider
   that serves it; ambiguous names hosted by several providers are an
   explicit error

### Multi-backend configuration

- `providers[].base_urls` — an endpoint pool for one engine: requests
  round-robin across replicas and retries fail over to the *next* host, so a
  dead replica is bypassed instead of retried in place.
- `model_alias_chains` — ordered fallback: `{"resilient":
  ["ollama-cloud:gemma4:31b", "vllm-local:google/gemma-4-31B-it"]}`.
- `models` entries carry upstream metadata:

```yaml
models:
  - gemma4:31b                          # plain entry
  - name: qwen3:32b
    type: thinking                      # reasoning can't be fully disabled:
                                        # chat exact routes are vetoed (§3.4)
  - name: llava:13b
    type: vision                        # accepts image inputs
    extra_params: {num_ctx: 16384}      # extra JSON merged upstream
```

- `extra_params` (provider or model level, model wins per key) merge
  additional JSON fields into every upstream request — engine-specific knobs
  like `seed`, `num_ctx`, or gateway-specific parameters. Correctness-
  critical fields (`logprobs`, `max_tokens`, sampler identity, ...) are
  protected and cannot be overridden.

### Thinking models (`<think>` handling)

Model cards document how (and whether) a model's reasoning can be turned
off; hearim maps that onto three techniques, configured per model:

```yaml
models:
  - name: qwen3:32b
    type: thinking
    thinking:
      disable_field: think          # model-card control, applied verbatim
      disable_value: false          # (e.g. reasoning_effort: "none")
      close_tag: "</think>"         # preload technique (exact path)
  - name: r1-style:14b
    type: thinking
    thinking:
      close_tag: "</think>"
      wait_close: true              # scan technique (approximate)
      max_think_tokens: 512
```

1. **disable** — the card's own switch (`think: false`,
   `reasoning_effort: "none", ...) is applied to every upstream request,
   overriding the engine default.
2. **close-tag preload** — for models that always open a `<think>` block,
   the closing tag is preloaded right after the answer marker
   (`...</answer-label>\n</think>\n`), so the very next token is the answer
   label. This keeps the exact single-decode-position semantics: the token
   label registry probes boundaries in this exact post-tag context.
3. **wait-close scan** — when preloading is not possible: generate through
   the reasoning block (bounded by `max_think_tokens`, default 256) and read
   the logprob distribution at the first position **after** `</think>`.
   Marked `x-jev-scoring-method: wait-close-tag`. This is approximate —
   the distribution is conditioned on the sampled reasoning text — so
   calibration profiles treat it as its own mode. If the block never
   closes, the question fails rather than scoring a reasoning token.

Without a `thinking` block, `type: thinking` models simply have no exact
chat route (TODO.md §3.4) unless the engine can fully disable reasoning.

### Vision (VLM) image inputs

State objects may declare images via top-level `image` (single URL) or
`images` (array). Only `https?://` and `data:image/...` URLs are accepted —
bare paths and other schemes are rejected at validation (state is data, not
a license to read local files). Up to 8 images, 20 MB per data URL. On
vision-capable chat routes the state message becomes OpenAI multimodal
content parts (`image_url`); non-vision routes reject image-bearing states
with 422.

### Observability

- `GET /metrics` — Prometheus text format (behind the same auth as the API):
  request counters and duration histograms, per-model question calls by
  scoring method and probability space, token counters split into
  input/output/cached-reported, `hearim_prefill_cache_ratio` per route
  (computed from *reported* cached tokens only — §8.5), budget utilization,
  route readiness gauges, and basic Go runtime gauges.
- Response header `x-jev-cached-input-tokens` carries the server-reported
  cached token count for the request; `x-jev-cache-plan` names the engine's
  cache strategy.

### Self-update

`hearim update` replaces the running binary with a GitHub release build,
verifying the SHA-256 from the release's `SHA256SUMS` asset first (design
followed from [hftools](https://github.com/ziozzang/hftools)). `update
-check` only reports; `-version v0.2.0` pins a release; `-force` reinstalls.
Interactive runs also print a once-a-day update notice when a newer release
exists — disable with `HEARIM_NO_UPDATE_CHECK=1`.

## Security notes

- State is data, not instructions: length-delimited blocks with escaped
  closing tags; prompt injection cannot be fully eliminated, only contained.
- API keys, tenant ids, trace ids never enter prompts.
- Raw state logging is off by default (hash + token counts only).
- Upstream secrets stay in environment variables; config files reference
  `${VAR}` / `${VAR:-default}`.
- Model output is not a sole basis for high-risk decisions (authorization,
  payment approval, ...).

## Development

```bash
go build ./...
go test ./...
go vet ./...
```

Package map (all under `internal/hearim/`):

| package | role |
|---|---|
| `canon` | RFC 8785 JCS canonicalization with I-JSON validation |
| `config` | YAML schema, defaults, validation |
| `jev` | request/response contract, semantic validation, error surface |
| `scoring` | softmax, candidate mass, entropy confidence, LCP healing, continuation modes |
| `compile` | EvaluationPlan, state-major/rubric-major templates, label mapping, prefix keys |
| `provider` | Adapter interface, route resolution, Ollama/llama.cpp/vLLM/SGLang/generic adapters |
| `registry` | token label registry: boot probing, persistence, readiness |
| `schedule` | prefix-affine lane scheduler |
| `eval` | strategy chain orchestration and reductions |
| `router` | alias/policy resolution and cost math |
| `usage` | usage aggregation, cost ledger, budget guard |
| `httpapi` | server assembly, handlers, proxy, health |
| `probe` | Phase 0 capability suite |
| `bench` | §12.3 benchmark runner |
| `selfupdate` | GitHub release self-update with SHA256SUMS verification and background update notices |
| `metrics` | minimal Prometheus text-exposition registry (stdlib only) |

## Probed capabilities (2026-09-21, real backends)

Phase 0 probes against live services shaped the defaults:

- **Ollama (cloud + local 0.24.0)**: `/v1/completions` accepts `logprobs`
  (int) but returns **no logprobs at all**; `/v1/chat/completions` returns
  full `choices[].logprobs.content[].top_logprobs` (capped at 20). hearim
  therefore routes Ollama scoring through chat (§3.4), with probe-only
  tokenization. GPT-OSS on Ollama (low/medium/high effort only) has no
  exact Ollama route.
- **Local e2e verified** with `qwen2.5:0.5b` on Ollama 0.24.0: registry
  built (numeric labels verified), `/v1/systemone` choice/score/noul
  answers with headers, usage, and fan-out — a small model shows weak
  label compliance (low candidate mass is caught by the floor), so use a
  real evaluation model for production calibration.
- Cloud `ollama.com/v1` three-model sweep (gemma4:31b, deepseek-v4.1-flash,
  gpt-oss:20b): no logprobs on either completions or chat at probe time —
  re-run `hearim probe` before trusting a cloud route; the readiness gate
  keeps unverified routes out of rotation.

## Compatibility disclaimers

hearim implements the **API contract** of TypeSafe AI Jev. It does not
reproduce Jev's model, probability calibration, or latency. The conditional
softmax, normalized-entropy confidence, and per-route calibration profiles
are hearim's own documented methods — validate thresholds against your own
labeled data (`hearim bench`).

## License

TBD
