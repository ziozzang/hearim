# CPU-only inference protocol scenarios

`hearim-mock` is a Go HTTP server that exercises the hearim adapters without
GPU access, Python inference dependencies, or model downloads. Response shapes
follow the source revisions in [the audit](PROVIDER-SOURCE-REVIEW.md). They are
**synthetic**, not responses recorded from running those upstream versions.

The mock implements a bounded subset of each engine: tokenizer, non-streaming
next-token logprobs, prompt-token logprobs for vLLM/native SGLang, and candidate
constraints for vLLM/native llama.cpp. Its tokenizer is BOS plus one token per
byte (`token_id = byte + 1`). This deliberately does not simulate BPE boundary
behavior, reasoning, multimodality, real KV caching, model quality, GPU kernels,
or the full request validation/feature set of any engine.

The fixed normalized vocabulary has `P(X)=0.4`, `P(A)=0.2`, `P(B)=0.1`,
`P(1)=0.15`, `P(0)=P(2)=0.05`, and 0.05 spread across the other 251 tokens.
It is independent of incoming prompts. Deterministic argmax sampling makes
mask behavior reproducible: an effective `{A,B}` mask changes the sampled
token from `X` to `A`. Raw candidate mass stays 0.3; processed mass becomes 1;
conditional `P(A)` is 2/3 in either case. No task-accuracy claims follow from
these numbers.

## Scenarios

| Scenario | Simulated behavior | Check |
|---|---|---|
| `normal` | Engine-specific response envelope; vLLM raw logprobs before mask; SGLang tuples in `meta_info`; llama.cpp raw/post-sampling forms | Parse exact labels and probability space |
| `selected-rejected` | HTTP 400 when selected-token fields are sent | Top-k fallback and readiness recover; disabling the field avoids rejection |
| `mask-rejected` | HTTP 400 when a candidate mask is sent | Rejection is surfaced; `constraint: none` avoids the field |
| `mask-ignored` | HTTP 200, allowlist ignored, `X` still sampled | Accepted JSON does not establish mask support |
| `missing-candidates` | Omit B/2 from output candidate entries | With other recovery paths disabled, strict mode returns 529 |
| `topk-missing` | Omit B/2 only from unmasked top-k entries | SGLang teacher forcing / llama.cpp grammar recovers the complete distribution |
| `processed-logprobs` | Return the processed distribution (vLLM scenario) | Configured `logprob_space: post-mask` gives candidate mass 1 |
| `post-sampling-ignored` | llama.cpp returns raw `top_logprobs` despite requesting `top_probs` | Adapter rejects the mismatched response |
| `no-tokenizer` | Tokenizer endpoints return 404 | SGLang top-k requests token text and uses probe-only labels |

These are fault-injection policies, not claims that every server version
behaves this way. The SGLang normal profile rejects native `allowed_token_ids`;
the ignored profile deliberately simulates an implementation that drops it.
The vLLM mock rejects more than 128 selected IDs independently of top-k.
The mock's top-k limit is 128, a test limit rather than an upstream default.

## Run the automated HTTP scenarios

```bash
go test -v ./internal/mockinfer
go test -race ./...
```

Tests start real loopback HTTP listeners for the mock servers and the gateway.
They cover direct wire behavior, production adapters, and
`POST /v1/systemone`, including conditional probabilities, scoring headers,
readiness, model settings, and strict failure. They do not require external
services. A passing test means hearim handled the specified synthetic scenario.

## Run standalone servers and the gateway

In separate terminals:

```bash
go run ./cmd/hearim-mock -engine vllm -addr 127.0.0.1:18080
go run ./cmd/hearim-mock -engine sglang -addr 127.0.0.1:18081
go run ./cmd/hearim-mock -engine llama.cpp -addr 127.0.0.1:18082
go run ./cmd/hearim serve -config hearim.mock.yaml -addr 127.0.0.1:18090
```

```bash
curl -s http://127.0.0.1:18080/mock/status
curl -s http://127.0.0.1:18090/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-vllm","state":"synthetic","questions":{"q":{"type":"choice","criteria":{"first":null,"second":null}}}}'
```

Use `mock-sglang` and `mock-llama` for the other routes. Numeric labels 1/2
yield `{first: 0.75, second: 0.25}`. Stop each process with Ctrl-C.
The sample config is localhost-only and has no API credentials.
If a port is occupied, choose another port and update the matching `base_url`;
do not stop an unrelated service.

### Local verification (2026-09-22)

Built and launched three standalone mock processes plus the hearim gateway.
All three routes returned HTTP 200 with conditional probabilities 0.75/0.25,
and `/readyz` returned `ready: true`. vLLM/SGLang used selected-token scoring;
llama.cpp used top-k. The vLLM smoke process used port 18083 because 18080 was
already occupied. `go test -race ./...`, `go vet ./...`, and `git diff --check`
also passed. These results cover synthetic HTTP scenarios, not real inference
server execution.

Select a scenario with `-scenario selected-rejected`. Model-specific scenarios
can share one server:

```bash
go run ./cmd/hearim-mock -engine vllm \
  -models 'standard=normal,restricted=selected-rejected'
```

Configure corresponding `models[].scoring` overrides in hearim, as described
in [PROVIDER-SCORING.md](PROVIDER-SCORING.md). Model dispatch requires a `model`
request field; native llama.cpp requests use the server-wide scenario.

## When real servers become available

Use small CPU-compatible models for protocol smoke tests if their engine,
architecture, and CPU instruction requirements fit the environment. vLLM
documents CPU serving and an OPT-125M example in its [CPU guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/).
SGLang's [CPU guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/cpu_server.mdx)
targets AMX-capable fourth-generation-or-newer Xeon processors. Model size
alone therefore does not establish CPU compatibility. The
[FunctionGemma model](https://huggingface.co/google/functiongemma-270m-it)
is 270M parameters, but its model page's generic launch examples are not
evidence of CPU operation on an arbitrary machine.

Real-server smoke tests can be added later without blocking this local suite.
Keep captured fixtures separately identified by server commit, model revision,
endpoint, and launch flags; do not relabel synthetic fixtures as measurements.
