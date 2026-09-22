# Provider source review (2026-09-22)

Implementation follow-up: the findings below record the pre-fix state.
The adapters now parse SGLang's native tuple envelope, reject its unsupported
allowlist, default vLLM selected-ID requests to at most 128 tokens, respect
raw versus configured processed logprobs, and explicitly request/parse
llama.cpp post-sampling probabilities for grammar scoring. Provider and model
overrides are documented in [PROVIDER-SCORING.md](PROVIDER-SCORING.md).
Fallback retries suppress selected-ID fields and incomplete top-k responses
continue to teacher forcing. Regression fixtures mirror the upstream source
shapes; they are not captured live-server measurements.

This is a **source-level audit, not a live conformance result**. The upstream
revisions inspected were SGLang `3f00fb7e`, vLLM `91d7324c`, and llama.cpp
`7ab4ee7b`. A model, server version, route, and tokenizer still need an
end-to-end probe before the route can be called exact. In particular, a server
accepting an unknown JSON key or returning a sampled label is **not** proof
that the requested logit mask or probability space took effect.

| Engine / surface | Upstream source behavior | Current hearim behavior | Assessment |
|---|---|---|---|
| SGLang native `/generate` | `token_ids_logprob` is a top-level request field. Selected and top logprobs are returned under `meta_info` as `(logprob, token_id, text-or-null)` tuples. Native `SamplingParams` does **not** declare `allowed_token_ids`. | Sends `token_ids_logprob` correctly, but parses `logprobs.output_token_ids_logprobs` as objects and sends `sampling_params.allowed_token_ids` in constrained mode. | **Blocker:** the parser does not match the current native response; the proposed mask cannot be assumed to work. `extra_body` is not a remedy for an unsupported native field. |
| vLLM `/v1/completions` | Protocol declares both `logprob_token_ids` and `allowed_token_ids`; the latter reaches sampler masking. Selected-token requests are capped at 128 IDs in `SamplingParams`. The sampler computes `raw_logprobs` before applying the allowlist in its default raw mode. | Sends both fields at the top level; advertises 256 selected IDs and marks a masked request `post-mask`. | **Partial:** field routing is real, but 129–255 choices exceed the inspected cap, and `post-mask` is not warranted for default raw logprobs. Verify returned entries and mode, not just HTTP success. |
| llama.cpp native `/completion` | `grammar` is supported. With default `post_sampling_probs=false`, `n_probs` returns pre-sampling top logprobs; `post_sampling_probs=true` changes the response to `top_probs`/`prob`. | Sends grammar but not `post_sampling_probs`; reads `top_logprobs` and labels the result `post-mask`. | **Partial:** top-k reading matches the raw response, but the grammar does not make those reported values post-mask. Current test mocks do not exercise the actual probability-space switch. |

## Source-level evidence and implications

### SGLang

- The [native request schema](https://github.com/sgl-project/sglang/blob/3f00fb7e2ef2fe54ef84be1fbb130c9f8276586f/python/sglang/srt/managers/io_struct.py) places `token_ids_logprob` on `GenerateReqInput`; the [sampling-parameter struct](https://github.com/sgl-project/sglang/blob/3f00fb7e2ef2fe54ef84be1fbb130c9f8276586f/python/sglang/srt/sampling/sampling_params.py) has no `allowed_token_ids`. `logit_bias` or a custom processor is a different mechanism and must be tested independently before being treated as an exact candidate mask.
- The [response assembly](https://github.com/sgl-project/sglang/blob/3f00fb7e2ef2fe54ef84be1fbb130c9f8276586f/python/sglang/srt/managers/tokenizer_manager.py) writes `output_token_ids_logprobs`, `output_top_logprobs`, and `input_token_logprobs` into `meta_info`; token logprobs are tuples. The local [parser](../internal/hearim/provider/sglang.go) expects a different envelope and object entries. Its mock [test](../internal/hearim/provider/adapter_test.go) repeats that invented shape, so a green unit test does not demonstrate compatibility.
- The same mismatch affects `ScoreContinuations`, which reads `logprobs.input_token_logprobs`. Until an integration fixture from a real SGLang server passes, mark native selected-ID and teacher-forced paths unverified.

### vLLM

- The [completion protocol](https://github.com/vllm-project/vllm/blob/91d7324cb19d301c72d849e457221ee8dd645024/vllm/entrypoints/openai/completion/protocol.py) accepts top-level `allowed_token_ids` and `logprob_token_ids`. An OpenAI client may send these via `extra_body`, but that client-side wrapper is flattened into the JSON body; it does not guarantee that every endpoint, proxy, or model honors the fields.
- [SamplingParams validation](https://github.com/vllm-project/vllm/blob/91d7324cb19d301c72d849e457221ee8dd645024/vllm/sampling_params.py) sets `MAX_LOGPROB_TOKEN_IDS = 128`. The local [adapter](../internal/hearim/provider/vllm.go) advertises 256. `--max-logprobs` controls a different top-k limit; setting it to 256 does not remove the selected-ID cap.
- The [sampler](https://github.com/vllm-project/vllm/blob/91d7324cb19d301c72d849e457221ee8dd645024/vllm/v1/sample/sampler.py) calculates raw logprobs before the allowlist. The sampled token is constrained, but the returned label logprobs are normally from the raw distribution. Normalizing those labels in hearim is still a valid *conditional* distribution, but it is not evidence of a post-mask server distribution. The configured `logprobs_mode` must be checked in a live probe.

### llama.cpp

- The [server contract](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/tools/server/README.md) documents `grammar`, `n_probs`, and `post_sampling_probs` separately. The [probability collector](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/tools/server/server-context.cpp) has separate pre- and post-sampling branches, and the [serializer](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/tools/server/server-task.cpp) emits `top_logprobs` in the former and `top_probs` in the latter.
- The local [adapter](../internal/hearim/provider/llamacpp.go) requests the former even when it sends a grammar. Its constrained-vocab result should therefore not be described as post-mask. A grammar can constrain the sampled token without changing the returned pre-sampling logprobs.
- This path also needs a real-server fixture for grammar acceptance, candidate coverage, and token-ID matching. The existing mock only checks that a `grammar` string was sent.

## Minimum conformance gate

For each server build and model: record the exact request JSON and response;
verify all candidate IDs are present, the probabilities' location and shape,
whether the mask changes **reported** probabilities or only the sampled token,
and whether the result is raw or post-mask. Include a deliberately impossible
label and a 129-candidate vLLM case. Do not enable a constrained fallback
merely because the request was accepted or a unit-test mock returned entries.
