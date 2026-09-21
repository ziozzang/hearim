# AI configuration prompt guide

> 한국어 버전은 [CONFIG-PROMPT.ko.md](CONFIG-PROMPT.ko.md)에 있습니다.

Paste one of the prompts below into any capable LLM (it works best with a
model that has the target provider's documentation in context, but the
probe-JSON prompt works with any). The output is a ready-to-paste hearim
provider block.

## Prompt 1 — from a model card

```text
You are configuring hearim, a Jev-compatible evaluation gateway that scores
single answer-label tokens via logprobs. Read the model card below and emit
a YAML block for hearim's providers[] and model_aliases[].

Rules you must follow:
- base the thinking control on what the card documents; map it to ONE of:
    thinking: {disable_field: <field>, disable_value: <value>}   # request field
    thinking: {user_suffix: "/no_think"}                          # command token at user-request tail
    prompt: {system_prefix: "<|think|>"}                          # prompt-level toggle
  and add thinking.close_tag when the model emits tag structure
  (</think>, <channel|>) that survives disabling.
- set type: vision only when the card documents image input.
- set type: thinking when reasoning cannot be fully disabled.
- endpoint: chat_completions unless the card documents a completions or
  native surface with logprobs.
- never invent parameter names not present in the card; if the card is
  silent on reasoning control, say so in a comment and leave thinking unset.
- output ONLY the YAML block plus one-line comments.

Model card:
<paste the model card here>

Provider base URL and engine:
<e.g. engine: generic-openai, base_url: https://...>
```

## Prompt 2 — from a probe report (recommended)

Run `hearim probe -discover-thinking -provider <id> -model <m>` first, then:

```text
You are configuring hearim (a Jev-compatible evaluation gateway). Below is a
JSON probe report from the live backend and the current provider YAML.
Adjust the YAML to fix every failed or warning check:
- thinking_discovery / thinking_control failures -> set models[].thinking
  exactly as the discovery string suggests (disable_field/disable_value,
  user_suffix, or close_tag when thinking_tags_emitted is true)
- cache_evidence / completions_logprobs failures on chat -> if
  completions_fallback_viable is true, pin endpoint: completions
- top_n_cap smaller than the labels you need -> note the candidate limit
- never remove strict_candidate_probabilities; never put secrets in the
  YAML (use ${ENV_VAR}).
Output only the corrected YAML.

Probe report:
<paste `hearim probe` JSON>

Current config:
<paste the provider block>
```

## What the LLM needs to know (cheat sheet)

| hearim knob | meaning |
|---|---|
| `endpoint` | force chat_completions / completions / native_generate |
| `paths:` | remap upstream URL paths per surface |
| `query_params:` / `headers:` | URL flags (detailed=true) / extra headers (X-Api-Key) |
| `extra_params:` | extra JSON body fields (protected fields ignored) |
| `models[].type` | chat / thinking (no exact chat route) / vision (required for images) |
| `models[].thinking.disable_field/disable_value` | card-documented request-field switch |
| `models[].thinking.user_suffix` | command token at the tail of the user request |
| `models[].thinking.close_tag` | preload `</think>`-style tags so the next token is the label |
| `models[].thinking.wait_close` | scan a bounded generation for the close tag (approximate) |
| `models[].prompt.system_prefix` / `template` | prompt-level toggle / full raw template override |

Measured provider quirks live in [PROVIDERS.md](../PROVIDERS.md).
