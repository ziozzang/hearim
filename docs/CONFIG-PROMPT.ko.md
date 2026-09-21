# AI 설정 프롬프트 가이드

> English version: [CONFIG-PROMPT.md](CONFIG-PROMPT.md).

아래 프롬프트 하나를 아무 유능한 LLM에 붙여넣으면(대상 프로바이더 문서가
컨텍스트에 있는 모델이면 가장 좋고, probe JSON 프롬프트는 아무 모델이나
동작) 바로 붙일 수 있는 hearim provider 블록이 나옵니다.

## 프롬프트 1 — 모델 카드로부터

```text
당신은 단일 답변-라벨 토큰을 logprob로 채점하는 Jev 호환 평가 게이트웨이
hearim을 설정하고 있다. 아래 모델 카드를 읽고 hearim의 providers[]와
model_aliases[]에 넣을 YAML 블록을 출력하라.

반드시 지킬 규칙:
- thinking 제어는 카드가 문서한 것에 근거하고, 다음 중 하나로 매핑하라:
    thinking: {disable_field: <필드>, disable_value: <값>}   # 요청 필드형
    thinking: {user_suffix: "/no_think"}                      # 유저 요청 끝 명령 토큰형
    prompt: {system_prefix: "<|think|>"}                      # 프롬프트 수준 토글
  그리고 비활성화해도 태그 구조(</think>, <channel|>)가 남으면
  thinking.close_tag를 추가하라.
- 카드가 이미지 입력을 문서할 때만 type: vision.
- reasoning을 완전히 끌 수 없으면 type: thinking.
- logprob이 있는 completions/native 서피스를 카드가 문서하지 않는 한
  endpoint: chat_completions.
- 카드에 없는 파라미터 이름을 지어내지 마라. 카드가 reasoning 제어에
  침묵하면 주석으로 밝히고 thinking을 비워 두라.
- YAML 블록과 한 줄 주석만 출력하라.

모델 카드:
<여기에 모델 카드 붙여넣기>

프로바이더 base URL과 engine:
<예: engine: generic-openai, base_url: https://...>
```

## 프롬프트 2 — probe 보고서로부터 (권장)

먼저 `hearim probe -discover-thinking -provider <id> -model <m>`을 실행한 뒤:

```text
당신은 Jev 호환 평가 게이트웨이 hearim을 설정하고 있다. 아래는 실제
백엔드에서 나온 JSON probe 보고서와 현재 provider YAML이다. 실패했거나
경고인 모든 항목을 고치도록 YAML을 수정하라:
- thinking_discovery / thinking_control 실패 -> discovery 문자열이 제안하는
  그대로 models[].thinking 설정 (disable_field/disable_value, user_suffix,
  thinking_tags_emitted가 true면 close_tag)
- chat에서 cache_evidence / completions_logprobs 실패 ->
  completions_fallback_viable이 true면 endpoint: completions로 고정
- top_n_cap이 필요한 라벨 수보다 작으면 -> 후보 제한 주석으로 명시
- strict_candidate_probabilities를 제거하지 마라. 시크릿을 YAML에 넣지
  마라(${ENV_VAR} 사용).
수정된 YAML만 출력하라.

Probe 보고서:
<hearim probe JSON 붙여넣기>

현재 설정:
<provider 블록 붙여넣기>
```

## LLM이 알아야 할 치트시트

| hearim 노브 | 의미 |
|---|---|
| `endpoint` | chat_completions / completions / native_generate 강제 지정 |
| `paths:` | 서피스별 업스트림 URL 경로 재매핑 |
| `query_params:` / `headers:` | URL 플래그(detailed=true) / 추가 헤더(X-Api-Key) |
| `extra_params:` | 추가 JSON 바디 필드(보호 필드는 무시) |
| `models[].type` | chat / thinking(exact chat 경로 없음) / vision(이미지에 필수) |
| `models[].thinking.disable_field/disable_value` | 카드 문서 기반 요청 필드 스위치 |
| `models[].thinking.user_suffix` | 유저 요청 끝에 붙는 명령 토큰 |
| `models[].thinking.close_tag` | `</think>`류 태그 프리로드 — 다음 토큰이 라벨 |
| `models[].thinking.wait_close` | 유한 생성에서 닫는 태그 스캔(근사) |
| `models[].prompt.system_prefix` / `template` | 프롬프트 수준 토글 / 전체 raw 템플릿 재정의 |

실측 프로바이더 특이사항은 [PROVIDERS.ko.md](../PROVIDERS.ko.md) 참고.
