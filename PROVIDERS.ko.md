# 프로바이더·모델 capability 노트 (실측)

> 기본 문서는 영어([PROVIDERS.md](PROVIDERS.md))이고, 이 파일은 전체 한국어 번역이다.
>
> 여기의 모든 내용은 문서 복사가 아니라 **`hearim probe`와 직접 API 호출로
> 측정**했다. capability는 변한다 — route를 신뢰하기 전에 `hearim probe`를
> 다시 실행하라. readiness 게이트가 미검증 route를 자동으로 운전에서
> 제외한다. 최종 조사: **2026-09-21**.

## 요약 매트릭스

| 엔진 / 호스트 | logprob 표면 | tokenizer | 비전 | 비고 |
|---|---|---|---|---|
| Ollama local 0.24.0 (chat) | ✅ `choices[].logprobs.content[].top_logprobs`(상한 20, choice 수준) | ❌ (probe-only) | ✅ | Ollama의 exact 경로 |
| Ollama local 0.24.0 (completions) | ❌ `logprobs` int는 받지만 미반환 | — | — | |
| Ollama local 0.24.0 (native generate) | ❌ | 아래 참조 | — | `context`에 토큰 id 반환 |
| Ollama Cloud (테스트 전 모델) | ❌ chat·completions, 텍스트·비전 모두 | — | ✅ 추론만 | 답은 정확하나 logprob 없음 → 오늘 기준 exact 평가 불가 |
| vLLM | `logprob_token_ids`로 예상(버전별 검증) | ✅ `/tokenize` | ✅ | 후보 ID 직접 경로 |
| SGLang | `token_ids_logprob`으로 예상 | ✅ 게이트웨이 `/v1/tokenize` | ✅ | prefix 친화 라우팅 권장 |
| llama.cpp | `/completion`의 `n_probs`로 예상 | ✅ `/tokenize` | 모델별 | `cache_prompt`, `tokens_cached` 관측 |

## Ollama Cloud 모델 (2026-09-21 실측)

vision+tools+thinking+cloud 태그 모델: `gemma4`(e2b–31b),
`qwen3.5`(0.8b–122b), `glm-5.3-flash`, `deepseek-v4.1-flash`,
`minimax-m3`, `kimi-k2.6`, `kimi-k3`, `kimi-k2.7-code`.

| 모델 | 이미지 입력 | 1-token 응답 | logprobs |
|---|---|---|---|
| gemma4:31b | ✅ (168바이트 PNG가 296 프롬프트 토큰) | `1` — 빨강/파랑 정확 판별 | ❌ |
| deepseek-v4.1-flash | ✅ (234 토큰) | `` (thinking이 토큰 소비) | ❌ |
| glm-5.3-flash | ✅ (190 토큰) | `` | ❌ |
| kimi-k3 | ✅ (190 토큰) | `` | ❌ |
| minimax-m3 | ✅ (217 토큰) | `` | ❌ |

해석: **cloud에서 비전 추론은 되지만 logprob 표면이 없다.** 따라서
hearim은 probe가 logprob을 목격할 때까지 cloud route를 exact 운전에서
제외한다. thinking VLM들은 `max_tokens: 1`에서 가시 콘텐츠가 비므로
생성 기반 fallback도 `thinking.wait_close`와 실제 토큰 예산이 필요하고,
그래도 읽을 logprob이 없다.

## z.ai GLM (2026-09-21 실측)

엔드포인트: `https://api.z.ai/api/coding/paas/v4`(코딩 구독 표면,
OpenAI 호환). 모델: glm-4.5 … glm-5.3-flash(x).

- 답은 **정확하다**(짝홀 → `1`). 단 전 모델이 상시 reasoner라 출력이
  `reasoning_content`에 들어가고 reasoning이 끝나야 `content`가 차오른다
  — `max_tokens: 1`은 reasoning에 소모된다(§3.4 hidden-reasoning 위험,
  `usage.completion_tokens_details.reasoning_tokens`으로 관측 가능).
- 이 표면에서는 `thinking: {"type": "disabled"}`가 무시된다.
- **`logprobs`가 조용히 무시된다** — 요청은 받지만 어느 모델, 어떤
  생성에서도 반환되지 않는다.

추가 실측(같은 날, 양쪽 엔드포인트):

- **표준 엔드포인트**(`https://api.z.ai/api/paas/v4`)는 코딩 키를 받고,
  코딩 엔드포인트와 달리 **`thinking: {"type": "disabled"}`가 작동한다**:
  glm-4.5-air가 즉시 가시 콘텐츠로 답한다. glm-5.3-flash는 아예 거부
  ("always engages in thinking and cannot be disabled"). hearim route에는
  표준 엔드포인트를 권장.
- `logprobs`는 양쪽 엔드포인트·전 모델에서 여전히 부재 — 클라이언트 쪽
  파라미터 형태로 바뀌지 않는다(Ollama 자체 호환 매트릭스로도 확인:
  logprobs와 `echo`는 chat·completions 모두 미지원 표기. local 0.24.0의
  chat 표면만 문서를 앞서 구현되어 있음).

결론: Ollama Cloud와 같은 부류 — 추론은 되지만 logprob 표면이 없어
오늘 기준 exact 평가 불가. z.ai나 Ollama Cloud가 logprob을 노출하는
순간 `engine: generic-openai` + `endpoint: chat_completions` +
`thinking: {disable_field: thinking, disable_value: {type: disabled}}`로
코드 변경 없이 부착된다. 객체 값 제어는 회귀 테스트로 보장한다.

## OpenRouter (2026-09-21 실측) — 됨

`https://openrouter.ai/api/v1`, OpenAI 호환. 모델별 통과 여부:

| 모델 | logprobs | 비고 |
|---|---|---|
| `openai/gpt-4o-mini` | ✅ top-10, 선명(`1` −0.0 / `0` −11.25) | `/v1/systemone` 전 과정 검증 |
| `meta-llama/llama-3.3-70b-instruct` | ❌ (프로바이더 측) | 모델별 확인 필요 |

hearim을 통한 종단간 검증 완료: 커스텀 헤더(`HTTP-Referer`,
`X-Title` — `headers:` 설정), `chat_completions` 강제 지정,
probe-only 레지스트리(OpenRouter에 tokenizer 엔드포인트 없음).
짝홀 noul → 0.9999999998; 라우팅 choice → billing/delivery 0.5/0.5
(정직한 모호성). 권장 설정:

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

## 모델 카드 조사: thinking 제어는 모델마다 다르다

- **gemma4** — **시스템 프롬프트 시작**의 `<|think|>` 토큰으로 토글.
  edge 외 모델은 꺼도 *태그 구조를 계속 출력*(`<|channel>thought …
  <channel|>`, 빈 블록). 설정:
  ```yaml
  models:
    - name: gemma4:31b
      prompt: {system_prefix: "<|think|>"}   # thinking을 켤 때만
      thinking: {close_tag: "<channel|>"}
  ```
- **gpt-oss:20b** — `reasoning_effort` low/medium/high뿐, 완전 비활성화
  불가 → exact chat 경로 없음(§3.4).
- **deepseek-v4.1-flash** — 카드에 제어 방법이 없음.
- **qwen3 계열** — `<think>…</think>` 블록; Ollama 네이티브는
  `think: false`, 또는 `close_tag: "</think>"` 프리로드.

`hearim probe`가 설정된 제어를 실모델로 검증해
`thinking_control_verified` / `thinking_tags_emitted`를 보고한다.

## 비전 실측 (로컬 gemma3:4b, Ollama 0.24.0)

- `/v1/systemone` 통한 전 과정 검증: 빨강/파랑 판별 noul = 0.755 /
  0.012 / 0.012 (빨강-빨강?, 파랑-빨강?, 빨강-파랑?) — 조건부 라벨
  logprob 기반.
- **state 텍스트 redaction이 중요**: base64 data URL을 state 텍스트 블록에
  두면 라벨 logprob이 방향을 잃었다(세 질문 모두 0.001–0.003). hearim은
  렌더링된 텍스트에서 이미지 값을 `<image:N>` placeholder로 치환하고
  이미지는 콘텐츠 파트로만 전달한다(테스트 이미지 기준 519 → 375 토큰).
- 소형 VLM은 라벨보다 자유 텍스트(`red`)로 답하기 쉽다 — 문제없다.
  hearim은 샘플된 토큰이 아니라 선언된 라벨의 조건부 분포를 채점한다.
  비전 chat route는 낮은 `candidate_mass`를 감안해
  `scoring.min_candidate_mass`를 조정하라.
- **텍스트 모델은 이미지를 조용히 삼킨다**(qwen2.5:0.5b로 실측) —
  그럴듯한 쓰레기를 반환한다. 그래서 hearim은 이미지 포함 state에
  명시적 `models[].type: vision`을 요구한다(아니면 422).
- moondream은 Ollama 0.24.0에서 네이티브 `/api/chat`조차 빈 출력 —
  모델/서버 비호환, hearim 문제가 아니다.

## 프로바이더별 기본 설정

```yaml
providers:
  - id: ollama-local
    engine: ollama
    base_url: http://localhost:11434
    request_timeout: 180s        # 비전 prefill + 모델 로드가 느릴 수 있음
    models:
      - name: gemma3:4b
        type: vision
```

엔드포인트 강제 지정·경로 재매핑·쿼리 파라미터·헤더·비용 헤더 통과는
README의 "Arbitrary providers" 절을 참고하라.
