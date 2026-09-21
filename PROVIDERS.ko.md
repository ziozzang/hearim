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

2차 추가 실측(같은 날, 소진): cloud의 네이티브 `/api/chat`·`/api/generate`
요청 구조체에는 `logprobs` **bool 필드가 존재**하고(숫자를 보내면
"cannot unmarshal number into Go struct field .logprobs of type bool"
오류 — 필드가 있다는 뜻), `logprobs: true`는 조용히 받아들여진다 —
그러나 응답에서 logprobs가 통째로 빠진다(스트리밍 포함). OpenAI 표면의
스트리밍 chat/completions + logprobs: 없음. 모델별 스윕(gpt-oss:20b,
deepseek-v4.1-flash, glm-5.3-flash, kimi-k3): 없음. 로컬 네이티브
`/api/chat`에 `logprobs: true`를 주면 **토큰별 logprob이 스트리밍된다**
(샘플된 토큰만 — 청크당 `{token, logprob, bytes}`). 로컬의 OpenAI chat
표면이 top-N까지 주므로 여전히 더 낫고, 이 관찰은 파이프가 로컬엔
존재하고 cloud에서 서버 쪽으로 걷어낸다는 증거로서만 중요하다.

추가 실측(같은 날): Ollama Cloud는 **`/v1/responses`(Responses API)를
노출**하고 `output_text` 콘텐츠 파트에 `logprobs` 필드가 있다 — 뼈대는
존재 — 그러나 서버가 `top_logprobs` 파라미터를 0으로 만들고(보낸 값과
무관하게 0으로 에코) 스트리밍 포함 모든 logprobs 배열이 비어 온다.
네이티브 `/api/chat`(그 API엔 logprob 파라미터 자체가 없음)도 정확히
답하지만 확률은 없다. 결론은 동일하되 이유가 정확해졌다: cloud는
서버 쪽 플래그 하나 거리이며, 오늘 클라이언트가 보낼 수 있는 것으로는
바꿀 수 없다.

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

### GLM 전체 매트릭스 (2026-09-21 실측)

| 모델 | 표준 chat think-off | logprobs (모든 표면) |
|---|---|---|
| glm-5.3 / 5.3-flash / 5.3-flashx | ❌ 서버 거부("always engages in thinking") | ❌ |
| glm-5.2 / 5.1 / 5 / 5-turbo / 4.7 / 4.6 / 4.5-air | ✅ 즉시 가시 콘텐츠 | ❌ |

GLM-5.3-Flash 심층(특히 질문한 모델): 긴 생성에서 reasoning 후 정답 `1`
도출 — 그러나 표준 엔드포인트(긴 생성·스트리밍 청크 모두)에서 logprobs
없음, 코딩 엔드포인트도 없음, 표준에는 `/completions` 자체가 없음(404),
z.ai 공식 API 레퍼런스에 **logprobs 파라미터가 요청·응답 스키마 모두에
존재하지 않음**. thinking은 서버에서 비활성화 불가라 생성 기반 회수도
wait-close 예산이 필요한데, 읽을 logprobs가 없다. 통계적 근사(N회 생성 후
라벨 빈도)만이 유일한 경로이나 exact 범위 밖이다.

매트릭스에서의 주의: thinking 비활성화가 품질을 해칠 수 있음 —
glm-5-turbo는 thinking off 상태에서 4를 두고 `odd`로 오답.

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

## Alibaba Qwen 토큰 플랜 (2026-09-21 실측) — 됨

`https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`
(OpenAI 호환, 코딩 구독 토큰 플랜). **세 번째로 검증된 exact 평가
프로바이더**(Ollama 로컬 chat, OpenRouter에 이어):

- logprobs: ✅ chat completions, `top_logprobs` 상한 **[0, 5]** — 실측
  최저 상한. N 스윕은 2라벨 질문에서 4로 수렴
- thinking: qwen3.6-flash / 3.7-plus / 3.8-flash에서
  `enable_thinking: false` 작동(즉시 라벨 콘텐츠). qwen3.7-max는 어느
  쪽도 logprobs 없음. thinking 모드에서도 top-5에 라벨이 뜨는 경우가
  있으나 의존하지 말 것
- `/v1/systemone` 종단간 검증: 짝홀 noul 0.9941 / 0.018, choice →
  billing, 2단계 rubric score 0.90

```yaml
providers:
  - id: qwen-plan
    engine: generic-openai
    base_url: ${QWEN_TOKEN_PLAN_BASE_URL}
    api_key: ${QWEN_TOKEN_PLAN_KEY}
    endpoint: chat_completions
    models:
      - name: qwen3.6-flash
        thinking: {disable_field: enable_thinking, disable_value: false}
```

## xAI OAuth 경유 Grok (2026-09-21 실측) — 토큰 갱신 막힘

`~/.grok/auth.json`에 `xai-oauth` 프로바이더(access+refresh 토큰,
discovery는 `https://auth.x.ai/oauth2/token`). 저장된 access 토큰이
만료(403 `bad-credentials`)됐고 갱신에 필요한 CLI의 client_id가
디스크에 없음(추측값은 `invalid_client` 거부). grok CLI 재로그인으로
갱신 가능. xAI 공개 API는 chat completions에서 logprobs를 지원하므로
유효 토큰만 있으면 기계적 부착:

```yaml
providers:
  - id: grok
    engine: generic-openai
    base_url: https://api.x.ai/v1
    api_key: ${GROK_OAUTH_ACCESS_TOKEN}
    endpoint: chat_completions
```

가능성: 높음(토큰 확보 후). 종단간 미검증.

## ChatGPT OAuth 경유 OpenAI Codex (2026-09-21 실측) — logprob 표면 없음

`~/.codex/auth.json`(auth_mode chatgpt)이
`https://chatgpt.com/backend-api/codex/responses`에서 작동 — 단
**Responses API의 잠긴 부분집합**: `store: false`와 `stream: true`
강제, `max_output_tokens`와 `top_logprobs`는 거부("Unsupported
parameter"), 스트리밍 delta에 확률 데이터 없음. 모델:
`gpt-5.6-sol`(config.toml 기준), 나머지는 400. 결론: 생성은 가능하나
**logprobs 없음 → exact 평가 불가**. 통계 근사만 가능.

## opencode go (타진 노트 — 미검증, 키 없음)

opencode Zen/go($10/월, 18개 모델)가 OpenAI 호환
`https://opencode.ai/zen/v1/chat/completions`(DeepSeek, MiniMax, GLM,
Kimi)와 Qwen/Claude용 Anthropic형 messages를 노출. hearim은
`engine: generic-openai` + 해당 base URL + `endpoint:
chat_completions` + `api_key`로 기계적으로 부착. 프로브가 정리해야
할 미지수: 모델별 logprobs 통과 여부(문서 없음), thinking 제어(GLM
계열은 z.ai와 동일한 상시 thinking 동작), 코딩 에이전트용 플랜의
레이트 리밋. 질문당 출력 1토큰이라 토큰 예산은 오래 감. `hearim
probe`를 먼저 돌리면 몇 번의 호출로 go/no-go가 나온다. 흥미로운
대칭: Zen도 Jev용 `/zen/v1/systemone`을 노출해서 hearim은 그 계약을
소비하는 쪽이면서 제공하는 쪽이 될 수 있다.

출처: [opencode.ai](https://opencode.ai), [opencode zen 문서](https://opencode.ai/docs/zen), [bitdoze 리뷰](https://www.bitdoze.com).

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
