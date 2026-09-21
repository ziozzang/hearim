# hearim (헤아림)

**Jev 호환 멀티백엔드 System One 게이트웨이 (Go 구현).**

hearim은 일반 autoregressive 백엔드(Ollama Cloud, llama.cpp, vLLM, SGLang, 일반
OpenAI 호환 서버) 위에 TypeSafe AI **Jev**의 `POST /v1/systemone` 계약을
노출한다. `Choice` / `Score` / `Noul` 질문을 **한 토큰 객관식 문제**로
컴파일하고, 후보 라벨의 logprob로부터 확률 분포를 복원한다.

생성된 텍스트를 답변 근거로 쓰지 않는다. 질문 하나는 정확히 한 번의
next-token scoring 호출이며, 라벨 fast path가 불가능하면 라벨 또는 선택지
전체를 teacher-forcing 한다.

### 이름의 뜻

**헤아림**은 동사 **헤아리다**의 명사형으로, 두 가지 얽힌 뜻을 갖는다:
*(하나하나) 세다*, 그리고 *생각하여 이해하다·깨닫다*. 이 게이트웨이가 하는
일이 정확히 그 겹침이다 — 이해하는 행위("이 state는 고객이 환불을 원한다는
뜻인가?")를 세는 행위로 바꾼다. 한 디코드 위치에서 라벨 logprob를 측정해
정규화한다. hearim은 세고, 호출자는 이해한다 — 정직한 숫자와 함께.

> 기본 문서는 영어([README.md](README.md))이고, 이 파일은 전체 한국어 번역이다.

---

## 동작 방식

```
클라이언트
  -> SystemOne 스키마 검증          (질문 단위 상세가 포함된 422)
  -> canonicalizer                   (RFC 8785 JCS: 안정적인 프롬프트 바이트)
  -> 모델 라우터                     (alias -> provider/engine/model)
  -> EvaluationPlan 컴파일러         (state-major / rubric-major 프롬프트)
  -> prefix-aware 스케줄러           (provider별 lane, prefix 친화성)
  -> ProviderAdapter                 (Ollama / llama.cpp / vLLM / SGLang)
  -> logprob reducer                 (조건부 softmax + entropy confidence)
  -> Jev 호환 응답
```

### 확률 복원

네 선택지 질문에서 모델에게는 다음이 보인다:

```text
A = 승인
B = 추가 검토
C = 거절
D = 판단 불가

<answer-label>
```

hearim은 `max_tokens: 1`과 `logprobs`를 요청해 라벨 logprob `l_A … l_D`를
읽고, 조건부 분포를 공개한다:

```text
p(A) = exp(l_A) / (exp(l_A) + exp(l_B) + exp(l_C) + exp(l_D))
```

`confidence = 1 − H(p)/ln(K)` (정규화 entropy). 둘 다 문서화된 구현 선택이며,
Jev와 회선(wire) 호환일 뿐 TypeSafe 모델과 수치 동일성을 보장하지 않는다.
진단 정보는 응답 헤더(`x-jev-implementation`, `x-jev-backend-model`,
`x-jev-question-calls`, `x-jev-cache-plan`, `x-jev-confidence-method`,
`x-jev-scoring-method`, `x-jev-probability-space`,
`x-jev-calibration-profile`)로만 전달되고 본문에는 들어가지 않는다.

### scoring 전략 체인

질문별로 어댑터는 다음 순서로 시도한다 (TODO.md §3.11 / §7.3):

1. **selected-token-ids** — 후보 token ID의 logprob 직접 요청
   (vLLM `logprob_token_ids`, SGLang `token_ids_logprob`)
2. **top-k** — top-N logprob를 텍스트/바이트 매칭으로 회수, 누락 시 더 큰
   N으로 한 번 재시도
3. **teacher-forced-label** — 라벨 토큰의 input-token logprob
   (안정 라벨에서는 next-token logprob와 일치해야 한다)
4. **constrained-vocab** — grammar / `allowed_token_ids` 마스크. 결과는
   `x-jev-probability-space: post-mask`로 표시된다
5. **choice-text continuation** — 선택지 문자열 전체 채점.
   `sum` / `token-mean` / `pmi` 정규화와 모드별 calibration 온도 τ 적용

strict 모드(기본)에서 복원 불가능한 후보가 있으면 요청 전체가
`529 backend_probability_unavailable`로 실패한다. 누락 후보를 확률 0으로
가정하지 않는다.

### Token Label Registry

부팅 시 모델·엔드포인트별로 어떤 라벨이 정확한 답변 마커 문맥에서 안정적인
단일 토큰인지 검증한다. `prefix + delimiter + label`을 tokenize하고
`tokenize(prefix)`가 정확한 prefix이며 토큰이 정확히 하나 더 붙는지 확인하며,
알파벳 전체가 조건을 만족할 때까지 delimiter 후보를 바꿔 시도하고, 실제
1-token completion probe로 확인한다. 레지스트리는
모델/digest/tokenizer/template/endpoint/close-tag/delimiter로 키를 지어
`data/registries/`에 영속화하며, 그 중 하나라도 바뀌면 폐기한다. 경계
재분절은 고정 backtrack이 아니라 exact-LCP healing(후보별 LCP의 최솟값)로
처리한다.

#### API에 tokenizer가 없는 경우

모든 게이트웨이가 /tokenize를 노출하지는 않는다. hearim은 정의된 단계로
성능을 낮춘다(§6.1 우선순위 3):

1. 레지스트리는 **inference probe**로 전환한다: 답변 위치에서 실제
   1-token scoring을 호출하고, 라벨이 top-logprob 항목으로 정확히 나타나면
   구성상 단일 토큰으로 출력 가능하다는 뜻이므로 `token_id: -1`,
   `boundary_policy: probe-only`로 등록한다. delimiter 후보와 알파벳을
   순서대로 시도해 라벨이 충분히 확보될 때까지 찾는다.
2. scoring은 token-ID 요청 대신 **텍스트/바이트 매칭 top-k**를 쓴다 —
   ID 기반 경로(`logprob_token_ids`, `token_ids_logprob`,
   teacher-forcing)는 ID가 미해결이면 자동으로 건너뛰며, 절대
   placeholder 값으로 전송하지 않는다.
3. readiness: completion probe가 여전히 라벨 logprob을 식별해야 하며,
   보고서에 `tokenizer_unverified` 경고가 남아 약한 검증 기반임을
   운영자에게 보여준다.

주의할 결과: 후보 회수가 직접 ID 요청 대신 백엔드의 top-logprob 상한
(Ollama는 20)에 묶이고, calibration은 probe-only route를 별도 프로파일로
취급해야 한다.

### Fan-out과 캐싱

Jev 요청 하나는 질문당 정확히 한 번의 upstream scoring 호출로 fan-out되며,
공유 prefix 단위로 묶여 유한한 lane에서 실행된다(Ollama Pro: 3). 스케줄러는
실행 중인 prefix, 최근 warm한 prefix, FIFO 순으로 우선하여, 공유 state
prefill이 시간적으로 인접하게 유지되도록 한다. prefix key는
`SHA256(model_digest ‖ tokenizer_revision ‖ template_version ‖ layout ‖
prefix_bytes)`이며, alias·타임스탬프·요청 ID·trace ID는 절대 프롬프트에
들어가지 않는다.

---

## 빠른 시작

```bash
# 빌드
go build -o hearim ./cmd/hearim

# 설정 (API 키를 파일에 넣지 말 것; 환경변수 확장 지원)
cp hearim.example.yaml hearim.yaml
export OLLAMA_API_KEY=...        # 시크릿 저장소(예: ~/env)에서
export HEARIM_API_KEY=...        # 클라이언트가 제시할 키

# Phase 0 capability 게이트를 먼저 확인
./hearim probe -config hearim.yaml -provider ollama-cloud -model gemma4:31b

# 실행
./hearim serve -config hearim.yaml -addr :8080
```

### 평가 요청

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

응답 (Jev 계약 형태; 확률은 선언된 후보 집합에 대한 조건부 분포):

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

세 primitive를 모두 지원한다:

| type | 의미 | 응답 |
|---|---|---|
| `choice` | 2~255개 명명 선택지 중 하나 | `choice`, 전체 `probabilities`, `confidence` |
| `score` | 2~10단계 순서형 척도 | 확률 가중 `score`, `legend`, `probabilities`, `confidence` |
| `noul` | 참/거짓 명제 | `noul` = P(true) |

### OpenAI 호환 표면

`POST /v1/completions`와 `POST /v1/chat/completions`는 기본 route의
provider로 투명 프록시한다. hearim은 일반 대화 트래픽에서 Jev 평가를
추측하지 않는다. 고급 용도로 `/v1/completions`에 비표준 `jev_candidates`
확장이 있다:

```json
{
  "model": "jev-gemma4",
  "prompt": "...\nANSWER:",
  "max_tokens": 1,
  "logprobs": 10,
  "jev_candidates": {"A": "billing", "B": "delivery", "C": "account", "D": "other"}
}
```

공개 제품 API는 `/v1/systemone`이며, 이 확장은 OpenAI 표준이 아니다.

### 운영 엔드포인트

- `GET /healthz` — 생존 확인
- `GET /readyz` — ready한 레지스트리를 가진 기본 route가 없으면 실패
- `GET /v1/routes` — route, 엔드포인트, 레지스트리, 캐시 계획
- `GET /metrics` — Prometheus 텍스트 형식(인증 필요)

---

## CLI

| 명령 | 용도 |
|---|---|
| `hearim serve` | 게이트웨이 실행 |
| `hearim probe` | Phase 0 capability suite: tokenizer endpoint, logprob 형태, 단일 토큰 라벨, teacher-forced parity, hidden reasoning·캐시 증거. JSON 보고서 출력, 모든 대상이 go/no-go 게이트에 실패하면 0이 아닌 종료 코드 |
| `hearim bench` | 라벨된 JSONL 코퍼스 실행: 정확도, macro F1, NLL, Brier, ECE, candidate mass, p50/p95 지연, 라벨 순서 순열 분산(`-permutations N`) |
| `hearim update` | GitHub 릴리즈에서 SHA256SUMS 검증 후 자가 업데이트 |

bench 코퍼스 형식:

```jsonl
{"state": "...", "question": {"type": "choice", "criteria": {"a": "...", "b": "..."}}, "expected": "a"}
{"state": "...", "question": {"type": "noul", "criteria": {"true": "...", "false": "..."}}, "expected": true}
{"state": "...", "question": {"type": "score", "criteria": ["low", "high"]}, "expected": 1}
```

## 모델명 해석

요청의 `model` 필드는 다음 순서로 해석된다:

1. 설정된 alias(`model_aliases`) 또는 폴백 체인(`model_alias_chains`,
   바인딩된 첫 route 사용)
2. 명시적 `provider:model`(prefix가 설정된 provider 이름일 때)
3. **bare 백엔드 모델명** — `"gemma4:31b"`는 그 모델을 서빙하는
   provider로 해석된다. 여러 provider가 같은 이름을 서빙하면 명시적
   오류로 알린다

## 임의 프로바이더: 서피스·경로·파라미터·헤더

provider별 노브를 조합하면 어떤 OpenAI형 게이트웨이든 붙일 수 있다:

```yaml
providers:
  - id: my-gateway
    engine: generic-openai
    base_url: https://gw.internal
    endpoint: chat_completions        # scoring 서피스 강제 지정(모델 수준
                                      # `models[].endpoint`이 이것보다 우선)
    paths:                            # 커스텀 URL 레이아웃
      chat_completions: /api/v2/chat
      completions: /api/v2/completions
      tokenize: /api/v2/tokenize
    query_params:                     # detailed=true 같은 플래그가 모든
      detailed: "true"                # 업스트림 URL에 붙는다
    headers:                          # 추가 요청 헤더(X-Api-Key,
      X-Api-Key: ${GW_TOKEN}          # HTTP-Referer, X-Title, ...)
    extra_params: {some_flag: 1}      # 추가 JSON 바디 필드
```

업스트림 **응답** 헤더 중 화이트리스트된 것 — 비용/청구(`x-cost*`,
`x-usage*`, `x-billed*`), 레이트 리밋(`x-ratelimit*`, `retry-after`,
`x-remaining*`, `x-quota*`), `x-request-id` — 는 성공한 시도에서 캡처해
클라이언트에게 `x-jev-upstream-<이름>`로 통과시킨다. 그 외 헤더는
프로바이더 내부에 머문다.

## 멀티 백엔드 설정

- `providers[].base_urls` — 한 엔진의 엔드포인트 풀: 요청은 레플리카 간
  round-robin되고 재시도는 *다음* 호스트로 장애조치되어, 죽은 레플리카를
  제자리에서 재시도하지 않고 우회한다.
- `model_alias_chains` — 순서 폴백: `{"resilient":
  ["ollama-cloud:gemma4:31b", "vllm-local:google/gemma-4-31B-it"]}`.
- `models` 항목은 업스트림 메타데이터를 담는다:

```yaml
models:
  - gemma4:31b                          # 일반 항목
  - name: qwen3:32b
    type: thinking                      # reasoning을 완전히 끌 수 없음:
                                        # chat exact 경로 거부(§3.4)
  - name: llava:13b
    type: vision                        # 이미지 입력 허용
    extra_params: {num_ctx: 16384}      # 업스트림에 병합되는 추가 JSON
```

- `extra_params`(provider 또는 모델 수준, 키 충돌 시 모델이 우선)는 모든
  업스트림 요청에 추가 JSON 필드를 병합한다 — `seed`, `num_ctx` 같은
  엔진 고유 노브나 게이트웨이 전용 파라미터. 정확성에 중요한 필드
  (`logprobs`, `max_tokens`, 샘플러 값 등)는 보호되어 덮어쓸 수 없다.

## thinking 모델 (`<think>` 처리)

모델 카드에는 reasoning을 끄는 방법이 문서되어 있다. hearim은 이를 모델별
세 가지 기법으로 매핑한다:

```yaml
models:
  - name: qwen3:32b
    type: thinking
    thinking:
      disable_field: think          # 모델 카드의 제어를 그대로 적용
      disable_value: false          # (예: reasoning_effort: "none")
      close_tag: "</think>"         # 프리로드 기법(정확 경로)
  - name: r1-style:14b
    type: thinking
    thinking:
      close_tag: "</think>"
      wait_close: true              # 스캔 기법(근사)
      max_think_tokens: 512
```

1. **disable** — 카드가 알려주는 스위치(`think: false`,
   `reasoning_effort: "none"` 등)를 모든 업스트림 요청에 적용하며 엔진
   기본값을 대체한다.
2. **close-tag 프리로드** — 항상 `<think>` 블록을 여는 모델에게는 답변
   마커 바로 뒤에 닫는 태그를 미리 넣는다
   (`...</answer-label>\n</think>\n`). 그러면 다음 토큰이 곧 답 라벨이어서
   정확한 단일 디코드 위치 의미론이 유지된다. token label registry도 이
   태그 이후 문맥에서 경계를 검증한다.
3. **wait-close 스캔** — 프리로드가 불가능할 때: reasoning 블록을
   `max_think_tokens`(기본 256)까지 생성하면서 `</think>` 이후 첫 위치의
   logprob 분포를 읽는다. `x-jev-scoring-method: wait-close-tag`로 표시된다.
   이 분포는 샘플링된 reasoning 텍스트에 조건화되므로 근사이며,
   calibration profile도 별도 모드로 취급한다. 블록이 닫히지 않으면
   reasoning 토큰을 채점하는 대신 실패한다.

`thinking` 블록이 없으면 `type: thinking` 모델은 엔진이 reasoning을 완전히
끌 수 없는 한 exact chat 경로가 없다(TODO.md §3.4).

일부 카드는 thinking을 **프롬프트 수준**에서 제어해서 요청 필드로는 표현할
수 없다 — 따라서 프롬프트 모양도 모델별로 강제 재지정 가능하다:

```yaml
models:
  - name: gemma4:31b
    prompt:
      system_prefix: "<|think|>"   # 시스템 시작의 제어 토큰(카드 문서화)
      # template: |                # 전체 raw 레이아웃 재정의(Go text/template):
      #   SYS {{.System}}          #   {{.System}} {{.State}} {{.Question}}
      #   {{.State}}               #   {{.Criteria}} {{.Marker}}
      #   {{.Question}}{{.Criteria}}{{.Marker}}
```

`system_prefix`는 시스템 블록(raw 레이아웃)과 chat 시스템 메시지 앞에
붙는다. 커스텀 `template`은 raw 프롬프트 전체를 대체한다 — 다섯 변수는
렌더링된 system/state/question/criteria 블록과 답변 마커이며, 커스텀
템플릿은 state-major prefix 공유를 모양 제어와 맞바꾸고 raw completion
경로에만 적용된다. 재정의는 템플릿 식별자에 해시로 반영되어 레지스트리와
prefix key가 재정의별로 분리되고, 라벨 경계도 정확히 그 문맥에서
검증된다.

`close_tag`는 자유 형식이라 모델 계열별 태그 구조를 그대로 쓸 수 있다.
2026-09-21에 조사한 Ollama Cloud 카드만 해도 세 가지로 다르다:

- **gemma4** — `<|think|>` 시스템 프롬프트 토큰으로 토글되지만, edge 외
  모델은 꺼도 *태그 구조를 계속 출력한다*(`<|channel>thought …
  <channel|>`, 빈 블록). `close_tag: "<channel|>"`로 빈 블록을 건너뛴다.
- **gpt-oss** — effort `low`/`medium`/`high`만 있고 완전 비활성화는 불가 →
  exact chat 경로가 없다.
- **deepseek-v4.1-flash** — 카드에 제어 방법이 아예 없다.

카드마다 다르고 시간이 지나면 달라지므로, `hearim probe`는 **설정된
제어를 실제 모델에 대해 검증**한다: disable 제어를 적용한 유한 생성에서
reasoning 마커를 스캔하고, 보고서에 `thinking_control_verified`를 기록하며,
gemma4처럼 태그가 남는 경우 `thinking_tags_emitted`로 표시해 close-tag
처리가 필요함을 알린다.

## 비전(VLM) 이미지 입력

state 객체는 최상위 `image`(단일 URL) 또는 `images`(배열)로 이미지를
선언할 수 있다. `https?://`와 `data:image/...` URL만 허용한다 — bare
경로와 다른 스킴은 검증 단계에서 거부된다(state는 데이터이지 로컬 파일을
읽는 권한이 아니다). 최대 8장, data URL 20MB. 이미지는 비전 가능 chat
경로에 **명시적 `models[].type: vision`**을 요구한다. 그 외 전부 —
타입 미지정 포함, 텍스트 모델이 이미지를 조용히 삼키고 그럴듯한 쓰레기를
반환하기 때문(실측) — 422다.

이미지는 OpenAI 멀티모달 `image_url` 콘텐츠 파트로 전달되고, 렌더링된
state 텍스트는 base64 페이로드 대신 `<image:N>` placeholder를 담는다
(식별자·해시·prefix key는 미변경 형태 유지). gemma3:4b로 실측: 텍스트에
페이로드를 두면 라벨 logprob이 방향을 잃었고, redaction이 판별을 되살렸다
(빨강/파랑 probe에서 0.755 / 0.012 / 0.012)며 프롬프트 토큰이 약 27%
줄었다. 소형 VLM은 라벨보다 자유 텍스트로 답하기 쉽다 — 무관하다.
hearim은 선언된 라벨의 조건부 분포를 채점한다. 비전 chat route는 낮은
`candidate_mass`를 감안하라.

## 관측

- `GET /metrics` — Prometheus 텍스트 형식(API와 동일한 인증 뒤에 있음):
  요청 카운터·지연 히스토그램, scoring 방법·확률 공간별 질문 호출,
  input/output/cached-reported로 분리된 토큰 카운터, route별
  `hearim_prefill_cache_ratio`(*보고된* 캐시 토큰만으로 계산 — §8.5),
  예산 사용률, route readiness 게이지, 기본 Go 런타임 게이지.
- 응답 헤더 `x-jev-cached-input-tokens`는 서버가 보고한 이 요청의 캐시
  토큰 수를, `x-jev-cache-plan`은 엔진의 캐시 전략을 전달한다.

## 자동 업데이트

`hearim update`는 실행 중인 바이너리를 GitHub 릴리즈 빌드로 교체하며,
릴리즈의 `SHA256SUMS` 에셋에 기록된 SHA-256을 먼저 검증한다(설계는
[hftools](https://github.com/ziozzang/hftools)를 따름). `update -check`는
확인만, `-version v0.2.0`은 특정 릴리즈 고정, `-force`는 재설치.
대화형 실행에서는 새 릴리즈가 있으면 1일 1회 업데이트 알림을 출력한다 —
`HEARIM_NO_UPDATE_CHECK=1`로 끌 수 있다.

## 모델 라우팅과 비용

라우터는 모델명이 아니라 `(provider, engine, endpoint, model)` tuple을
선택한다. `policy:auto-v1` alias는 §9 비용 규칙으로 해석된다: 손익분기
캐시 비율 이상이면 캐시 입력 단가가 낮은 후보가 입력비에서 이기고
(gemma4:31b vs deepseek-v4.1-flash의 손익분기는 h ≈ 0.1754), 그 이하이면
품질 게이트를 통과한 최저가 후보가 이긴다. 품질 게이트는 명시적
(`quality_gate_pass`)이며 가격만으로 결정하지 않는다. 예산 가드는 포함
한도의 70%/90%에 throttle을, 100%에 hard-stop을 건다.

## 보안 노트

- state는 명령이 아니라 데이터다: 길이 구분 블록과 escape된 닫는 태그.
  prompt injection을 완전히 제거할 수는 없고 봉쇄만 가능하다.
- API 키, 테넌트 ID, trace ID는 절대 프롬프트에 들어가지 않는다.
- 원본 state 로깅은 기본 비활성(해시 + 토큰 수만 기록).
- upstream 시크릿은 환경변수에 두고 설정 파일은 `${VAR}` / `${VAR:-기본값}`로
  참조한다.
- 모델 출력은 고위험 결정(권한 부여, 결제 승인 등)의 단독 근거로 쓰지 않는다.

## 개발

```bash
go build ./...
go test ./...
go vet ./...
```

패키지 구성 (모두 `internal/hearim/` 아래):

| 패키지 | 역할 |
|---|---|
| `canon` | RFC 8785 JCS canonicalization + I-JSON 검증 |
| `config` | YAML 스키마, 기본값, 검증 |
| `jev` | 요청/응답 계약, 의미 검증, 오류 표면 |
| `scoring` | softmax, candidate mass, entropy confidence, LCP healing, continuation 모드 |
| `compile` | EvaluationPlan, state-major/rubric-major 템플릿, 라벨 매핑, prefix key |
| `provider` | Adapter 인터페이스, route 해석, Ollama/llama.cpp/vLLM/SGLang/일반 어댑터 |
| `registry` | token label registry: 부팅 probe, 영속화, readiness |
| `schedule` | prefix 친화 lane 스케줄러 |
| `eval` | 전략 체인 오케스트레이션과 reduction |
| `router` | alias/정책 해석과 비용 계산 |
| `usage` | 사용량 집계, 원가 ledger, 예산 가드 |
| `httpapi` | 서버 조립, 핸들러, 프록시, 헬스 |
| `probe` | Phase 0 capability suite |
| `bench` | §12.3 벤치마크 실행기 |
| `selfupdate` | SHA256SUMS 검증 기반 GitHub 릴리즈 자가 업데이트와 백그라운드 알림 |
| `metrics` | 최소 Prometheus 텍스트 노출 레지스트리(stdlib만) |

프로바이더·모델별 실측 상세는 [PROVIDERS.ko.md](PROVIDERS.ko.md)를 참고하라.

## 실측 capability (2026-09-21, 실제 백엔드)

Phase 0 프로브를 실서비스에 대해 실행한 결과가 기본값에 반영되어 있다:

- **Ollama (cloud + local 0.24.0)**: `/v1/completions`는 `logprobs`(int)를
  받지만 **logprob를 전혀 반환하지 않고**, `/v1/chat/completions`가
  `choices[].logprobs.content[].top_logprobs`를 완전히 반환한다(상한 20).
  따라서 hearim은 Ollama scoring을 chat 경로로 라우팅하며(§3.4)
  tokenization은 probe-only다. Ollama의 GPT-OSS(low/medium/high만 지원)는
  exact Ollama route가 없다.
- **로컬 e2e 검증 완료**: Ollama 0.24.0의 `qwen2.5:0.5b`로 레지스트리
  구축(숫자 라벨 검증), `/v1/systemone` choice/score/noul 응답·헤더·사용량·
  fan-out 확인. 소형 모델은 라벨 준수도가 낮아 candidate mass floor에
  걸리므로 운영 calibration에는 제대로 된 평가 모델을 쓸 것.
- cloud `ollama.com/v1` 3개 모델(gemma4:31b, deepseek-v4.1-flash,
  gpt-oss:20b) 스윕: 프로브 시점에 completions·chat 모두 logprob 없음 —
  cloud route를 신뢰하기 전에 `hearim probe`를 다시 실행하라. readiness
  게이트가 미검증 route를 운영에서 제외한다.

## 호환성 주의

hearim은 TypeSafe AI Jev의 **API 계약**을 구현한다. Jev의 모델, 확률
보정, 지연시간을 재현하지 않는다. 조건부 softmax, 정규화 entropy
confidence, route별 calibration profile은 hearim 자체의 문서화된
방법이며, 임계값은 자체 라벨 데이터(`hearim bench`)로 검증해야 한다.

## 라이선스

미정
