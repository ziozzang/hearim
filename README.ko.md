# hearim (헤아림)

**Jev 호환 멀티백엔드 System One 게이트웨이 (Go 구현).**

hearim은 일반 autoregressive 백엔드(Ollama Cloud, llama.cpp, vLLM, SGLang, 일반
OpenAI 호환 서버) 위에 TypeSafe AI **Jev**의 `POST /v1/systemone` 계약을
노출한다. `Choice` / `Score` / `Noul` 질문을 **한 토큰 객관식 문제**로
컴파일하고, 후보 라벨의 logprob로부터 확률 분포를 복원한다.

생성된 텍스트를 답변 근거로 쓰지 않는다. 질문 하나는 정확히 한 번의
next-token scoring 호출이며, 라벨 fast path가 불가능하면 라벨 또는 선택지
전체를 teacher-forcing 한다.

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
모델/digest/tokenizer/template/endpoint/delimiter로 키를 지어
`data/registries/`에 영속화하며, 그 중 하나라도 바뀌면 폐기한다. 경계
재분절은 고정 backtrack이 아니라 exact-LCP healing(후보별 LCP의 최솟값)로
처리한다.

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

---

## CLI

| 명령 | 용도 |
|---|---|
| `hearim serve` | 게이트웨이 실행 |
| `hearim probe` | Phase 0 capability suite: tokenizer endpoint, logprob 형태, 단일 토큰 라벨, teacher-forced parity, hidden reasoning·캐시 증거. JSON 보고서 출력, 모든 대상이 go/no-go 게이트에 실패하면 0이 아닌 종료 코드 |
| `hearim bench` | 라벨된 JSONL 코퍼스 실행: 정확도, macro F1, NLL, Brier, ECE, candidate mass, p50/p95 지연, 라벨 순서 순열 분산(`-permutations N`) |

bench 코퍼스 형식:

```jsonl
{"state": "...", "question": {"type": "choice", "criteria": {"a": "...", "b": "..."}}, "expected": "a"}
{"state": "...", "question": {"type": "noul", "criteria": {"true": "...", "false": "..."}}, "expected": true}
{"state": "...", "question": {"type": "score", "criteria": ["low", "high"]}, "expected": 1}
```

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
