package jev

import (
	"encoding/json"
	"net/http"
	"testing"
)

const choiceBody = `{
  "model": "jev-local-auto",
  "state": {"ticket": "결제 후 다운로드 링크를 받지 못했습니다.", "account_age_days": 820},
  "questions": {
    "routing": {
      "type": "choice",
      "instructions": "가장 적절한 처리 큐를 선택하라.",
      "criteria": {
        "billing": "결제, 환불 또는 청구 문제",
        "delivery": "상품 또는 디지털 콘텐츠 전달 문제",
        "account": "로그인 또는 계정 문제",
        "other": null
      }
    }
  }
}`

func TestValidateChoice(t *testing.T) {
	pr, err := Validate([]byte(choiceBody))
	if err != nil {
		t.Fatal(err)
	}
	if pr.Model != "jev-local-auto" {
		t.Errorf("model = %q", pr.Model)
	}
	if len(pr.Questions) != 1 || pr.Questions[0].Type != TypeChoice {
		t.Fatalf("questions = %+v", pr.Questions)
	}
	q := pr.Questions[0]
	// Insertion order must be preserved for stable label mapping.
	want := []string{"billing", "delivery", "account", "other"}
	for i, n := range want {
		if q.ChoiceNames[i] != n {
			t.Errorf("order[%d] = %q, want %q", i, q.ChoiceNames[i], n)
		}
	}
	if q.ChoiceDescriptions["other"] != nil {
		t.Error("null description should stay nil")
	}
	if q.ChoiceDescriptions["billing"] == nil || *q.ChoiceDescriptions["billing"] != "결제, 환불 또는 청구 문제" {
		t.Error("description lost")
	}
	if _, ok := pr.State.(map[string]any); !ok {
		t.Errorf("state = %T", pr.State)
	}
}

func TestValidateScoreAndNoul(t *testing.T) {
	body := `{
	  "model": "m",
	  "state": "답변: 비밀번호를 재설정한 뒤 다시 로그인해 보세요.",
	  "questions": {
	    "quality": {
	      "type": "score",
	      "instructions": "고객지원 답변의 유용성을 평가하라.",
	      "criteria": ["low", "mid", "high", "highest"]
	    },
	    "refund_requested": {
	      "type": "noul",
	      "instructions": "환불 의사가 명시되어 있는가?",
	      "criteria": {"true": "직접적인 취소 또는 환불 요청이 있다", "false": "질문일 뿐이다"}
	    },
	    "bare_noul": {"type": "noul"}
	  }
	}`
	pr, err := Validate([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Questions) != 3 {
		t.Fatalf("questions = %d", len(pr.Questions))
	}
	for _, q := range pr.Questions {
		switch q.ID {
		case "quality":
			if len(q.Levels) != 4 || q.Levels[0] != "low" {
				t.Errorf("levels = %v", q.Levels)
			}
		case "refund_requested":
			if q.TrueDesc == nil || *q.TrueDesc == "" {
				t.Error("noul true description lost")
			}
		case "bare_noul":
			if q.TrueDesc != nil || q.FalseDesc != nil {
				t.Error("absent noul criteria should be nil")
			}
		}
	}
}

func TestValidateRejectsBadRequests(t *testing.T) {
	cases := []struct{ name, body string }{
		{"missing model", `{"state":"s","questions":{"q":{"type":"noul"}}}`},
		{"missing state", `{"model":"m","questions":{"q":{"type":"noul"}}}`},
		{"no questions", `{"model":"m","state":"s","questions":{}}`},
		{"bad type", `{"model":"m","state":"s","questions":{"q":{"type":"essay"}}}`},
		{"choice too few", `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":null}}}}`},
		{"choice as array", `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":["a","b"]}}}`},
		{"score too many", `{"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["1","2","3","4","5","6","7","8","9","10","11"]}}}`},
		{"score non-array", `{"model":"m","state":"s","questions":{"q":{"type":"score","criteria":{"a":"b"}}}}`},
		{"state number", `{"model":"m","state":42,"questions":{"q":{"type":"noul"}}}`},
		{"instructions number", `{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":7}}}`},
		{"empty option name", `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"":"d","b":"e"}}}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Validate([]byte(tc.body))
			if err == nil {
				t.Fatalf("expected validation error")
			}
			ve, ok := err.(*RequestValidationError)
			if !ok || len(ve.Details) == 0 {
				t.Fatalf("error %v lacks details", err)
			}
			wire := FromValidationError(ve)
			if wire.HTTPStatus() != http.StatusUnprocessableEntity {
				t.Errorf("status = %d", wire.HTTPStatus())
			}
		})
	}
}

func TestErrorStatuses(t *testing.T) {
	cases := map[ErrorCode]int{
		CodeUnauthorized:                  http.StatusUnauthorized,
		CodeValidationFailed:              http.StatusUnprocessableEntity,
		CodeRateLimited:                   http.StatusTooManyRequests,
		CodeBackendOverloaded:             529,
		CodeBackendProbabilityUnavailable: 529,
		CodeNoExactEvaluationRoute:        529,
		CodeInternal:                      http.StatusInternalServerError,
	}
	for code, want := range cases {
		e := NewError(code, "x")
		if e.HTTPStatus() != want {
			t.Errorf("%s -> %d, want %d", code, e.HTTPStatus(), want)
		}
	}
}

func TestChoice255Accepted(t *testing.T) {
	crit := map[string]any{}
	for i := 0; i < 255; i++ {
		crit[string(rune('a'+i%26))+string(rune('a'+i/26))] = nil
	}
	raw := string(mustJSON(map[string]any{
		"model": "m",
		"state": "s",
		"questions": map[string]any{
			"q": map[string]any{"type": "choice", "criteria": crit},
		},
	}))
	pr, err := Validate([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Questions[0].ChoiceNames) != 255 {
		t.Errorf("options = %d, want 255", len(pr.Questions[0].ChoiceNames))
	}
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}
