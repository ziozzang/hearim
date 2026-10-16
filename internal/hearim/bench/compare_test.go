package bench

import (
	"encoding/json"
	"math"
	"testing"

	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
)

func TestCompareAgreement(t *testing.T) {
	model := []any{
		&jev.ChoiceAnswer{Type: "choice", Choice: "a", Probabilities: map[string]float64{"a": 0.8, "b": 0.2}},
		&jev.NoulAnswer{Type: "noul", Noul: 0.9},
		&jev.ScoreAnswer{Type: "score", Score: 0.4, Probabilities: map[string]float64{"0": 0.6, "1": 0.4}},
		nil, // failed record
	}
	reference := []any{
		&jev.ChoiceAnswer{Type: "choice", Choice: "a", Probabilities: map[string]float64{"a": 0.6, "b": 0.4}},
		&jev.NoulAnswer{Type: "noul", Noul: 0.3},
		&jev.ScoreAnswer{Type: "score", Score: 0.9, Probabilities: map[string]float64{"0": 0.1, "1": 0.9}},
		&jev.NoulAnswer{Type: "noul", Noul: 0.5},
	}
	ag := Compare(model, reference)
	if ag.Compared != 3 || ag.Skipped != 1 {
		t.Fatalf("compared=%d skipped=%d", ag.Compared, ag.Skipped)
	}
	if want := 1.0 / 3; math.Abs(ag.Top1Agreement-want) > 1e-9 {
		t.Fatalf("top1 = %v, want %v", ag.Top1Agreement, want)
	}
	// TVDs: choice 0.2, noul 0.6, score 0.5.
	if want := (0.2 + 0.6 + 0.5) / 3; math.Abs(ag.MeanTVD-want) > 1e-9 {
		t.Fatalf("mean tvd = %v, want %v", ag.MeanTVD, want)
	}
	if math.Abs(ag.MaxTVD-0.6) > 1e-9 || math.Abs(ag.MeanAbsScoreDiff-0.5) > 1e-9 {
		t.Fatalf("max tvd = %v, score diff = %v", ag.MaxTVD, ag.MeanAbsScoreDiff)
	}
	if len(ag.Worst) != 3 || ag.Worst[0].Type != "noul" || ag.Worst[0].Index != 1 {
		t.Fatalf("worst = %+v", ag.Worst)
	}
	if ag.ByType["choice"].Top1Agreement != 1 || ag.ByType["noul"].Top1Agreement != 0 {
		t.Fatalf("by type = %+v", ag.ByType)
	}
}

func TestCompareSkipsTypeMismatch(t *testing.T) {
	ag := Compare(
		[]any{&jev.NoulAnswer{Type: "noul", Noul: 0.9}},
		[]any{&jev.ChoiceAnswer{Type: "choice", Probabilities: map[string]float64{"a": 1}}},
	)
	if ag.Compared != 0 || ag.Skipped != 1 {
		t.Fatalf("compared=%d skipped=%d", ag.Compared, ag.Skipped)
	}
}

func TestDecodeJevAnswerLegendShapes(t *testing.T) {
	for name, raw := range map[string]string{
		"object": `{"type":"score","score":1,"confidence":0.97,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.01,"1":0.98,"2":0.01}}`,
		"array":  `{"type":"score","score":1,"confidence":0.97,"legend":["Calm","Frustrated","Very angry"],"probabilities":{"0":0.01,"1":0.98,"2":0.01}}`,
	} {
		ans, err := provider.DecodeJevAnswer(json.RawMessage(raw))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		sa, ok := ans.(*jev.ScoreAnswer)
		if !ok || len(sa.Legend) != 3 || sa.Legend["2"] != "Very angry" || sa.Probabilities["1"] != 0.98 {
			t.Fatalf("%s: decoded %+v", name, ans)
		}
	}
	if _, err := provider.DecodeJevAnswer(json.RawMessage(`{"type":"score","legend":{"low":"a"}}`)); err == nil {
		t.Fatal("non-index legend keys must fail")
	}
	// Structured criteria: legend values are objects, kept as JSON text.
	ans, err := provider.DecodeJevAnswer(json.RawMessage(`{"type":"score","legend":{"0":{"what":"Calm","examples":["thanks"]}},"probabilities":{"0":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := ans.(*jev.ScoreAnswer).Legend["0"]; got != `{"what":"Calm","examples":["thanks"]}` {
		t.Fatalf("structured legend = %q", got)
	}
	if _, err := provider.DecodeJevAnswer(json.RawMessage(`{"type":"mystery"}`)); err == nil {
		t.Fatal("unknown answer type must fail")
	}
}
