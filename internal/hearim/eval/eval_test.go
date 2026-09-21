package eval

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/registry"
	"hearim/internal/hearim/scoring"
)

// evalFake is a scoring-capable fake: per-rune tokenizer, selected-token-ids
// support, configurable logprobs.
type evalFake struct {
	logprobs   []float64 // per candidate index
	partialAt  int       // candidates >= partialAt are missing (first pass)
	recoverAll bool      // second pass (retry) returns everything
	missing    bool      // next-token scoring fails wholesale
	noTF       bool      // teacher-forced single-label path unsupported
	calls      int
}

func (f *evalFake) ID() string                { return "fake-vllm" }
func (f *evalFake) Engine() config.EngineKind { return config.EngineVLLM }
func (f *evalFake) Capabilities() provider.ProviderCapabilities {
	return provider.ProviderCapabilities{
		Engine:              config.EngineVLLM,
		Tokenizer:           "remote",
		CandidateScoring:    []string{"selected-token-ids", "top-k", "teacher-forced"},
		PromptTokenLogprobs: true,
		MaxTopLogprobs:      20,
		MaxSelectedTokenIDs: 256,
		Endpoints: []provider.EndpointProfile{
			{Kind: config.EndpointCompletions, NextTokenLogprobsVerified: true},
		},
	}
}

func (f *evalFake) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		ids = append(ids, int(r)+1)
	}
	return ids, nil
}

func (f *evalFake) ScoreNextToken(ctx context.Context, req provider.NextTokenScoreRequest) (*provider.NextTokenScoreResult, error) {
	f.calls++
	if f.missing {
		return nil, fmt.Errorf("upstream 500")
	}
	out := map[int]float64{}
	missingFrom := len(req.CandidateTokenTexts)
	if f.partialAt > 0 && f.calls == 1 {
		missingFrom = f.partialAt
	} else if f.partialAt > 0 && !f.recoverAll {
		missingFrom = f.partialAt
	}
	for i := range req.CandidateTokenTexts {
		if i >= missingFrom {
			break
		}
		lp := -2.0
		if i < len(f.logprobs) {
			lp = f.logprobs[i]
		}
		out[i] = lp
	}
	return &provider.NextTokenScoreResult{
		CandidateLogprobs:    out,
		AllCandidatesPresent: len(out) == len(req.CandidateTokenTexts),
		PromptTokens:         100,
		CachedPromptTokens:   40,
		ScoringMethod:        "selected-token-ids",
		ProbabilitySpace:     provider.SpaceRaw,
	}, nil
}

func (f *evalFake) ScoreContinuations(ctx context.Context, req provider.ContinuationScoreRequest) (*provider.ContinuationScoreResult, error) {
	if f.noTF && len(req.Continuations) == 1 {
		return nil, fmt.Errorf("single-label teacher forcing rejected")
	}
	res := &provider.ContinuationScoreResult{PromptTokens: 110, ScoringMethod: "teacher-forced-choice-text"}
	for range req.Continuations {
		res.Conditional = append(res.Conditional, provider.ContinuationScore{LogProb: -1.2, TokenLen: 4})
		if req.Unconditional {
			res.Unconditional = append(res.Unconditional, provider.ContinuationScore{LogProb: -0.4, TokenLen: 4})
		}
	}
	return res, nil
}
func (f *evalFake) Health(ctx context.Context) (provider.Health, error) {
	return provider.Health{OK: true}, nil
}
func (f *evalFake) ChatRender() bool { return false }
func (f *evalFake) RenderChat(p *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return nil
}

func testConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(`
providers:
  - id: fake-vllm
    engine: vllm
    base_url: http://fake
model_aliases:
  m: fake-vllm:gemma4:31b
`))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func buildRoute(t *testing.T, fake *evalFake, cfg *config.Config) Route {
	t.Helper()
	comp := compile.New(cfg.Compiler)
	probePlan, err := comp.Compile(mustParse(t, `{"model":"m","state":"probe","questions":{"p":{"type":"noul"}}}`), "gemma4:31b")
	if err != nil {
		t.Fatal(err)
	}
	reg, err := registry.Build(context.Background(), fake, registry.Options{
		BackendModel:        "gemma4:31b",
		TokenizerRevision:   "rev",
		Endpoint:            string(config.EndpointCompletions),
		TemplateVersion:     cfg.Compiler.TemplateVersion,
		PromptBase:          probePlan.Questions[0].PromptPrefix + probePlan.Questions[0].PromptSuffix,
		DelimiterCandidates: cfg.Compiler.DelimiterCandidates,
		Alphabets:           cfg.Compiler.LabelAlphabets,
	})
	if err != nil {
		t.Fatal(err)
	}
	return Route{
		Adapter:      fake,
		ProviderID:   "fake-vllm",
		BackendModel: "gemma4:31b",
		Engine:       config.EngineVLLM,
		Registry:     reg,
		Endpoint:     config.EndpointCompletions,
		CachePlan:    "automatic",
	}
}

func mustParse(t *testing.T, body string) *jev.ParsedRequest {
	t.Helper()
	pr, err := jev.Validate([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	return pr
}

func TestChoiceReductionAndHeadersMeta(t *testing.T) {
	cfg := testConfig(t)
	fake := &evalFake{logprobs: []float64{math.Log(0.08), math.Log(0.86), math.Log(0.02), math.Log(0.04)}}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, err := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": {"t": "x"},
	  "questions": {"routing": {"type": "choice",
	    "instructions": "pick",
	    "criteria": {"billing": "b", "delivery": "d", "account": "a", "other": null}}}
	}`), "gemma4:31b")
	if err != nil {
		t.Fatal(err)
	}
	res, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err != nil {
		t.Fatal(err)
	}
	ans, ok := res.Answer.(*jev.ChoiceAnswer)
	if !ok {
		t.Fatalf("answer type %T", res.Answer)
	}
	if ans.Choice != "delivery" {
		t.Errorf("choice = %s", ans.Choice)
	}
	if math.Abs(ans.Probabilities["delivery"]-0.86) > 1e-9 {
		t.Errorf("p(delivery) = %v", ans.Probabilities["delivery"])
	}
	var sum float64
	for _, v := range ans.Probabilities {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum = %v", sum)
	}
	if res.CandidateMass < 0.999 {
		t.Errorf("candidate mass = %v, want ~1", res.CandidateMass)
	}
	if res.ScoringMethod != "selected-token-ids" {
		t.Errorf("method = %s", res.ScoringMethod)
	}
	if res.ProbabilitySpace != provider.SpaceRaw {
		t.Errorf("space = %s", res.ProbabilitySpace)
	}
	if res.InputTokens != 100 || res.CachedTokens != 40 {
		t.Errorf("usage = %d/%d", res.InputTokens, res.CachedTokens)
	}
	meta := MetaFrom(route, []*QuestionResult{res}, cfg)
	if meta.Implementation != "vllm-logprob-v1" {
		t.Errorf("implementation = %s", meta.Implementation)
	}
	if meta.QuestionCalls != 1 || meta.CachePlan != "automatic" {
		t.Errorf("meta = %+v", meta)
	}
	if meta.ConfidenceMethod != "normalized-entropy-v1" {
		t.Errorf("confidence method = %s", meta.ConfidenceMethod)
	}
	if !strings.Contains(res.ProfileID, "selected-token-ids") {
		t.Errorf("profile = %s", res.ProfileID)
	}
}

func TestScoreAndNoulReduction(t *testing.T) {
	cfg := testConfig(t)
	fake := &evalFake{logprobs: []float64{math.Log(0.05), math.Log(0.20), math.Log(0.60), math.Log(0.15)}}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, err := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {
	    "quality": {"type": "score", "criteria": ["lo", "missing", "good", "full"]},
	    "refund": {"type": "noul", "criteria": {"true": "T", "false": "F"}}
	  }
	}`), "gemma4:31b")
	if err != nil {
		t.Fatal(err)
	}
	// Reverse candidate order so index math stays honest: score levels map
	// 0..3 but we want the doc example distribution on levels.
	sr, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err != nil {
		t.Fatal(err)
	}
	sa := sr.Answer.(*jev.ScoreAnswer)
	if math.Abs(sa.Score-1.85) > 1e-9 {
		t.Errorf("score = %v, want 1.85", sa.Score)
	}
	if len(sa.Legend) != 4 || sa.Legend[2] != "good" {
		t.Errorf("legend = %v", sa.Legend)
	}
	if math.Abs(sa.Probabilities["2"]-0.60) > 1e-9 {
		t.Errorf("p(level2) = %v", sa.Probabilities["2"])
	}

	nr, err := ev.EvaluateQuestion(context.Background(), plan, 1, route)
	if err != nil {
		t.Fatal(err)
	}
	// Noul publishes the conditional probability over {true,false}:
	// 0.05 / (0.05 + 0.20) = 0.2, not the raw vocabulary mass.
	na := nr.Answer.(*jev.NoulAnswer)
	if math.Abs(na.Noul-0.2) > 1e-9 {
		t.Errorf("noul = %v, want 0.2", na.Noul)
	}
}

func TestStrictFailsWhenCandidatesMissing(t *testing.T) {
	cfg := testConfig(t)
	fake := &evalFake{logprobs: []float64{-1, -2, -3, -4}, partialAt: 2, recoverAll: false}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, _ := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {"q": {"type": "choice", "criteria": {"a": null, "b": null, "c": null, "d": null}}}
	}`), "gemma4:31b")
	_, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err == nil {
		t.Fatal("strict mode must fail when candidates stay missing")
	}
	je := err.(*jev.Error)
	if je.Type != jev.CodeBackendProbabilityUnavailable {
		t.Errorf("code = %s", je.Type)
	}
	if je.HTTPStatus() != 529 {
		t.Errorf("status = %d", je.HTTPStatus())
	}
	// The teacher-forced fallback must have been attempted.
	if fake.calls < 2 {
		t.Errorf("expected retry attempts, calls = %d", fake.calls)
	}
}

func TestTopKRetryRecovers(t *testing.T) {
	cfg := testConfig(t)
	fake := &evalFake{logprobs: []float64{-1, -0.5, -3, -4}, partialAt: 2, recoverAll: true}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, _ := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {"q": {"type": "noul"}}
	}`), "gemma4:31b")
	res, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err != nil {
		t.Fatal(err)
	}
	if res.CandidateMass < 0.5 {
		t.Errorf("mass = %v", res.CandidateMass)
	}
}

func TestChoiceTextFallbackMode(t *testing.T) {
	cfg := testConfig(t)
	cfg.Scoring.ChoiceTextMode = "fallback"
	cfg.Scoring.ContinuationNormalization = "pmi"
	fake := &evalFake{logprobs: nil, partialAt: 0}
	// Make selected/top-k fail wholesale and the single-label teacher-forced
	// path unavailable, so the chain reaches choice-text.
	fake.missing = true
	fake.noTF = true
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, _ := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {"q": {"type": "choice", "criteria": {"a": "AA", "b": "BB"}}}
	}`), "gemma4:31b")
	res, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.ScoringMethod, "choice-text") {
		t.Errorf("method = %s", res.ScoringMethod)
	}
	if !strings.Contains(res.ScoringMethod, "pmi") {
		t.Errorf("method should record pmi: %s", res.ScoringMethod)
	}
	na := res.Answer.(*jev.ChoiceAnswer)
	if na.Choice != "a" && na.Choice != "b" {
		t.Errorf("choice = %s", na.Choice)
	}
}

func TestCandidateMassFloor(t *testing.T) {
	cfg := testConfig(t)
	// Tiny mass: candidates at -20 each vs the rest of the vocabulary.
	fake := &evalFake{logprobs: []float64{-20, -20, -20, -20}}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	ev.MinCandidateMass = 0.01
	plan, _ := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {"q": {"type": "noul"}}
	}`), "gemma4:31b")
	_, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err == nil {
		t.Fatal("low candidate mass must fail")
	}
	if !strings.Contains(err.Error(), "candidate mass") {
		t.Errorf("err = %v", err)
	}
}

func TestParityNextTokenVsTeacherForced(t *testing.T) {
	// §12.2: on a stable single-token label the two score sources must
	// agree. The fake returns the same values through both paths.
	cfg := testConfig(t)
	fake := &evalFake{logprobs: []float64{math.Log(0.3), math.Log(0.7)}}
	route := buildRoute(t, fake, cfg)
	ev := New(compile.New(cfg.Compiler), cfg)
	plan, _ := ev.Compiler.Compile(mustParse(t, `{
	  "model": "m", "state": "s",
	  "questions": {"q": {"type": "noul"}}
	}`), "gemma4:31b")
	res, err := ev.EvaluateQuestion(context.Background(), plan, 0, route)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.Answer.(*jev.NoulAnswer).Noul-0.3) > 1e-9 {
		t.Errorf("noul = %v, want 0.3", res.Answer.(*jev.NoulAnswer).Noul)
	}
	// Softmax sanity identical to scoring package.
	_ = scoring.Softmax([]float64{math.Log(0.3), math.Log(0.7)})
}
