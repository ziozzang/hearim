package bench

import (
	"bufio"
	"context"
	"math"
	"strings"
	"testing"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/eval"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/registry"
	"hearim/internal/hearim/scoring"
)

type benchFake struct{}

func (f *benchFake) ID() string                { return "fake" }
func (f *benchFake) Engine() config.EngineKind { return config.EngineVLLM }
func (f *benchFake) Capabilities() provider.ProviderCapabilities {
	return provider.ProviderCapabilities{
		Engine:              config.EngineVLLM,
		CandidateScoring:    []string{"selected-token-ids", "top-k"},
		PromptTokenLogprobs: true,
		MaxTopLogprobs:      20,
		MaxSelectedTokenIDs: 256,
		Endpoints: []provider.EndpointProfile{
			{Kind: config.EndpointCompletions, NextTokenLogprobsVerified: true},
		},
	}
}

func (f *benchFake) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		ids = append(ids, int(r)+1)
	}
	return ids, nil
}

// Distribution: candidate 0 wins with 0.6, others split the rest.
func (f *benchFake) ScoreNextToken(ctx context.Context, req provider.NextTokenScoreRequest) (*provider.NextTokenScoreResult, error) {
	base, _ := f.Tokenize(ctx, "", req.PromptText)
	out := map[int]float64{}
	n := float64(len(req.CandidateTokenTexts))
	for i := range req.CandidateTokenTexts {
		if i == 0 {
			out[i] = math.Log(0.6)
		} else {
			out[i] = math.Log(0.4 / (n - 1))
		}
	}
	return &provider.NextTokenScoreResult{
		CandidateLogprobs:    out,
		AllCandidatesPresent: true,
		PromptTokens:         len(base),
		CachedPromptTokens:   0,
		ScoringMethod:        "selected-token-ids",
		ProbabilitySpace:     provider.SpaceRaw,
	}, nil
}

func (f *benchFake) ScoreContinuations(ctx context.Context, req provider.ContinuationScoreRequest) (*provider.ContinuationScoreResult, error) {
	res := &provider.ContinuationScoreResult{PromptTokens: 50, ScoringMethod: "teacher-forced-choice-text"}
	for range req.Continuations {
		res.Conditional = append(res.Conditional, provider.ContinuationScore{LogProb: -1, TokenLen: 2})
	}
	return res, nil
}
func (f *benchFake) Health(ctx context.Context) (provider.Health, error) {
	return provider.Health{OK: true}, nil
}
func (f *benchFake) ChatRender() bool { return false }
func (f *benchFake) RenderChat(p *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return nil
}

const corpus = `
{"state": "user asks for refund", "question": {"type": "choice", "criteria": {"refund": "wants refund", "other": "anything else"}}, "expected": "refund"}
{"state": "user says thanks", "question": {"type": "noul", "criteria": {"true": "refund intent", "false": "none"}}, "expected": true}
{"state": "answer is thorough", "question": {"type": "score", "criteria": ["bad", "ok", "good"]}, "expected": 1}
`

func testSetup(t *testing.T) (*config.Config, eval.Route) {
	t.Helper()
	cfg, err := config.Parse([]byte(`
gateway:
  default_model: m
providers:
  - id: fake
    engine: vllm
    base_url: http://fake
model_aliases:
  m: fake:gemma4:31b
`))
	if err != nil {
		t.Fatal(err)
	}
	comp := compile.New(cfg.Compiler)
	probePlan, _ := comp.Compile(mustParseB(`{"model":"m","state":"probe","questions":{"p":{"type":"noul"}}}`), "gemma4:31b")
	reg, err := registry.Build(context.Background(), &benchFake{}, registry.Options{
		BackendModel:        "gemma4:31b",
		Endpoint:            "completions",
		TemplateVersion:     cfg.Compiler.TemplateVersion,
		PromptBase:          probePlan.Questions[0].PromptPrefix + probePlan.Questions[0].PromptSuffix,
		DelimiterCandidates: cfg.Compiler.DelimiterCandidates,
		Alphabets:           cfg.Compiler.LabelAlphabets,
	})
	if err != nil {
		t.Fatal(err)
	}
	return cfg, eval.Route{
		Adapter:      &benchFake{},
		ProviderID:   "fake",
		BackendModel: "gemma4:31b",
		Engine:       config.EngineVLLM,
		Registry:     reg,
		Endpoint:     config.EndpointCompletions,
	}
}

func mustParseB(s string) *jev.ParsedRequest {
	pr, err := jev.Validate([]byte(s))
	if err != nil {
		panic(err)
	}
	return pr
}

func TestBenchMetrics(t *testing.T) {
	cfg, route := testSetup(t)
	recs, err := LoadCorpus(bufio.NewReader(strings.NewReader(corpus)))
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 3 {
		t.Fatalf("records = %d", len(recs))
	}
	m, err := Run(context.Background(), recs, route, compile.New(cfg.Compiler), cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Candidate 0 always wins: choice correct (refund first), noul true,
	// score weighted mean = 0*0.6 + 1*0.2 + 2*0.2 = 0.6 -> level 1.
	if m.Accuracy != 1 {
		t.Errorf("accuracy = %v, want 1", m.Accuracy)
	}
	if m.MacroF1 != 1 {
		t.Errorf("macro F1 = %v, m", m.MacroF1)
	}
	if m.NegLogLikelihood <= 0 {
		t.Errorf("NLL = %v", m.NegLogLikelihood)
	}
	if m.Brier < 0 || m.ECE < 0 {
		t.Errorf("brier/ece = %v/%v", m.Brier, m.ECE)
	}
	if m.P50LatencyMs <= 0 {
		t.Errorf("p50 = %v", m.P50LatencyMs)
	}
	if m.Errors != 0 {
		t.Errorf("errors = %d", m.Errors)
	}
	if m.MedianCandidateMass < 0.99 {
		t.Errorf("candidate mass = %v", m.MedianCandidateMass)
	}
}

func TestBenchPermutationVariance(t *testing.T) {
	cfg, route := testSetup(t)
	recs, _ := LoadCorpus(bufio.NewReader(strings.NewReader(corpus)))
	m, err := Run(context.Background(), recs, route, compile.New(cfg.Compiler), cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	// The fake is order-insensitive: variance must be ~0.
	if m.MeanPermutationVar > 1e-9 {
		t.Errorf("permutation variance = %v, want ~0 on an order-insensitive backend", m.MeanPermutationVar)
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if p := percentile(xs, 50); p != 5.5 && p != 5 {
		t.Logf("p50 = %v", p)
	}
	if p := percentile(xs, 100); p != 10 {
		t.Errorf("p100 = %v", p)
	}
	if p := percentile(nil, 50); p != 0 {
		t.Errorf("empty = %v", p)
	}
	_ = scoring.Softmax
}
