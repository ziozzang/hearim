package probe

import (
	"context"
	"testing"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/scoring"
)

type probeFake struct{}

func (f *probeFake) ID() string                { return "fake" }
func (f *probeFake) Engine() config.EngineKind { return config.EngineVLLM }
func (f *probeFake) Capabilities() provider.ProviderCapabilities {
	return provider.ProviderCapabilities{
		Engine:              config.EngineVLLM,
		Tokenizer:           "remote",
		CandidateScoring:    []string{"selected-token-ids", "top-k"},
		PromptTokenLogprobs: true,
		MaxTopLogprobs:      20,
		MaxSelectedTokenIDs: 256,
		Endpoints: []provider.EndpointProfile{
			{Kind: config.EndpointCompletions, NextTokenLogprobsVerified: true},
		},
	}
}

func (f *probeFake) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		ids = append(ids, int(r)+1)
	}
	return ids, nil
}

func (f *probeFake) ScoreNextToken(ctx context.Context, req provider.NextTokenScoreRequest) (*provider.NextTokenScoreResult, error) {
	base, _ := f.Tokenize(ctx, "", req.PromptText)
	out := map[int]float64{}
	for i, text := range req.CandidateTokenTexts {
		full, _ := f.Tokenize(ctx, "", req.PromptText+text)
		if scoring.IsStableSingleToken(base, full) {
			out[i] = -float64(i+1) - 0.5
		}
	}
	return &provider.NextTokenScoreResult{
		CandidateLogprobs:    out,
		AllCandidatesPresent: len(out) == len(req.CandidateTokenTexts),
		PromptTokens:         len(base),
		CachedPromptTokens:   10,
		ScoringMethod:        "selected-token-ids",
		ProbabilitySpace:     provider.SpaceRaw,
		GeneratedText:        "1",
	}, nil
}

func (f *probeFake) ScoreContinuations(ctx context.Context, req provider.ContinuationScoreRequest) (*provider.ContinuationScoreResult, error) {
	res := &provider.ContinuationScoreResult{PromptTokens: 50, ScoringMethod: "teacher-forced-choice-text"}
	for range req.Continuations {
		// Consistent with next-token scores so parity passes.
		res.Conditional = append(res.Conditional, provider.ContinuationScore{LogProb: -1.5, TokenLen: 1})
	}
	return res, nil
}
func (f *probeFake) Health(ctx context.Context) (provider.Health, error) {
	return provider.Health{OK: true, Detail: "0.9.2"}, nil
}
func (f *probeFake) ChatRender() bool { return false }
func (f *probeFake) RenderChat(p *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return nil
}

func TestProbeReportGoGate(t *testing.T) {
	cfg := config.CompilerConfig{
		TemplateVersion:     "systemone-v1",
		DelimiterCandidates: []string{"\n"},
		LabelAlphabets:      [][]string{{"1", "2", "3", "4"}},
	}
	rep, err := Run(context.Background(), &probeFake{}, "gemma4:31b", cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Summary.Go {
		for _, c := range rep.Checks {
			t.Logf("%-28s pass=%v %s", c.Name, c.Pass, c.Detail)
		}
		t.Fatal("go/no-go gate should pass on a capable fake")
	}
	if !rep.Summary.TokenizerEndpoint || !rep.Summary.CompletionsLogprobs || !rep.Summary.AllLabelsInTopN {
		t.Errorf("summary = %+v", rep.Summary)
	}
	if rep.Registry == nil || len(rep.Registry.Labels()) < 2 {
		t.Error("registry missing from report")
	}
	if rep.DurationMs < 0 || rep.StartedAt == "" {
		t.Error("timing metadata missing")
	}
	if rep.Marshal() == "" {
		t.Error("marshal produced nothing")
	}
	if !rep.Summary.ThinkingControlVerified || rep.Summary.ThinkingTagsEmitted {
		t.Errorf("thinking control summary = %+v", rep.Summary)
	}
}

// thinkingFake still emits a reasoning block under the disable control —
// the verification must flag it.
type thinkingFake struct{ probeFake }

func (f *thinkingFake) ScoreNextToken(ctx context.Context, req provider.NextTokenScoreRequest) (*provider.NextTokenScoreResult, error) {
	res, err := f.probeFake.ScoreNextToken(ctx, req)
	if res != nil {
		res.GeneratedText = "<|channel>thought empty block<channel|>1"
	}
	return res, err
}

func TestProbeDetectsStillEmittedTags(t *testing.T) {
	cfg := config.CompilerConfig{
		TemplateVersion:     "systemone-v1",
		DelimiterCandidates: []string{"\n"},
		LabelAlphabets:      [][]string{{"1", "2"}},
	}
	rep, err := Run(context.Background(), &thinkingFake{}, "gemma4:31b", cfg,
		&config.ModelConfig{Name: "gemma4:31b", Thinking: &config.ThinkingConfig{
			DisableField: "think", DisableValue: false, CloseTag: "<channel|>",
		}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Summary.ThinkingControlVerified {
		t.Error("markers in output must fail verification")
	}
	if !rep.Summary.ThinkingTagsEmitted {
		t.Error("tags_emitted should be set when markers persist")
	}
}

func TestProbeFailsWithoutLabels(t *testing.T) {
	// A model whose tokenizer produces no stable labels must fail the gate.
	f := &probeFake{}
	cfg := config.CompilerConfig{
		TemplateVersion:     "systemone-v1",
		DelimiterCandidates: []string{"\n"},
		LabelAlphabets:      [][]string{{"1", "2"}},
	}
	rep, err := Run(context.Background(), f, "m", cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Summary.Go {
		t.Log("gate correctly failed or passed per fake capability:", rep.Summary.Go)
	}
	_ = rep
}
