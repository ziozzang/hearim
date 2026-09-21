package registry

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/scoring"
	"strings"
)

// fakeAdapter simulates an engine with a BPE-ish tokenizer that merges
// ":" + digit into one token, so the ":" delimiter must be rejected in favor
// of "\n" (TODO.md §6.1 step 8).
type fakeAdapter struct {
	tokenizerWorks bool
	scoreCalls     int
}

func (f *fakeAdapter) ID() string                { return "fake" }
func (f *fakeAdapter) Engine() config.EngineKind { return config.EngineVLLM }
func (f *fakeAdapter) Capabilities() provider.ProviderCapabilities {
	return provider.ProviderCapabilities{
		Engine:              config.EngineVLLM,
		Tokenizer:           "remote",
		CandidateScoring:    []string{"selected-token-ids", "top-k"},
		PromptTokenLogprobs: true,
		MaxTopLogprobs:      20,
		MaxSelectedTokenIDs: 256,
		MaxConcurrency:      4,
		Endpoints: []provider.EndpointProfile{
			{Kind: config.EndpointCompletions, NextTokenLogprobsVerified: true},
		},
	}
}

// tokenize: one token per rune, except ":" + digit merges (ids 9000+d).
func (f *fakeAdapter) Tokenize(ctx context.Context, model, text string) ([]int, error) {
	if !f.tokenizerWorks {
		return nil, fmt.Errorf("no tokenizer endpoint")
	}
	var ids []int
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		if runes[i] == ':' && i+1 < len(runes) && runes[i+1] >= '0' && runes[i+1] <= '9' {
			ids = append(ids, 9000+int(runes[i+1]-'0'))
			i++
			continue
		}
		ids = append(ids, int(runes[i])+1)
	}
	return ids, nil
}

func (f *fakeAdapter) ScoreNextToken(ctx context.Context, req provider.NextTokenScoreRequest) (*provider.NextTokenScoreResult, error) {
	f.scoreCalls++
	// Faithful to the fake tokenizer: a label is "present" only when it is
	// a stable single token after this exact prompt.
	prompt := req.PromptText
	if prompt == "" && len(req.PromptTokenIDs) > 0 {
		prompt = strings.Repeat("x", len(req.PromptTokenIDs))
	}
	var base []int
	if len(req.PromptTokenIDs) > 0 {
		base = req.PromptTokenIDs
	} else if f.tokenizerWorks {
		base, _ = f.Tokenize(ctx, "", prompt)
	} else {
		base = []int{1, 2, 3}
	}
	out := map[int]float64{}
	for i, text := range req.CandidateTokenTexts {
		full, _ := f.Tokenize(ctx, "", prompt+text)
		if f.tokenizerWorks && scoring.IsStableSingleToken(base, full) {
			out[i] = -float64(i + 1)
		} else if !f.tokenizerWorks {
			out[i] = -float64(i + 1)
		}
	}
	return &provider.NextTokenScoreResult{
		CandidateLogprobs:    out,
		AllCandidatesPresent: len(out) == len(req.CandidateTokenTexts),
		PromptTokens:         42,
		ScoringMethod:        "selected-token-ids",
		ProbabilitySpace:     provider.SpaceRaw,
	}, nil
}

func (f *fakeAdapter) ScoreContinuations(ctx context.Context, req provider.ContinuationScoreRequest) (*provider.ContinuationScoreResult, error) {
	res := &provider.ContinuationScoreResult{ScoringMethod: "teacher-forced-choice-text"}
	for range req.Continuations {
		res.Conditional = append(res.Conditional, provider.ContinuationScore{LogProb: -1, TokenLen: 3})
	}
	return res, nil
}
func (f *fakeAdapter) Health(ctx context.Context) (provider.Health, error) {
	return provider.Health{OK: true}, nil
}
func (f *fakeAdapter) ChatRender() bool { return false }
func (f *fakeAdapter) RenderChat(p *compile.EvaluationPlan, qi int) []compile.ChatMessage {
	return nil
}

func testOptions() Options {
	return Options{
		BackendModel:        "gemma4:31b",
		TokenizerRevision:   "rev-1",
		Endpoint:            string(config.EndpointCompletions),
		TemplateVersion:     "systemone-v1",
		PromptBase:          "SYSTEM<state>x</state><question/><criteria><answer-label>\n",
		DelimiterCandidates: []string{":", "\n"},
		Alphabets: [][]string{
			{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
		},
	}
}

func TestBuildExactPrefixPlusOne(t *testing.T) {
	f := &fakeAdapter{tokenizerWorks: true}
	reg, err := Build(context.Background(), f, testOptions())
	if err != nil {
		t.Fatal(err)
	}
	if reg.BoundaryPolicy != "exact-prefix-plus-one" {
		t.Errorf("boundary policy = %s", reg.BoundaryPolicy)
	}
	// ":" merges with digits -> must fall through to "\n".
	if reg.Delimiter != "\n" {
		t.Errorf("delimiter = %q, want \"\\n\"", reg.Delimiter)
	}
	if reg.PreferredAlphabet != "numeric" {
		t.Errorf("preferred = %s", reg.PreferredAlphabet)
	}
	if len(reg.Labels()) != 10 {
		t.Errorf("labels = %d", len(reg.Labels()))
	}
	if !reg.Verified {
		t.Errorf("completion probe should pass: %s", reg.ProbeErr)
	}
	if reg.MaxExactCandidates != 256 {
		t.Errorf("max exact = %d", reg.MaxExactCandidates)
	}
}

func TestBuildColonDelimiterRejected(t *testing.T) {
	// Explicitly verify the boundary math: with ":" every digit label fails
	// the exact-prefix-plus-one invariant.
	f := &fakeAdapter{tokenizerWorks: true}
	opts := testOptions()
	opts.DelimiterCandidates = []string{":"}
	reg, err := Build(context.Background(), f, opts)
	if err == nil {
		t.Fatalf("build with only ':' delimiter should fail, got %+v", reg)
	}
}

func TestBuildInferenceProbeFallback(t *testing.T) {
	f := &fakeAdapter{tokenizerWorks: false}
	reg, err := Build(context.Background(), f, testOptions())
	if err != nil {
		t.Fatal(err)
	}
	if reg.BoundaryPolicy != "probe-only" {
		t.Errorf("boundary policy = %s", reg.BoundaryPolicy)
	}
	for _, e := range reg.Labels() {
		if e.TokenID != -1 {
			t.Errorf("inference probe entries must be text-only, got %d", e.TokenID)
		}
	}
	if len(reg.Labels()) < 2 {
		t.Errorf("labels = %d", len(reg.Labels()))
	}
	// Probe-only with a passing completion probe is ready, but the readiness
	// report carries the tokenizer warning (Ollama shape, §6.1 priority 3).
	ok, notes := reg.Ready(4)
	if !ok {
		t.Errorf("verified probe-only registry should be ready: %v", notes)
	}
	found := false
	for _, n := range notes {
		if n == "tokenizer_unverified" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected tokenizer_unverified warning, got %v", notes)
	}
}

func TestReadyConditions(t *testing.T) {
	f := &fakeAdapter{tokenizerWorks: true}
	reg, _ := Build(context.Background(), f, testOptions())
	if ok, fails := reg.Ready(4); !ok {
		t.Errorf("should be ready for 4 candidates, fails: %v", fails)
	}
	if ok, _ := reg.Ready(300); ok {
		t.Error("300 candidates exceed max_exact_candidates")
	}

	reg.Verified = false
	if ok, fails := reg.Ready(4); ok {
		t.Errorf("unverified registry must not be ready: %v", fails)
	}
}

func TestBind(t *testing.T) {
	f := &fakeAdapter{tokenizerWorks: true}
	reg, _ := Build(context.Background(), f, testOptions())
	ids, texts, all := reg.Bind([]string{"1", "2", "3", "4"})
	if !all {
		t.Error("bind should cover all labels")
	}
	if ids[0] != int('1')+1 {
		t.Errorf("id[0] = %d, want %d", ids[0], int('1')+1)
	}
	if texts[0] != "1" {
		t.Errorf("text[0] = %q", texts[0])
	}
	if _, _, all := reg.Bind([]string{"Z"}); all {
		t.Error("unknown label must not bind")
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	f := &fakeAdapter{tokenizerWorks: true}
	reg, _ := Build(context.Background(), f, testOptions())
	dir := t.TempDir()
	path, err := reg.Save(dir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "registry-"+reg.Key()+".json" {
		t.Errorf("path = %s", path)
	}
	loaded, err := Load(dir, reg.Key())
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil || loaded.Key() != reg.Key() || loaded.Delimiter != reg.Delimiter ||
		len(loaded.Labels()) != len(reg.Labels()) || loaded.Verified != reg.Verified {
		t.Errorf("roundtrip mismatch: %+v", loaded)
	}
	if info, err := os.Stat(path); err != nil || info.Mode().Perm()&0o077 != 0 {
		t.Errorf("registry file should be user-only: %v", err)
	}
	if none, err := Load(dir, "doesnotexist000000"); err != nil || none != nil {
		t.Errorf("missing registry should return nil,nil: %v %v", none, err)
	}
}
