package provider

import (
	"crypto/sha256"
	"fmt"

	"hearim/internal/hearim/config"
)

type scoringProfile struct {
	selected       bool
	maxSelected    int
	promptLogprobs bool
	constraint     string
	space          string
	selectedField  string
}

func resolveScoring(cfg config.ProviderConfig, model string) scoringProfile {
	p := scoringProfile{constraint: "none", space: SpaceRaw, selectedField: cfg.SelectedTokenField}
	switch cfg.Engine {
	case config.EngineVLLM:
		p.selected, p.maxSelected, p.promptLogprobs, p.constraint = true, 128, true, "allowed_token_ids"
		if p.selectedField == "" {
			p.selectedField = "logprob_token_ids"
		}
	case config.EngineSGLang:
		p.selected, p.maxSelected, p.promptLogprobs = true, 256, true
		p.selectedField = "token_ids_logprob"
	case config.EngineLlamaPP:
		p.constraint = "grammar"
	case config.EngineGeneric:
		p.selected, p.maxSelected = p.selectedField != "", 128
	}
	apply := func(s *config.ProviderScoringConfig) {
		if s == nil {
			return
		}
		if s.SelectedTokenIDs != nil {
			p.selected = *s.SelectedTokenIDs
		}
		if s.MaxSelectedTokenIDs > 0 {
			p.maxSelected = s.MaxSelectedTokenIDs
		}
		if s.PromptTokenLogprobs != nil {
			p.promptLogprobs = *s.PromptTokenLogprobs
		}
		if s.Constraint != "" {
			p.constraint = s.Constraint
		}
		if s.LogprobSpace != "" {
			p.space = s.LogprobSpace
		}
	}
	apply(cfg.Scoring)
	if m := cfg.Models.Find(model); m != nil {
		apply(m.Scoring)
	}
	if p.selected && p.selectedField == "" {
		p.selectedField = "logprob_token_ids"
	}
	return p
}

// CapabilitiesForModel resolves settings without mutating shared adapter state.
func CapabilitiesForModel(a Adapter, model string) ProviderCapabilities {
	c := a.Capabilities()
	cfg, ok := adapterConfig(a)
	if !ok {
		return c
	}
	p := resolveScoring(cfg, model)
	c.CandidateScoring = []string{"top-k"}
	if p.selected {
		c.CandidateScoring = append(c.CandidateScoring, "selected-token-ids")
	}
	if p.promptLogprobs {
		c.CandidateScoring = append(c.CandidateScoring, "teacher-forced")
	}
	if p.constraint != "none" {
		c.CandidateScoring = append(c.CandidateScoring, "constrained-vocab")
	}
	c.PromptTokenLogprobs = p.promptLogprobs
	c.MaxSelectedTokenIDs = 0
	if p.selected {
		c.MaxSelectedTokenIDs = p.maxSelected
	}
	c.Endpoints = append([]EndpointProfile(nil), c.Endpoints...)
	for i := range c.Endpoints {
		c.Endpoints[i].MaxSelectedTokenIDs = c.MaxSelectedTokenIDs
		if c.Engine == config.EngineSGLang && c.Endpoints[i].Kind != config.EndpointNativeGenerate {
			c.Endpoints[i].MaxSelectedTokenIDs = 0
		}
	}
	return c
}

func CapabilitiesForRoute(a Adapter, model string, endpoint config.EndpointKind) ProviderCapabilities {
	c := CapabilitiesForModel(a, model)
	if endpoint == config.EndpointCompletions || endpoint == config.EndpointChatCompletion {
		if c.Engine == config.EngineSGLang || c.Engine == config.EngineLlamaPP {
			c.CandidateScoring = []string{"top-k"}
			c.MaxSelectedTokenIDs = 0
			c.PromptTokenLogprobs = false
		}
	}
	return c
}

func adapterConfig(a Adapter) (config.ProviderConfig, bool) {
	switch a := a.(type) {
	case *VLLMAdapter:
		return a.cfg, true
	case *SGLangAdapter:
		return a.cfg, true
	case *LlamaCppAdapter:
		return a.cfg, true
	case *GenericAdapter:
		return a.cfg, true
	case *OllamaAdapter:
		return a.cfg, true
	default:
		return config.ProviderConfig{}, false
	}
}

func ScoringProfileKey(a Adapter, model string) string {
	cfg, ok := adapterConfig(a)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("scoring-v2:%s:%+v", cfg.Engine, resolveScoring(cfg, model)))))
}

func validCandidateIDs(req NextTokenScoreRequest) bool {
	if len(req.CandidateTokenIDs) == 0 || len(req.CandidateTokenIDs) != len(req.CandidateTokenTexts) {
		return false
	}
	for _, id := range req.CandidateTokenIDs {
		if id < 0 {
			return false
		}
	}
	return true
}

func prepareScoring(cfg config.ProviderConfig, req NextTokenScoreRequest) (NextTokenScoreRequest, error) {
	p := resolveScoring(cfg, req.Model.Model)
	req.SelectedTokenField = ""
	if p.selected && !req.DisableSelectedTokenIDs && validCandidateIDs(req) && len(req.CandidateTokenIDs) <= p.maxSelected {
		req.SelectedTokenField = p.selectedField
	}
	req.ConstraintMode, req.LogprobSpace = p.constraint, p.space
	if req.ConstrainToCandidates {
		if req.WaitClose {
			return req, fmt.Errorf("provider: candidate constraint cannot be used before the reasoning close tag")
		}
		switch p.constraint {
		case "none":
			return req, fmt.Errorf("provider: candidate constraint disabled/unsupported for %s/%s", cfg.Engine, req.Model.Model)
		case "allowed_token_ids":
			if !validCandidateIDs(req) {
				return req, fmt.Errorf("provider: allowed_token_ids requires resolved candidate IDs")
			}
		case "grammar":
			if len(req.CandidateTokenTexts) == 0 {
				return req, fmt.Errorf("provider: grammar requires candidate labels")
			}
		}
	}
	return req, nil
}
