// Package compile turns validated Jev requests into EvaluationPlans: the
// internal normal form of TODO.md §4, the prompt templates of §5, the label
// mapping rules of §6.4, and the prefix keys of §8.2.
package compile

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"hearim/internal/hearim/canon"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/jev"
)

// Reduction names how candidate probabilities become the public answer.
type Reduction string

const (
	ReductionCategorical     Reduction = "categorical"
	ReductionOrdinalMean     Reduction = "ordinal-mean"
	ReductionTrueProbability Reduction = "true-probability"
)

// Candidate is one scoreable answer option.
type Candidate struct {
	// TokenText is the single-token label the model scores (e.g. "A", " 1").
	TokenText string
	// TokenID is resolved by the token label registry; 0 until bound.
	TokenID int
	// PublicValue is choice name (string) or score level index (int) or
	// noul truth (bool).
	PublicValue any
	// PublicKey is the JSON key used in the probabilities map.
	PublicKey string
	// SequenceText is the full public continuation for the choice-text
	// scorer (choice option name; unused for label paths).
	SequenceText string
}

// CompiledQuestion is a single question compiled against a shared state.
type CompiledQuestion struct {
	ID           string
	Type         jev.QuestionType
	PromptPrefix string // shared invariant prefix for this layout
	PromptSuffix string // question-specific tail including answer marker
	Candidates   []Candidate
	Reduction    Reduction
}

// PlanAPIVersion identifies the internal normal form (TODO.md §4).
const PlanAPIVersion = "systemone-compat-v1"

// EvaluationPlan is the compiled, backend-ready evaluation (TODO.md §4).
type EvaluationPlan struct {
	APIVersion      string
	PublicModel     string
	BackendModel    string
	CanonicalState  string
	StateHash       string
	TemplateVersion string
	Layout          config.Layout
	Questions       []*CompiledQuestion
	// Delimiter sits between the answer marker and the label token; the
	// registry probes delimiter candidates and records the adopted one.
	Delimiter string
}

// Compiler compiles validated requests into evaluation plans.
type Compiler struct {
	Config config.CompilerConfig
}

// New creates a compiler from configuration.
func New(cfg config.CompilerConfig) *Compiler {
	return &Compiler{Config: cfg}
}

// Compile builds the plan for a parsed request.
func (c *Compiler) Compile(pr *jev.ParsedRequest, backendModel string) (*EvaluationPlan, error) {
	canonicalState, err := canon.String(pr.State)
	if err != nil {
		return nil, fmt.Errorf("compile: canonicalize state: %w", err)
	}
	stateHash := sha256.Sum256([]byte(canonicalState))

	layout := c.Config.DefaultLayout
	if layout == "" {
		layout = config.LayoutStateMajor
	}
	// TODO.md §5.2: multiple questions in one request -> state-major.
	if len(pr.Questions) >= 2 && layout != config.LayoutStateMajor {
		layout = config.LayoutStateMajor
	}

	plan := &EvaluationPlan{
		APIVersion:      PlanAPIVersion,
		PublicModel:     pr.Model,
		BackendModel:    backendModel,
		CanonicalState:  canonicalState,
		StateHash:       hex.EncodeToString(stateHash[:]),
		TemplateVersion: c.Config.TemplateVersion,
		Layout:          layout,
		Delimiter:       defaultDelimiter(c.Config.DelimiterCandidates),
	}

	for _, q := range pr.Questions {
		cq, err := c.compileQuestion(q, canonicalState, layout)
		if err != nil {
			return nil, err
		}
		plan.Questions = append(plan.Questions, cq)
	}
	return plan, nil
}

func defaultDelimiter(candidates []string) string {
	if len(candidates) > 0 {
		return candidates[0]
	}
	return "\n"
}

func (c *Compiler) compileQuestion(q *jev.ParsedQuestion, canonicalState string, layout config.Layout) (*CompiledQuestion, error) {
	cands, reduction, err := c.mapCandidates(q)
	if err != nil {
		return nil, err
	}

	instructions := ""
	if q.HasInstructions {
		s, err := canon.String(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("compile: question %s: instructions: %w", q.ID, err)
		}
		instructions = s
	}
	criteria := renderCriteria(q, cands)

	stateBlock := renderStateBlock(canonicalState)
	questionBlock := renderQuestionBlock(q.Type, instructions)
	criteriaBlock := renderCriteriaBlock(criteria)
	marker := answerMarker + planDelimiterHint

	cq := &CompiledQuestion{
		ID:         q.ID,
		Type:       q.Type,
		Candidates: cands,
		Reduction:  reduction,
	}

	switch layout {
	case config.LayoutStateMajor:
		// [fixed system][state][question][criteria][marker]
		cq.PromptPrefix = systemBlock(c.Config.TemplateVersion) + stateBlock
		cq.PromptSuffix = questionBlock + criteriaBlock + marker
	case config.LayoutRubricMajor:
		// [fixed system][question][criteria][state][marker]
		cq.PromptPrefix = systemBlock(c.Config.TemplateVersion) + questionBlock + criteriaBlock
		cq.PromptSuffix = stateBlock + marker
	default:
		return nil, fmt.Errorf("compile: unknown layout %q", layout)
	}
	return cq, nil
}

// mapCandidates assigns single-token labels per TODO.md §6.4:
//
//	2..10   -> numeric alphabet "1".."9","0"
//	11..26  -> upper alpha
//	more    -> next verified alphabet, else rejected (strict mode)
//
// choice preserves request criteria insertion order; score maps levels
// 0..K-1; noul fixes 1=true, 0=false. Question ids and option names are
// never used as answer tokens.
func (c *Compiler) mapCandidates(q *jev.ParsedQuestion) ([]Candidate, Reduction, error) {
	alphabets := c.Config.LabelAlphabets
	if len(alphabets) == 0 {
		alphabets = defaultAlphabets
	}

	switch q.Type {
	case jev.TypeChoice:
		cands := make([]Candidate, 0, len(q.ChoiceNames))
		labels, err := pickLabels(alphabets, len(q.ChoiceNames))
		if err != nil {
			return nil, "", err
		}
		for i, name := range q.ChoiceNames {
			cands = append(cands, Candidate{
				TokenText:    labels[i],
				PublicValue:  name,
				PublicKey:    name,
				SequenceText: choiceSequenceText(name, q.ChoiceDescriptions[name]),
			})
		}
		return cands, ReductionCategorical, nil

	case jev.TypeScore:
		labels, err := pickLabels(alphabets, len(q.Levels))
		if err != nil {
			return nil, "", err
		}
		cands := make([]Candidate, 0, len(q.Levels))
		for i := range q.Levels {
			cands = append(cands, Candidate{
				TokenText:    labels[i],
				PublicValue:  i,
				PublicKey:    fmt.Sprintf("%d", i),
				SequenceText: q.Levels[i],
			})
		}
		return cands, ReductionOrdinalMean, nil

	case jev.TypeNoul:
		return []Candidate{
			{TokenText: "1", PublicValue: true, PublicKey: "true",
				SequenceText: noulSequenceText(true, q.TrueDesc)},
			{TokenText: "0", PublicValue: false, PublicKey: "false",
				SequenceText: noulSequenceText(false, q.FalseDesc)},
		}, ReductionTrueProbability, nil

	default:
		return nil, "", fmt.Errorf("compile: question %s: unknown type %q", q.ID, q.Type)
	}
}

var defaultAlphabets = [][]string{
	{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
	{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
		"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"},
}

// pickLabels selects a label sequence long enough for n candidates from the
// configured alphabets, in order. TODO.md §6.4: numeric first, then upper
// alpha, then further verified alphabets; exhaustion is a strict-mode
// rejection (choice-text continuation is the fallback path, not more labels).
func pickLabels(alphabets [][]string, n int) ([]string, error) {
	if n < 2 {
		return nil, fmt.Errorf("compile: need at least 2 candidates, got %d", n)
	}
	for _, a := range alphabets {
		if len(a) >= n {
			return a[:n], nil
		}
	}
	return nil, fmt.Errorf("compile: %d candidates exceed verified single-token labels (%d); "+
		"use the choice-text continuation scorer or extend label_alphabets", n, totalLabels(alphabets))
}

func totalLabels(alphabets [][]string) int {
	total := 0
	for _, a := range alphabets {
		total += len(a)
	}
	return total
}

// Prompt returns the full upstream prompt for one question.
func (p *EvaluationPlan) Prompt(q *CompiledQuestion) string {
	return q.PromptPrefix + q.PromptSuffix
}

// PromptWithDelimiter returns the prompt with a registry-adopted delimiter
// before the label token.
func (p *EvaluationPlan) PromptWithDelimiter(q *CompiledQuestion, delimiter string) string {
	return q.PromptPrefix + q.PromptSuffix + delimiter
}

// CommonPrefixHash is the cache key for the shared prefix of this plan
// (TODO.md §8.2). The delimiter is excluded: it belongs to the label, and
// the registry may adopt different delimiters per model without breaking
// the shared prompt cache across models would anyway differ by digest.
func (p *EvaluationPlan) CommonPrefixHash(modelDigest, tokenizerRevision string) string {
	prefix := ""
	if len(p.Questions) > 0 {
		prefix = p.Questions[0].PromptPrefix
		if p.Layout == config.LayoutStateMajor {
			// All questions share this prefix in state-major layout.
			for _, q := range p.Questions {
				if q.PromptPrefix != prefix {
					prefix = commonStringPrefix(prefix, q.PromptPrefix)
				}
			}
		}
	}
	return PrefixKey(modelDigest, tokenizerRevision, p.TemplateVersion, string(p.Layout), prefix)
}

func commonStringPrefix(a, b string) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return a[:i]
}

// PrefixKey computes the §8.2 prefix cache key:
//
//	SHA256(model_digest || tokenizer_revision || template_version || layout || prefix_bytes)
//
// Latest aliases, timestamps, request ids and trace ids never enter the key.
func PrefixKey(modelDigest, tokenizerRevision, templateVersion, layout, prefix string) string {
	h := sha256.New()
	h.Write([]byte(modelDigest))
	h.Write([]byte{0})
	h.Write([]byte(tokenizerRevision))
	h.Write([]byte{0})
	h.Write([]byte(templateVersion))
	h.Write([]byte{0})
	h.Write([]byte(layout))
	h.Write([]byte{0})
	h.Write([]byte(prefix))
	return hex.EncodeToString(h.Sum(nil))
}
