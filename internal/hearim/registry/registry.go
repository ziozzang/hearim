// Package registry builds and persists the per-model Token Label Registry of
// TODO.md §6: which answer labels are stable single tokens in the exact
// prompt context, under which delimiter, verified by real completion probes.
//
// Labels are chosen for being a single token in context — not for human
// readability — so "A" vs " A" and "1" vs " 1" are distinguished by probing,
// never assumed.
package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"hearim/internal/hearim/config"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/scoring"
)

// LabelEntry is one verified label token.
type LabelEntry struct {
	Label     string `json:"label"`
	TokenID   int    `json:"token_id"` // -1 when only text-matched
	TokenText string `json:"token_text"`
}

// Registry is the §6.3 record for one backend model + endpoint + template.
type Registry struct {
	BackendModel         string                  `json:"backend_model"`
	ModelDigest          string                  `json:"model_digest,omitempty"`
	TokenizerRevision    string                  `json:"tokenizer_revision"`
	Endpoint             string                  `json:"endpoint"`
	ChatTemplateHash     string                  `json:"chat_template_hash,omitempty"`
	ReasoningProfileHash string                  `json:"reasoning_profile_hash,omitempty"`
	TemplateVersion      string                  `json:"template_version"`
	Delimiter            string                  `json:"delimiter"`
	BoundaryPolicy       string                  `json:"boundary_policy"` // exact-prefix-plus-one | probe-only
	Alphabets            map[string][]LabelEntry `json:"alphabets"`
	PreferredAlphabet    string                  `json:"preferred_alphabet"`
	MaxSingleTokenLabels int                     `json:"max_single_token_labels"`
	MaxExactCandidates   int                     `json:"max_exact_candidates"`
	// Verified means the completion probe identified label logprobs (§6.1
	// step 10).
	Verified bool   `json:"verified"`
	ProbeErr string `json:"probe_error,omitempty"`
	BuiltAt  string `json:"built_at"`
}

// Options configure a build.
type Options struct {
	BackendModel      string
	ModelDigest       string
	TokenizerRevision string
	Endpoint          string
	ChatTemplateHash  string
	TemplateVersion   string
	// PromptBase is the compiled prompt ending at the answer marker.
	PromptBase string
	// DelimiterCandidates tried in order (compiler config).
	DelimiterCandidates []string
	// Alphabets: ordered label sets; names are assigned as numeric,
	// upper_alpha, alpha3, alpha4...
	Alphabets [][]string
}

// alphabetName names the i-th alphabet per §6.3.
func alphabetName(i int) string {
	switch i {
	case 0:
		return "numeric"
	case 1:
		return "upper_alpha"
	default:
		return fmt.Sprintf("alpha%d", i+1)
	}
}

// Build constructs the registry for one model/endpoint pair (TODO.md §6.1).
//
// Tokenizer access priority (§6.1):
//  1. backend tokenize endpoint -> exact-prefix-plus-one verification
//  2. matching local tokenizer   -> not implemented in v1 (needs HF
//     tokenizer linkage); treated as absence
//  3. otherwise: limited inference probe via real completion logprobs
func Build(ctx context.Context, adpt provider.Adapter, opts Options) (*Registry, error) {
	if len(opts.Alphabets) == 0 {
		return nil, fmt.Errorf("registry: no label alphabets configured")
	}
	reg := &Registry{
		BackendModel:      opts.BackendModel,
		ModelDigest:       opts.ModelDigest,
		TokenizerRevision: opts.TokenizerRevision,
		Endpoint:          opts.Endpoint,
		ChatTemplateHash:  opts.ChatTemplateHash,
		TemplateVersion:   opts.TemplateVersion,
		Alphabets:         map[string][]LabelEntry{},
		BoundaryPolicy:    "probe-only",
		BuiltAt:           time.Now().UTC().Format(time.RFC3339),
	}
	caps := adpt.Capabilities()
	reg.MaxExactCandidates = maxExactCandidates(caps)

	if len(opts.DelimiterCandidates) == 0 {
		opts.DelimiterCandidates = []string{"\n"}
	}

	// Attempt the exact tokenize path (priority 1).
	tokenizeWorks := true
	if _, err := adpt.Tokenize(ctx, opts.BackendModel, opts.PromptBase); err != nil {
		tokenizeWorks = false
	}

	if tokenizeWorks {
		reg.BoundaryPolicy = "exact-prefix-plus-one"
		// Find the first delimiter where every label of some alphabet is a
		// stable single token (§6.1 steps 6-9).
		for _, delim := range opts.DelimiterCandidates {
			prefix := opts.PromptBase + delim
			base, err := adpt.Tokenize(ctx, opts.BackendModel, prefix)
			if err != nil {
				continue
			}
			built := 0
			for ai, alphabet := range opts.Alphabets {
				entries := make([]LabelEntry, 0, len(alphabet))
				for _, label := range alphabet {
					full, err := adpt.Tokenize(ctx, opts.BackendModel, prefix+label)
					if err != nil {
						break
					}
					if !scoring.IsStableSingleToken(base, full) {
						break
					}
					entries = append(entries, LabelEntry{
						Label:     label,
						TokenID:   full[len(full)-1],
						TokenText: label,
					})
				}
				if len(entries) == len(alphabet) && len(entries) >= 2 {
					name := alphabetName(ai)
					if reg.PreferredAlphabet == "" {
						reg.PreferredAlphabet = name
						reg.Delimiter = delim
					}
					reg.Alphabets[name] = entries
					built++
				}
			}
			if reg.PreferredAlphabet != "" {
				break // first delimiter with a full stable alphabet wins
			}
			_ = built
		}
	}

	if reg.PreferredAlphabet == "" {
		// Priority 3: limited inference probe — ask the backend to score the
		// labels at the answer position; entries returning the label text
		// exactly are single tokens in this context.
		prefix := opts.PromptBase + opts.DelimiterCandidates[0]
		var texts []string
		for _, alphabet := range opts.Alphabets {
			for _, label := range alphabet {
				if label != "" {
					texts = append(texts, label)
				}
			}
			break // probe the primary alphabet only
		}
		if len(texts) == 0 {
			return nil, fmt.Errorf("registry: empty primary alphabet")
		}
		res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
			Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
			Endpoint:            endpointKind(opts.Endpoint),
			PromptText:          prefix,
			CandidateTokenTexts: texts,
			TopK:                caps.MaxTopLogprobs,
		})
		if err != nil {
			return nil, fmt.Errorf("registry: inference probe failed: %w", err)
		}
		entries := make([]LabelEntry, 0, len(texts))
		for i, text := range texts {
			if lp, ok := res.CandidateLogprobs[i]; ok && lp < 0 {
				entries = append(entries, LabelEntry{Label: text, TokenID: -1, TokenText: text})
			}
		}
		if len(entries) >= 2 {
			reg.Alphabets[alphabetName(0)] = entries
			reg.PreferredAlphabet = alphabetName(0)
			reg.Delimiter = opts.DelimiterCandidates[0]
		}
	}

	if reg.PreferredAlphabet == "" {
		return nil, fmt.Errorf("registry: no stable single-token labels for %s", opts.BackendModel)
	}
	reg.MaxSingleTokenLabels = len(reg.Alphabets[reg.PreferredAlphabet])

	// Step 10: completion probe must identify label logprobs.
	reg.runCompletionProbe(ctx, adpt, opts)
	return reg, nil
}

// runCompletionProbe verifies that a real 1-token completion returns
// identifiable logprobs for the registered labels (§6.1 step 10).
func (r *Registry) runCompletionProbe(ctx context.Context, adpt provider.Adapter, opts Options) {
	entries := r.Alphabets[r.PreferredAlphabet]
	ids := make([]int, 0, len(entries))
	texts := make([]string, 0, len(entries))
	for _, e := range entries {
		ids = append(ids, e.TokenID)
		texts = append(texts, e.TokenText)
	}
	res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
		Endpoint:            endpointKind(r.Endpoint),
		PromptText:          opts.PromptBase + r.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                adpt.Capabilities().MaxTopLogprobs,
	})
	if err != nil {
		r.ProbeErr = err.Error()
		return
	}
	found := 0
	for i := range texts {
		if _, ok := res.CandidateLogprobs[i]; ok {
			found++
		}
	}
	r.Verified = found >= 2
	if !r.Verified {
		r.ProbeErr = fmt.Sprintf("completion probe found %d/%d labels", found, len(texts))
	}
}

func endpointKind(s string) config.EndpointKind {
	return config.EndpointKind(s)
}

func maxExactCandidates(caps provider.ProviderCapabilities) int {
	if caps.SupportsSelectedTokenIDs() && caps.MaxSelectedTokenIDs > 0 {
		return caps.MaxSelectedTokenIDs
	}
	if caps.MaxTopLogprobs > 0 {
		return caps.MaxTopLogprobs
	}
	return 20
}

// Labels returns the stable label sequence of the preferred alphabet.
func (r *Registry) Labels() []LabelEntry {
	return r.Alphabets[r.PreferredAlphabet]
}

// Bind maps compiled candidate label texts to registry entries. Returns
// ids (-1 when only text-bound), texts, and whether all bound.
func (r *Registry) Bind(tokenTexts []string) (ids []int, boundTexts []string, all bool) {
	byLabel := map[string]LabelEntry{}
	for _, e := range r.Labels() {
		byLabel[e.Label] = e
	}
	ids = make([]int, len(tokenTexts))
	boundTexts = make([]string, len(tokenTexts))
	all = true
	for i, tt := range tokenTexts {
		if e, ok := byLabel[tt]; ok {
			ids[i] = e.TokenID
			boundTexts[i] = e.TokenText
		} else {
			ids[i] = -1
			boundTexts[i] = tt
			all = false
		}
	}
	return ids, boundTexts, all
}

// Ready lists the §6.5 readiness failures for a required candidate count.
// A probe-only registry (no tokenizer endpoint, labels verified by real
// completion probes) is ready when its completion probe passed; the missing
// tokenizer revision is then a warning, not a failure — Ollama is exactly
// this shape (TODO.md §6.1 priority 3).
func (r *Registry) Ready(requiredCandidates int) (bool, []string) {
	var fails, warns []string
	if r.BoundaryPolicy != "exact-prefix-plus-one" {
		warns = append(warns, "tokenizer_unverified")
	}
	if !r.Verified {
		fails = append(fails, "completion_probe_failed: "+r.ProbeErr)
	}
	labels := r.Labels()
	if len(labels) < 2 {
		fails = append(fails, "fewer_than_two_single_token_labels")
	}
	seen := map[int]bool{}
	for _, e := range labels {
		if e.TokenID >= 0 {
			if seen[e.TokenID] {
				fails = append(fails, "duplicate_token_id_mapping")
			}
			seen[e.TokenID] = true
		}
	}
	if requiredCandidates > r.MaxExactCandidates {
		fails = append(fails, fmt.Sprintf("required_candidates_%d_exceeds_max_exact_%d", requiredCandidates, r.MaxExactCandidates))
	}
	if len(fails) == 0 {
		return true, warns
	}
	return false, append(fails, warns...)
}

// OptionsKey derives the persistence key from build options without
// building, so callers can check for a cached registry first.
func OptionsKey(opts Options) string {
	r := &Registry{
		BackendModel:      opts.BackendModel,
		ModelDigest:       opts.ModelDigest,
		TokenizerRevision: opts.TokenizerRevision,
		TemplateVersion:   opts.TemplateVersion,
		Endpoint:          opts.Endpoint,
	}
	// Delimiter participates via the built registry's Key; before building
	// the first delimiter candidate stands in.
	if len(opts.DelimiterCandidates) > 0 {
		r.Delimiter = opts.DelimiterCandidates[0]
	}
	return r.Key()
}

// Key identifies the registry file on disk.
func (r *Registry) Key() string {
	h := sha256.New()
	h.Write([]byte(r.BackendModel))
	h.Write([]byte{0})
	h.Write([]byte(r.ModelDigest))
	h.Write([]byte{0})
	h.Write([]byte(r.TokenizerRevision))
	h.Write([]byte{0})
	h.Write([]byte(r.TemplateVersion))
	h.Write([]byte{0})
	h.Write([]byte(r.Endpoint))
	h.Write([]byte{0})
	h.Write([]byte(r.Delimiter))
	return hex.EncodeToString(h.Sum(nil))[:24]
}

// Save persists the registry as JSON (§6.1 step 11).
func (r *Registry) Save(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "registry-"+r.Key()+".json")
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// Load reads a persisted registry by key; returns nil when absent.
func Load(dir, key string) (*Registry, error) {
	path := filepath.Join(dir, "registry-"+key+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var r Registry
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}
