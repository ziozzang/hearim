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
	"strconv"
	"strings"
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
	CloseTag             string                  `json:"close_tag,omitempty"`
	UserSuffix           string                  `json:"user_suffix,omitempty"`
	Delimiter            string                  `json:"delimiter"`
	BoundaryPolicy       string                  `json:"boundary_policy"` // exact-prefix-plus-one | probe-only
	Alphabets            map[string][]LabelEntry `json:"alphabets"`
	PreferredAlphabet    string                  `json:"preferred_alphabet"`
	MaxSingleTokenLabels int                     `json:"max_single_token_labels"`
	MaxExactCandidates   int                     `json:"max_exact_candidates"`
	// TopN is the smallest top_logprobs request that recovered every label
	// of the preferred alphabet at the probe context (auto-tuned sweep).
	// TopNCap is the server's accepted maximum (0 = not discovered).
	TopN          int `json:"top_n,omitempty"`
	TopNCap       int `json:"top_n_cap,omitempty"`
	TopNRecovered int `json:"top_n_recovered,omitempty"`
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
	// CloseTag (e.g. "</think>") is preloaded between the answer marker and
	// the delimiter for <think>-style models: labels are then scored in the
	// post-thinking context (TODO.md §3.4 technique).
	CloseTag string
	// UserSuffix (e.g. legacy-GLM "/no_think") is appended to the tail of
	// the user content before the close tag; probes measure under it so
	// label boundaries and N match production.
	UserSuffix string
	// DelimiterCandidates tried in order (compiler config).
	DelimiterCandidates []string
	// ReasoningField/ReasoningValue/NoReasoning replicate the production
	// thinking control for every probe: N and label boundaries must be
	// measured under the same reasoning configuration the live traffic
	// uses, or the discovered values are wrong for it.
	ReasoningField string
	ReasoningValue any
	NoReasoning    bool
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
		CloseTag:          opts.CloseTag,
		UserSuffix:        opts.UserSuffix,
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
		// stable single token (§6.1 steps 6-9). Probing happens in the
		// post-close-tag context so verified boundaries transfer exactly.
		for _, delim := range opts.DelimiterCandidates {
			prefix := opts.PromptBase + opts.UserSuffix + opts.CloseTag + delim
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
		// exactly are single tokens in this context. Delimiter candidates
		// and alphabets are tried in order, because without a tokenizer
		// there is no way to predict which context keeps labels whole.
		var lastErr error
	DelimLoop:
		for _, delim := range opts.DelimiterCandidates {
			prefix := opts.PromptBase + opts.UserSuffix + opts.CloseTag + delim
			for ai, alphabet := range opts.Alphabets {
				texts := make([]string, 0, len(alphabet))
				for _, label := range alphabet {
					if label != "" {
						texts = append(texts, label)
					}
				}
				if len(texts) < 2 {
					continue
				}
				probeReq := provider.NextTokenScoreRequest{
					Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
					Endpoint:            endpointKind(opts.Endpoint),
					PromptText:          prefix,
					CandidateTokenTexts: texts,
					TopK:                caps.MaxTopLogprobs,
					NoReasoning:         opts.NoReasoning,
					ReasoningField:      opts.ReasoningField,
					ReasoningValue:      opts.ReasoningValue,
				}
				res, err := adpt.ScoreNextToken(ctx, probeReq)
				if err != nil {
					// Tight-range servers (Qwen token plan: [0,5]) reject the
					// default N outright; retry once at the parsed cap.
					if cap := parseTopNCap(err.Error()); cap > 0 && cap < probeReq.TopK {
						probeReq.TopK = cap
						res, err = adpt.ScoreNextToken(ctx, probeReq)
					}
				}
				if err != nil {
					lastErr = err
					continue
				}
				entries := make([]LabelEntry, 0, len(texts))
				for i, text := range texts {
					if lp, ok := res.CandidateLogprobs[i]; ok && lp < 0 {
						entries = append(entries, LabelEntry{Label: text, TokenID: -1, TokenText: text})
					}
				}
				if len(entries) >= 2 {
					reg.Alphabets[alphabetName(ai)] = entries
					reg.PreferredAlphabet = alphabetName(ai)
					reg.Delimiter = delim
					break DelimLoop
				}
			}
		}
		if reg.PreferredAlphabet == "" && lastErr != nil {
			return nil, fmt.Errorf("registry: inference probe failed: %w", lastErr)
		}
	}

	if reg.PreferredAlphabet == "" {
		return nil, fmt.Errorf("registry: no stable single-token labels for %s", opts.BackendModel)
	}
	reg.MaxSingleTokenLabels = len(reg.Alphabets[reg.PreferredAlphabet])

	// Optimal-N discovery: sweep the top_logprobs ladder to find the
	// smallest request that recovers every label, and the server's cap
	// (servers ERROR on oversized N — Ollama: "must be between 0 and 20" —
	// instead of clamping, so a blind multiply-on-retry just crashes into
	// the limit).
	reg.sweepTopN(ctx, adpt, opts)

	// Step 10: completion probe must identify label logprobs.
	reg.runCompletionProbe(ctx, adpt, opts)
	return reg, nil
}

// topNLadder is the sweep order: small values first, bounded by typical
// server caps. 20 covers Ollama; 32/64 cover permissive servers.
var topNLadder = []int{4, 8, 10, 16, 20, 32, 64}

// registryLabels returns the preferred alphabet's label strings.
func registryLabels(r *Registry) []string {
	out := []string{}
	for _, e := range r.Alphabets[r.PreferredAlphabet] {
		out = append(out, e.Label)
	}
	return out
}

// sweepTopN finds the minimal N at which every candidate label is visible
// in the top-N logit entries, and the server cap. A label that never shows
// up cannot be scored: partial recovery is recorded, TopN stays 0.
func (r *Registry) sweepTopN(ctx context.Context, adpt provider.Adapter, opts Options) {
	ids, texts, _ := r.Bind(registryLabels(r))
	caps := adpt.Capabilities()
	for _, n := range topNLadder {
		if caps.MaxTopLogprobs > 0 && n > caps.MaxTopLogprobs*8 {
			break
		}
		res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
			Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
			Endpoint:            endpointKind(r.Endpoint),
			PromptText:          opts.PromptBase + opts.UserSuffix + opts.CloseTag + r.Delimiter,
			CandidateTokenIDs:   ids,
			CandidateTokenTexts: texts,
			TopK:                n,
			NoReasoning:         opts.NoReasoning,
			ReasoningField:      opts.ReasoningField,
			ReasoningValue:      opts.ReasoningValue,
		})
		if err != nil {
			// A rejected N reveals the server cap (parse "between 0 and X"
			// when present; otherwise the last working N is the cap).
			r.TopNCap = parseTopNCap(err.Error())
			if r.TopNCap == 0 {
				r.TopNCap = prevLadder(n)
			}
			break
		}
		found := 0
		for i := range texts {
			if _, ok := res.CandidateLogprobs[i]; ok {
				found++
			}
		}
		r.TopNRecovered = found
		if found == len(texts) {
			r.TopN = n
			if r.TopNCap == 0 {
				r.TopNCap = caps.MaxTopLogprobs
			}
			return
		}
	}
	if r.TopNCap == 0 && caps.MaxTopLogprobs > 0 {
		r.TopNCap = caps.MaxTopLogprobs
	}
}

func prevLadder(n int) int {
	prev := 0
	for _, v := range topNLadder {
		if v >= n {
			break
		}
		prev = v
	}
	return prev
}

// parseTopNCap extracts the cap from server rejections. Known phrasings:
//
//	Ollama:  "top_logprobs must be between 0 and 20"
//	Qwen:    "Range of top_logprobs should be [0, 5]"
func parseTopNCap(msg string) int {
	for _, marker := range []string{"between 0 and ", "must be between 0 and "} {
		if i := strings.Index(msg, marker); i >= 0 {
			if v := trailingInt(msg[i+len(marker):]); v > 0 {
				return v
			}
		}
	}
	if i := strings.Index(msg, "[0,"); i >= 0 {
		if v := trailingInt(msg[i+3:]); v > 0 {
			return v
		}
	}
	return 0
}

func trailingInt(s string) int {
	s = strings.TrimLeft(s, " \t")
	end := 0
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0
	}
	v, err := strconv.Atoi(s[:end])
	if err != nil {
		return 0
	}
	return v
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
	// The completion probe runs under the production reasoning control and
	// respects the discovered N ladder (tight-range servers reject the
	// engine default outright).
	topK := r.TopN
	if topK <= 0 {
		topK = adpt.Capabilities().MaxTopLogprobs
	}
	res, err := adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
		Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
		Endpoint:            endpointKind(r.Endpoint),
		PromptText:          opts.PromptBase + opts.UserSuffix + opts.CloseTag + r.Delimiter,
		CandidateTokenIDs:   ids,
		CandidateTokenTexts: texts,
		TopK:                topK,
		NoReasoning:         opts.NoReasoning,
		ReasoningField:      opts.ReasoningField,
		ReasoningValue:      opts.ReasoningValue,
	})
	if err != nil {
		if cap := parseTopNCap(err.Error()); cap > 0 && cap < topK {
			res, err = adpt.ScoreNextToken(ctx, provider.NextTokenScoreRequest{
				Model:               provider.ModelIdentity{Provider: adpt.ID(), Model: opts.BackendModel, Digest: opts.ModelDigest},
				Endpoint:            endpointKind(r.Endpoint),
				PromptText:          opts.PromptBase + opts.UserSuffix + opts.CloseTag + r.Delimiter,
				CandidateTokenIDs:   ids,
				CandidateTokenTexts: texts,
				TopK:                cap,
				NoReasoning:         opts.NoReasoning,
				ReasoningField:      opts.ReasoningField,
				ReasoningValue:      opts.ReasoningValue,
			})
		}
	}
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
	r.CloseTag = opts.CloseTag
	r.UserSuffix = opts.UserSuffix
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
	h.Write([]byte(r.CloseTag))
	h.Write([]byte{0})
	h.Write([]byte(r.UserSuffix))
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
