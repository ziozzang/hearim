// Package scoring implements the probability restoration mathematics of
// TODO.md §7: candidate conditional softmax, candidate mass, normalized
// entropy confidence, per-primitive reductions, calibration temperature, and
// exact-LCP token healing helpers.
package scoring

import (
	"fmt"
	"math"
)

// Softmax converts candidate log-probabilities into the conditional
// distribution over the declared candidate set (TODO.md §7.2):
//
//	p(i) = exp(l_i) / Σ_j exp(l_j)
//
// computed with the standard max-shift for numerical stability. The result
// sums to 1 within floating-point tolerance even when the inputs are raw
// (unnormalized) logprobs such as sequence log-likelihoods.
func Softmax(logprobs []float64) []float64 {
	if len(logprobs) == 0 {
		return nil
	}
	maxLogp := math.Inf(-1)
	for _, l := range logprobs {
		if l > maxLogp {
			maxLogp = l
		}
	}
	weights := make([]float64, len(logprobs))
	var z float64
	for i, l := range logprobs {
		w := math.Exp(l - maxLogp)
		weights[i] = w
		z += w
	}
	if z == 0 || math.IsInf(z, 1) || math.IsNaN(z) {
		// Degenerate inputs (all -Inf, overflow): return uniform to keep the
		// response shape well-formed; callers treat candidate_mass as 0.
		u := 1 / float64(len(logprobs))
		out := make([]float64, len(logprobs))
		for i := range out {
			out[i] = u
		}
		return out
	}
	for i := range weights {
		weights[i] /= z
	}
	return weights
}

// CandidateMass is Σ exp(l_i) over candidates in the raw vocabulary space
// (TODO.md §7.2). A low value means the model put most probability mass
// outside the declared candidates: prompt non-compliance or model mismatch.
// Inputs are true logprobs (<= 0); the result is in [0, len].
func CandidateMass(logprobs []float64) float64 {
	maxLogp := math.Inf(-1)
	for _, l := range logprobs {
		if l > maxLogp {
			maxLogp = l
		}
	}
	var sum float64
	for _, l := range logprobs {
		sum += math.Exp(l - maxLogp)
	}
	return sum * math.Exp(maxLogp)
}

// Entropy is H(p) = −Σ p_i ln p_i with 0·ln 0 = 0.
func Entropy(p []float64) float64 {
	var h float64
	for _, x := range p {
		if x > 0 {
			h -= x * math.Log(x)
		}
	}
	return h
}

// Confidence is the normalized-entropy confidence (TODO.md §7.5):
//
//	confidence = 1 − H(p) / ln(K)
//
// Near 0 for uniform distributions, near 1 when mass concentrates on one
// candidate. This is hearim's documented method (x-jev-confidence-method:
// normalized-entropy-v1) and is not guaranteed to equal TypeSafe Jev's
// internal confidence.
func Confidence(p []float64) float64 {
	if len(p) < 2 {
		return 1
	}
	lnk := math.Log(float64(len(p)))
	if lnk == 0 {
		return 1
	}
	c := 1 - Entropy(p)/lnk
	if c < 0 {
		c = 0
	}
	if c > 1 {
		c = 1
	}
	return c
}

// Argmax returns the first index of the maximum probability.
func Argmax(p []float64) int {
	best := 0
	for i := 1; i < len(p); i++ {
		if p[i] > p[best] {
			best = i
		}
	}
	return best
}

// WeightedMean computes Σ i·p_i — the Score reduction (TODO.md §7.4), where
// p indexes levels 0..K−1 from the lowest description upward.
func WeightedMean(p []float64) float64 {
	var sum float64
	for i, x := range p {
		sum += float64(i) * x
	}
	return sum
}

// CalibrationProfile identifies the calibration bucket of an evaluation:
// the same model under a different engine, quantization, template, or scoring
// mode is a different profile (TODO.md §16 point 8).
type CalibrationProfile struct {
	ProbabilitySpace string
	ProviderProfile  string
	Model            string
	Engine           string
	TemplateVersion  string
	ScoringMethod    string
	Tau              float64
}

// ID renders the profile identifier reported in x-jev-calibration-profile.
func (c CalibrationProfile) ID() string {
	tau := c.Tau
	if tau == 0 {
		tau = 1
	}
	id := fmt.Sprintf("%s|%s|%s|%s|tau=%g", c.Model, c.Engine, c.TemplateVersion, c.ScoringMethod, tau)
	if c.ProbabilitySpace != "" {
		id += "|space=" + c.ProbabilitySpace
	}
	if c.ProviderProfile != "" {
		id += "|provider-profile=" + c.ProviderProfile
	}
	return id
}

// ContinuationMode selects how a choice-text sequence score is normalized
// (TODO.md §7.3.1).
type ContinuationMode string

const (
	ModeSum       ContinuationMode = "sum"        // Σ_t log P(c_t | prefix, c_<t)
	ModeTokenMean ContinuationMode = "token-mean" // per-token average
	ModePMI       ContinuationMode = "pmi"        // conditional − unconditional
)

// NormalizeContinuation applies the mode then a calibration temperature and
// returns candidate logprobs suitable for Softmax. tau <= 0 is treated as 1.
func NormalizeContinuation(condScores []float64, condTokenLens []float64, uncondScores []float64, mode ContinuationMode, tau float64) ([]float64, error) {
	if len(condScores) == 0 || len(condScores) != len(condTokenLens) {
		return nil, fmt.Errorf("scoring: continuation inputs length mismatch")
	}
	if tau <= 0 {
		tau = 1
	}
	out := make([]float64, len(condScores))
	for i := range condScores {
		s := condScores[i]
		switch mode {
		case ModeSum:
			// raw sequence log-likelihood
		case ModeTokenMean:
			if condTokenLens[i] <= 0 {
				return nil, fmt.Errorf("scoring: token-mean requires positive token length")
			}
			s /= condTokenLens[i]
		case ModePMI:
			if i >= len(uncondScores) {
				return nil, fmt.Errorf("scoring: pmi requires unconditional scores")
			}
			s -= uncondScores[i]
		default:
			return nil, fmt.Errorf("scoring: unknown continuation mode %q", mode)
		}
		out[i] = s / tau
	}
	return out, nil
}

// LongestCommonPrefix returns the length of the common prefix of two token
// ID sequences.
func LongestCommonPrefix(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// HealStart computes the exact-LCP healing start position (TODO.md §6.2):
//
//	healStart = min_i LCP(baseIds, fullIds[i])
//
// Every candidate is re-scored from this position under its own full
// tokenization, so boundary re-segmentation is included in the score.
func HealStart(base []int, fulls [][]int) int {
	if len(fulls) == 0 {
		return 0
	}
	min := len(base)
	for _, f := range fulls {
		l := LongestCommonPrefix(base, f)
		if l < min {
			min = l
		}
	}
	return min
}

// IsStableSingleToken is the fast-path invariant (TODO.md §6.2):
// tokenize(prefix+label) must equal tokenize(prefix) plus exactly one token.
func IsStableSingleToken(base, full []int) bool {
	return len(full) == len(base)+1 && LongestCommonPrefix(base, full) == len(base)
}
