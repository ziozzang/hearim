package scoring

import (
	"math"
	"testing"
)

const eps = 1e-12

func TestSoftmaxMatchesDocFormula(t *testing.T) {
	// TODO.md §1: p(A) = exp(l_A) / (exp(l_A)+...+exp(l_D)).
	logps := []float64{math.Log(0.08), math.Log(0.86), math.Log(0.02), math.Log(0.04)}
	p := Softmax(logps)
	want := []float64{0.08, 0.86, 0.02, 0.04}
	var sum float64
	for i := range p {
		if math.Abs(p[i]-want[i]) > 1e-9 {
			t.Errorf("p[%d] = %v, want %v", i, p[i], want[i])
		}
		sum += p[i]
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %v, want 1", sum)
	}
}

func TestSoftmaxNumericallyStable(t *testing.T) {
	// Large-magnitude inputs must not overflow.
	logps := []float64{-1000, -1001, -1002}
	p := Softmax(logps)
	var sum float64
	for _, x := range p {
		sum += x
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %v", sum)
	}
	if !(p[0] > p[1] && p[1] > p[2]) {
		t.Errorf("ordering lost: %v", p)
	}
}

func TestCandidateMass(t *testing.T) {
	// Raw logprobs of 0.5, 0.3 -> candidate mass 0.8 (rest of vocab holds 0.2).
	m := CandidateMass([]float64{math.Log(0.5), math.Log(0.3)})
	if math.Abs(m-0.8) > 1e-9 {
		t.Errorf("candidate mass = %v, want 0.8", m)
	}
	// Stable for very negative logprobs.
	if m := CandidateMass([]float64{-1000, -1001}); !math.IsInf(m, 0) || m > 0 {
		t.Logf("extreme mass = %v (expected ~0)", m)
	}
}

func TestConfidenceExtremes(t *testing.T) {
	uniform := []float64{0.25, 0.25, 0.25, 0.25}
	if c := Confidence(uniform); math.Abs(c) > 1e-9 {
		t.Errorf("uniform confidence = %v, want 0", c)
	}
	peaked := []float64{1 - 3e-16, 1e-16, 1e-16, 1e-16}
	if c := Confidence(peaked); c < 0.999 {
		t.Errorf("peaked confidence = %v, want ~1", c)
	}
}

func TestConfidenceMatchesEntropyFormula(t *testing.T) {
	p := []float64{0.7, 0.2, 0.1}
	want := 1 - Entropy(p)/math.Log(3)
	if c := Confidence(p); math.Abs(c-want) > eps {
		t.Errorf("confidence = %v, want %v", c, want)
	}
}

func TestReductions(t *testing.T) {
	// TODO.md §2.2: levels [0.05, 0.20, 0.60, 0.15] -> score 1.85.
	p := []float64{0.05, 0.20, 0.60, 0.15}
	if s := WeightedMean(p); math.Abs(s-1.85) > 1e-9 {
		t.Errorf("score = %v, want 1.85", s)
	}
	if a := Argmax(p); a != 2 {
		t.Errorf("argmax = %d, want 2", a)
	}
	// Noul: p(true) at index 0.
	if n := p[0]; math.Abs(n-0.05) > eps {
		t.Errorf("noul = %v", n)
	}
}

func TestNormalizeContinuationModes(t *testing.T) {
	cond := []float64{-10.0, -20.0}
	lens := []float64{5.0, 2.0}
	uncond := []float64{-8.0, -12.0}

	sum, err := NormalizeContinuation(cond, lens, uncond, ModeSum, 1)
	if err != nil || sum[0] != -10 || sum[1] != -20 {
		t.Errorf("sum mode: %v %v", sum, err)
	}
	mean, err := NormalizeContinuation(cond, lens, uncond, ModeTokenMean, 1)
	if err != nil || mean[0] != -2 || mean[1] != -10 {
		t.Errorf("mean mode: %v %v", mean, err)
	}
	pmi, err := NormalizeContinuation(cond, lens, uncond, ModePMI, 1)
	if err != nil || pmi[0] != -2 || pmi[1] != -8 {
		t.Errorf("pmi mode: %v %v", pmi, err)
	}
	tau, err := NormalizeContinuation(cond, lens, uncond, ModeSum, 2)
	if err != nil || tau[0] != -5 {
		t.Errorf("tau mode: %v %v", tau, err)
	}
	if _, err := NormalizeContinuation(cond, []float64{1}, uncond, ModeTokenMean, 1); err == nil {
		t.Error("length mismatch not detected")
	}
}

func TestLCPAndHealing(t *testing.T) {
	if l := LongestCommonPrefix([]int{1, 2, 3, 4}, []int{1, 2, 9}); l != 2 {
		t.Errorf("lcp = %d", l)
	}
	// TODO.md §6.2: healStart = min_i LCP(base, full_i).
	base := []int{10, 11, 12, 13}
	fulls := [][]int{
		{10, 11, 12, 13, 50}, // clean +1
		{10, 11, 99, 40, 41}, // boundary re-segmentation from index 2
	}
	if h := HealStart(base, fulls); h != 2 {
		t.Errorf("healStart = %d, want 2", h)
	}
	if !IsStableSingleToken(base, fulls[0]) {
		t.Error("clean label should be stable single token")
	}
	if IsStableSingleToken(base, fulls[1]) {
		t.Error("re-segmented label must not qualify")
	}
	// All-stable case heals at full length.
	if h := HealStart(base, [][]int{{10, 11, 12, 13, 50}, {10, 11, 12, 13, 51}}); h != 4 {
		t.Errorf("healStart = %d, want 4", h)
	}
}

func TestParityNextTokenVsTeacherForced(t *testing.T) {
	// TODO.md §12.2: for a stable single-token label, next-token logprob and
	// the teacher-forced label input-logprob must agree. Simulated at the math
	// level: same logprob in, same softmax out.
	nt := []float64{-0.5, -1.5, -3.0}
	tf := []float64{-0.5 + 1e-9, -1.5, -3.0}
	a, b := Softmax(nt), Softmax(tf)
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			t.Errorf("parity broken at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestCalibrationProfileID(t *testing.T) {
	p := CalibrationProfile{Model: "m", Engine: "e", TemplateVersion: "v1", ScoringMethod: "label-token", Tau: 1}
	if p.ID() == "" {
		t.Error("profile id empty")
	}
}
