// Package bench runs labeled corpora through routes and computes the model
// benchmark metrics of TODO.md §12.3: accuracy, macro F1, NLL, Brier, ECE,
// candidate mass, latency percentiles, and label-order permutation variance.
package bench

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/eval"
	"hearim/internal/hearim/jev"
)

// Record is one labeled corpus line.
type Record struct {
	Model    string          `json:"model,omitempty"` // optional alias override
	State    json.RawMessage `json:"state"`
	Question json.RawMessage `json:"question"`
	Expected json.RawMessage `json:"expected"` // choice name | level index | bool
}

// LoadCorpus reads JSONL lines.
func LoadCorpus(r *bufio.Reader) ([]Record, error) {
	var out []Record
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var rec Record
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("bench: corpus line: %w", err)
		}
		out = append(out, rec)
	}
	return out, sc.Err()
}

// Metrics is the aggregated outcome (§12.3).
type Metrics struct {
	Records             int     `json:"records"`
	Accuracy            float64 `json:"accuracy"`
	MacroF1             float64 `json:"macro_f1"`
	NegLogLikelihood    float64 `json:"negative_log_likelihood"`
	Brier               float64 `json:"brier_score"`
	ECE                 float64 `json:"expected_calibration_error"`
	MedianCandidateMass float64 `json:"median_candidate_mass"`
	P50LatencyMs        float64 `json:"p50_latency_ms"`
	P95LatencyMs        float64 `json:"p95_latency_ms"`
	MeanPermutationVar  float64 `json:"mean_permutation_variance"`
	Errors              int     `json:"errors"`
}

// Run evaluates a corpus against one route. When permutations > 1, choice
// records are re-run with shuffled option orders to estimate label-order
// bias (§12.3); the restored distribution is mapped back to option meaning.
func Run(ctx context.Context, recs []Record, route eval.Route, compiler *compile.Compiler,
	cfg *config.Config, permutations int) (*Metrics, error) {

	ev := eval.New(compiler, cfg)
	m := &Metrics{Records: len(recs)}
	var latencies []float64
	var masses []float64
	var outcomes []outcomeT
	var permVars []float64

	for idx, rec := range recs {
		start := time.Now()
		model := rec.Model
		if model == "" {
			model = cfg.Gateway.DefaultModel
		}
		body, err := buildRequestBody(model, rec)
		if err != nil {
			m.Errors++
			continue
		}
		pr, err := jev.Validate(body)
		if err != nil {
			m.Errors++
			continue
		}
		plan, err := compiler.Compile(pr, route.BackendModel)
		if err != nil {
			m.Errors++
			continue
		}
		res, err := ev.EvaluateQuestion(ctx, plan, 0, route)
		if err != nil {
			m.Errors++
			continue
		}
		latencies = append(latencies, float64(time.Since(start).Microseconds())/1000)
		masses = append(masses, res.CandidateMass)

		switch ans := res.Answer.(type) {
		case *jev.ChoiceAnswer:
			var expected string
			_ = json.Unmarshal(rec.Expected, &expected)
			outcomes = append(outcomes, outcomeT{
				correct:  ans.Choice == expected,
				pred:     ans.Choice,
				expected: expected,
				conf:     maxProb(ans.Probabilities),
				correctP: ans.Probabilities[expected],
			})
			if permutations > 1 {
				v := permuteVariance(ctx, rec, model, route, compiler, ev, permutations)
				if v >= 0 {
					permVars = append(permVars, v)
				}
			}
		case *jev.NoulAnswer:
			var expected bool
			_ = json.Unmarshal(rec.Expected, &expected)
			// Binary metrics via the true-class probability.
			pTrue := ans.Noul
			if !expected {
				pTrue = 1 - pTrue
			}
			outcomes = append(outcomes, outcomeT{
				correct:  (ans.Noul >= 0.5) == expected,
				pred:     fmt.Sprintf("%v", ans.Noul >= 0.5),
				expected: fmt.Sprintf("%v", expected),
				conf:     math.Max(pTrue, 1-pTrue),
				correctP: pTrue,
			})
		case *jev.ScoreAnswer:
			var expected int
			_ = json.Unmarshal(rec.Expected, &expected)
			outcomes = append(outcomes, outcomeT{
				correct:  int(math.Round(ans.Score)) == expected,
				pred:     fmt.Sprintf("%d", int(math.Round(ans.Score))),
				expected: fmt.Sprintf("%d", expected),
				conf:     maxProb(ans.Probabilities),
				correctP: ans.Probabilities[fmt.Sprintf("%d", expected)],
			})
		}
		_ = idx
	}

	m.compute(outcomes, latencies, masses, permVars)
	return m, nil
}

// permuteVariance shuffles the listing order of choice criteria K times,
// evaluates each, reads back the probability of the reference option by
// name, and returns the variance (TODO.md §12.3 label-order bias).
func permuteVariance(ctx context.Context, rec Record, model string, route eval.Route,
	compiler *compile.Compiler, ev *eval.Evaluator, k int) float64 {
	var req struct {
		State     json.RawMessage          `json:"state"`
		Questions map[string]*jev.Question `json:"questions"`
	}
	body, err := buildRequestBody(model, rec)
	if err != nil {
		return -1
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return -1
	}
	q := req.Questions["q"]
	if q == nil || q.Type != jev.TypeChoice {
		return -1
	}
	order, descs := orderedCriteria(q.Criteria)
	if len(order) < 2 {
		return -1
	}

	// Reference option: the first listed one.
	ref := order[0]
	var probs []float64
	rng := newLocalRng(12345)
	for i := 0; i <= k; i++ {
		perm := append([]string{}, order...)
		if i > 0 {
			shuffle(perm, rng)
		}
		mixed := map[string]*string{}
		for _, name := range order {
			mixed[name] = descs[name]
		}
		_ = mixed
		crit := marshalOrdered(perm, descs)
		q2 := *q
		q2.Criteria = crit
		p, err := evalChoiceProb(ctx, req.State, &q2, model, route, compiler, ev, ref)
		if err != nil {
			continue
		}
		probs = append(probs, p)
	}
	if len(probs) < 2 {
		return -1
	}
	mean := 0.0
	for _, p := range probs {
		mean += p
	}
	mean /= float64(len(probs))
	v := 0.0
	for _, p := range probs {
		v += (p - mean) * (p - mean)
	}
	return v / float64(len(probs))
}

// orderedCriteria extracts criteria keys in JSON insertion order with their
// descriptions.
func orderedCriteria(raw json.RawMessage) ([]string, map[string]*string) {
	var m map[string]*string
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, nil
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	_, _ = dec.Token()
	var order []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, nil
		}
		key, ok := tok.(string)
		if !ok {
			return nil, nil
		}
		order = append(order, key)
		var skip any
		_ = dec.Decode(&skip)
	}
	return order, m
}

// marshalOrdered renders criteria JSON with a specific key insertion order.
func marshalOrdered(order []string, descs map[string]*string) json.RawMessage {
	var b strings.Builder
	b.WriteByte('{')
	for i, name := range order {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(name)
		b.Write(k)
		b.WriteByte(':')
		v, _ := json.Marshal(descs[name])
		b.Write(v)
	}
	b.WriteByte('}')
	return json.RawMessage(b.String())
}

// evalChoiceProb evaluates one permuted request and returns the probability
// of the reference option.
func evalChoiceProb(ctx context.Context, state json.RawMessage, q *jev.Question, model string,
	route eval.Route, compiler *compile.Compiler, ev *eval.Evaluator, ref string) (float64, error) {
	body, err := json.Marshal(map[string]any{
		"model":     model,
		"state":     json.RawMessage(state),
		"questions": map[string]any{"q": q},
	})
	if err != nil {
		return 0, err
	}
	pr, err := jev.Validate(body)
	if err != nil {
		return 0, err
	}
	plan, err := compiler.Compile(pr, route.BackendModel)
	if err != nil {
		return 0, err
	}
	res, err := ev.EvaluateQuestion(ctx, plan, 0, route)
	if err != nil {
		return 0, err
	}
	ca, ok := res.Answer.(*jev.ChoiceAnswer)
	if !ok {
		return 0, fmt.Errorf("bench: not a choice answer")
	}
	return ca.Probabilities[ref], nil
}

type outcomeT struct {
	correct  bool
	pred     string
	expected string
	conf     float64
	correctP float64
}

func (m *Metrics) compute(outcomes []outcomeT, latencies, masses, permVars []float64) {
	n := len(outcomes)
	if n == 0 {
		return
	}
	correct := 0
	for _, o := range outcomes {
		if o.correct {
			correct++
		}
	}
	m.Accuracy = float64(correct) / float64(n)

	// Macro F1 over classes present in expected.
	classes := map[string][]int{} // class -> [tp, fp, fn]
	for _, o := range outcomes {
		c := classes[o.expected]
		if c == nil {
			c = []int{0, 0, 0}
		}
		if o.correct {
			c[0]++
		} else {
			c[2]++ // fn for expected
			p := classes[o.pred]
			if p == nil {
				p = []int{0, 0, 0}
			}
			p[1]++ // fp for pred
			classes[o.pred] = p
		}
		classes[o.expected] = c
	}
	var f1sum float64
	for _, c := range classes {
		prec := 0.0
		if c[0]+c[1] > 0 {
			prec = float64(c[0]) / float64(c[0]+c[1])
		}
		rec := 0.0
		if c[0]+c[2] > 0 {
			rec = float64(c[0]) / float64(c[0]+c[2])
		}
		if prec+rec > 0 {
			f1sum += 2 * prec * rec / (prec + rec)
		}
	}
	m.MacroF1 = f1sum / float64(len(classes))

	var nll, brier float64
	for _, o := range outcomes {
		p := clamp01(o.correctP)
		nll += -math.Log(max(p, 1e-15))
		brier += (1 - p) * (1 - p)
	}
	m.NegLogLikelihood = nll / float64(n)
	m.Brier = brier / float64(n)

	// ECE with 10 confidence bins.
	bins := 10
	sums := make([]float64, bins)
	counts := make([]int, bins)
	accs := make([]float64, bins)
	for _, o := range outcomes {
		b := int(o.conf * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		counts[b]++
		sums[b] += o.conf
		if o.correct {
			accs[b]++
		}
	}
	var ece float64
	for b := 0; b < bins; b++ {
		if counts[b] == 0 {
			continue
		}
		avgConf := sums[b] / float64(counts[b])
		avgAcc := accs[b] / float64(counts[b])
		ece += float64(counts[b]) / float64(n) * math.Abs(avgAcc-avgConf)
	}
	m.ECE = ece

	m.P50LatencyMs = percentile(latencies, 50)
	m.P95LatencyMs = percentile(latencies, 95)
	if len(masses) > 0 {
		m.MedianCandidateMass = percentile(masses, 50)
	}
	if len(permVars) > 0 {
		sum := 0.0
		for _, v := range permVars {
			sum += v
		}
		m.MeanPermutationVar = sum / float64(len(permVars))
	}
}

func buildRequestBody(model string, rec Record) ([]byte, error) {
	body := map[string]any{
		"model":     model,
		"state":     json.RawMessage(rec.State),
		"questions": map[string]any{"q": json.RawMessage(rec.Question)},
	}
	return json.Marshal(body)
}

func maxProb(pm map[string]float64) float64 {
	max := 0.0
	for _, v := range pm {
		if v > max {
			max = v
		}
	}
	return max
}

func clamp01(p float64) float64 {
	if p < 0 {
		return 0
	}
	if p > 1 {
		return 1
	}
	return p
}

func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64{}, xs...)
	sort.Float64s(sorted)
	idx := p / 100 * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

type localRng struct{ state uint64 }

func newLocalRng(seed uint64) *localRng { return &localRng{state: seed} }

func (r *localRng) next() uint64 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return r.state
}

func shuffle(s []string, rng *localRng) {
	for i := len(s) - 1; i > 0; i-- {
		j := int(rng.next() % uint64(i+1))
		s[i], s[j] = s[j], s[i]
	}
}
