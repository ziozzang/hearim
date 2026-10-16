package bench

import (
	"fmt"
	"math"
	"sort"

	"hearim/internal/hearim/jev"
)

// Agreement compares two routes' answers on the same corpus — typically a
// hearim token-scoring route against a real Jev reference (engine "jev").
// It needs no labels: it measures how close the distributions are.
type Agreement struct {
	Records  int `json:"records"`
	Compared int `json:"compared"`
	// Skipped counts records where either side failed or the answer types
	// differ.
	Skipped int `json:"skipped"`
	// Top1Agreement: same argmax (choice/score) or same side of 0.5 (noul).
	Top1Agreement float64 `json:"top1_agreement"`
	// MeanTVD / MaxTVD: total variation distance, ½·Σ|p−q| over options
	// (noul compares {p, 1−p}). 0 = identical, 1 = disjoint.
	MeanTVD float64 `json:"mean_total_variation"`
	MaxTVD  float64 `json:"max_total_variation"`
	// MeanAbsScoreDiff covers score questions only.
	MeanAbsScoreDiff float64                   `json:"mean_abs_score_diff,omitempty"`
	ByType           map[string]*TypeAgreement `json:"by_type"`
	// Worst lists the largest disagreements (by TVD), most severe first.
	Worst []Disagreement `json:"worst,omitempty"`
}

// TypeAgreement is Agreement restricted to one question type.
type TypeAgreement struct {
	Compared      int     `json:"compared"`
	Top1Agreement float64 `json:"top1_agreement"`
	MeanTVD       float64 `json:"mean_total_variation"`
}

// Disagreement is one record where the routes diverge.
type Disagreement struct {
	Index     int     `json:"index"`
	Type      string  `json:"type"`
	Model     string  `json:"model_top1"`
	Reference string  `json:"reference_top1"`
	TVD       float64 `json:"total_variation"`
}

// maxWorst bounds the Worst list.
const maxWorst = 10

// Compare computes the agreement between index-aligned answer slices as
// returned by RunDetailed.
func Compare(model, reference []any) *Agreement {
	ag := &Agreement{Records: len(model), ByType: map[string]*TypeAgreement{}}
	n := min(len(model), len(reference))
	var top1, tvdSum, scoreDiffSum float64
	var scoreN int
	var all []Disagreement
	for i := 0; i < n; i++ {
		qt, pa, ta, sa, okA := distribution(model[i])
		qtB, pb, tb, sb, okB := distribution(reference[i])
		if !okA || !okB || qt != qtB {
			ag.Skipped++
			continue
		}
		ag.Compared++
		tvd := totalVariation(pa, pb)
		same := ta == tb
		if same {
			top1++
		}
		tvdSum += tvd
		ag.MaxTVD = math.Max(ag.MaxTVD, tvd)
		if qt == string(jev.TypeScore) {
			scoreDiffSum += math.Abs(sa - sb)
			scoreN++
		}
		bt := ag.ByType[qt]
		if bt == nil {
			bt = &TypeAgreement{}
			ag.ByType[qt] = bt
		}
		bt.Compared++
		if same {
			bt.Top1Agreement++
		}
		bt.MeanTVD += tvd
		all = append(all, Disagreement{Index: i, Type: qt, Model: ta, Reference: tb, TVD: tvd})
	}
	ag.Skipped += len(model) - n
	if ag.Compared > 0 {
		ag.Top1Agreement = top1 / float64(ag.Compared)
		ag.MeanTVD = tvdSum / float64(ag.Compared)
	}
	if scoreN > 0 {
		ag.MeanAbsScoreDiff = scoreDiffSum / float64(scoreN)
	}
	for _, bt := range ag.ByType {
		bt.Top1Agreement /= float64(bt.Compared)
		bt.MeanTVD /= float64(bt.Compared)
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].TVD > all[j].TVD })
	for _, d := range all {
		if len(ag.Worst) == maxWorst || d.TVD < 1e-9 {
			break
		}
		ag.Worst = append(ag.Worst, d)
	}
	return ag
}

// distribution normalizes an answer to (type, distribution, top1, score).
func distribution(ans any) (string, map[string]float64, string, float64, bool) {
	switch a := ans.(type) {
	case *jev.ChoiceAnswer:
		return string(jev.TypeChoice), a.Probabilities, argmax(a.Probabilities), 0, len(a.Probabilities) > 0
	case *jev.ScoreAnswer:
		return string(jev.TypeScore), a.Probabilities, argmax(a.Probabilities), a.Score, len(a.Probabilities) > 0
	case *jev.NoulAnswer:
		p := clamp01(a.Noul)
		return string(jev.TypeNoul), map[string]float64{"true": p, "false": 1 - p},
			fmt.Sprintf("%v", p >= 0.5), 0, true
	default:
		return "", nil, "", 0, false
	}
}

func totalVariation(p, q map[string]float64) float64 {
	sum := 0.0
	for k, v := range p {
		sum += math.Abs(v - q[k])
	}
	for k, v := range q {
		if _, ok := p[k]; !ok {
			sum += math.Abs(v)
		}
	}
	return sum / 2
}

// argmax returns the highest-probability key; ties break by key order so
// the result is deterministic.
func argmax(pm map[string]float64) string {
	keys := make([]string, 0, len(pm))
	for k := range pm {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best, bestP := "", -1.0
	for _, k := range keys {
		if pm[k] > bestP {
			best, bestP = k, pm[k]
		}
	}
	return best
}
