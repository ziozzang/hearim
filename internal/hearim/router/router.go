// Package router resolves public model aliases to (provider, engine,
// endpoint, model) route tuples and applies the cost/budget policy of
// TODO.md §9.
package router

import (
	"fmt"
	"sort"
	"sync"

	"hearim/internal/hearim/config"
	"hearim/internal/hearim/eval"
)

// Router maps public models to routes.
type Router struct {
	cfg      *config.Config
	mu       sync.RWMutex
	routes   map[string]eval.Route // alias -> resolved route
	autoRank []config.RouteCandidate
}

// New builds a router over configuration; adapters are attached later with
// Bind when registries are ready.
func New(cfg *config.Config) *Router {
	return &Router{cfg: cfg, routes: map[string]eval.Route{}}
}

// Resolve maps a public model name to a route. Aliases come from config;
// "provider:model" strings resolve directly; "policy:auto-v1" selects the
// auto cost policy of TODO.md §9.
func (r *Router) Resolve(publicModel string) (eval.Route, error) {
	if publicModel == "" {
		publicModel = r.cfg.Gateway.DefaultModel
	}
	r.mu.RLock()
	if route, ok := r.routes[publicModel]; ok {
		r.mu.RUnlock()
		return route, nil
	}
	r.mu.RUnlock()

	target, ok := r.cfg.ModelAliases[publicModel]
	if !ok {
		if isProviderModelRef(publicModel) {
			target = publicModel
		} else {
			return eval.Route{}, fmt.Errorf("router: unknown model %q", publicModel)
		}
	}

	if target == "policy:auto-v1" {
		target = r.autoTarget()
	}

	providerID, model := splitProviderModel(target)
	for _, route := range r.All() {
		if route.ProviderID == providerID && route.BackendModel == model {
			r.mu.Lock()
			r.routes[publicModel] = route
			r.mu.Unlock()
			return route, nil
		}
	}
	return eval.Route{}, fmt.Errorf("router: no route for %s (%s)", publicModel, target)
}

func isProviderModelRef(s string) bool {
	_, _, ok := splitProviderModelOK(s)
	return ok
}

func splitProviderModel(s string) (string, string) {
	p, m, _ := splitProviderModelOK(s)
	return p, m
}

func splitProviderModelOK(s string) (string, string, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] == ':' {
			// Skip URL-ish colons (none expected in aliases).
			return s[:i], s[i+1:], true
		}
	}
	return "", "", false
}

// AutoTarget resolves a routing policy reference ("policy:auto-v1") to a
// backend model name using the §9 cost policy.
func (r *Router) AutoTarget(policyRef string) (string, bool) {
	switch policyRef {
	case "policy:auto-v1", "auto":
		if m := r.autoTarget(); m != "" {
			return m, true
		}
	}
	return "", false
}

// autoTarget implements the §9 recommendation: at or above the break-even
// cache ratio prefer the high-cache-rate candidate; below, the cheapest
// quality-gated candidate.
func (r *Router) autoTarget() string {
	cands := r.gatedCandidates()
	if len(cands) == 0 {
		return ""
	}
	h := r.cfg.Router.ExpectedCacheRatio
	effective := func(c config.RouteCandidate) float64 {
		return c.InputPer1M*(1-h) + c.CachedInputPer1M*h
	}
	sort.Slice(cands, func(i, j int) bool {
		return effective(cands[i]) < effective(cands[j])
	})
	return cands[0].Model
}

func (r *Router) gatedCandidates() []config.RouteCandidate {
	gate := r.cfg.Router.RequireQualityGate == nil || *r.cfg.Router.RequireQualityGate
	var out []config.RouteCandidate
	for _, c := range r.cfg.Router.Candidates {
		if gate && !c.QualityGatePass {
			continue
		}
		out = append(out, c)
	}
	return out
}

// Bind registers a fully resolved route under an alias.
func (r *Router) Bind(alias string, route eval.Route) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.routes[alias] = route
}

// All returns the currently bound routes.
func (r *Router) All() []eval.Route {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]eval.Route, 0, len(r.routes))
	for _, v := range r.routes {
		out = append(out, v)
	}
	return out
}

// EffectiveInputRate computes the blended input rate at cache ratio h
// (TODO.md §9).
func EffectiveInputRate(c config.RouteCandidate, h float64) float64 {
	return c.InputPer1M*(1-h) + c.CachedInputPer1M*h
}

// BreakEvenCacheRatio solves for the cache ratio at which two candidates
// cost the same on input tokens:
//
//	a_in(1-h) + a_cached·h = b_in(1-h) + b_cached·h
//	=> h = (b_in − a_in) / ((b_in − b_cached) − (a_in − a_cached))
func BreakEvenCacheRatio(a, b config.RouteCandidate) (float64, error) {
	den := (b.InputPer1M - b.CachedInputPer1M) - (a.InputPer1M - a.CachedInputPer1M)
	if den == 0 {
		return 0, fmt.Errorf("router: candidates have identical cache sensitivity")
	}
	h := (b.InputPer1M - a.InputPer1M) / den
	if h < 0 || h > 1 {
		return 0, fmt.Errorf("router: no break-even in [0,1] (h=%g)", h)
	}
	return h, nil
}
