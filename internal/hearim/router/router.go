// Package router resolves public model aliases to (provider, engine,
// endpoint, model) route tuples and applies the cost/budget policy of
// TODO.md §9.
package router

import (
	"fmt"
	"sort"
	"strings"
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
// auto cost policy of TODO.md §9; chains (model_alias_chains) resolve to the
// first bound route in order.
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

	// Fallback chain: first bound route wins; healthy-route preference is
	// applied by the caller (readiness gates the binding itself).
	if chain, ok := r.cfg.ModelAliasChains[publicModel]; ok && len(chain) > 0 {
		var lastErr error
		for _, target := range chain {
			route, err := r.resolveTarget(target)
			if err != nil {
				lastErr = err
				continue
			}
			r.mu.Lock()
			r.routes[publicModel] = route
			r.mu.Unlock()
			return route, nil
		}
		if lastErr != nil {
			return eval.Route{}, fmt.Errorf("router: no route in chain for %s: %w", publicModel, lastErr)
		}
		return eval.Route{}, fmt.Errorf("router: empty chain resolution for %s", publicModel)
	}

	target, ok := r.cfg.ModelAliases[publicModel]
	if !ok {
		// "provider:model" only when the prefix names a configured provider;
		// bare backend model names (which often contain colons, like
		// "gemma4:31b") resolve to the provider that serves them.
		if pid, model, isRef := splitProviderModelOK(publicModel); isRef && r.hasProvider(pid) {
			return r.resolveTarget(pid + ":" + model)
		}
		return r.resolveTarget(":" + publicModel) // bare-name lookup
	}
	return r.resolveTarget(target)
}

// hasProvider reports whether a provider id is configured.
func (r *Router) hasProvider(id string) bool {
	for _, p := range r.cfg.Providers {
		if p.ID == id {
			return true
		}
	}
	return false
}

// resolveTarget resolves one "provider:model" or "policy:*" target to a
// bound route.
func (r *Router) resolveTarget(target string) (eval.Route, error) {
	if strings.HasPrefix(target, "policy:") {
		model, ok := r.AutoTarget(target)
		if !ok {
			return eval.Route{}, fmt.Errorf("router: policy %q found no gated candidate", target)
		}
		target = model
		// Fall through with a bare model name; pin via candidates first.
		if pid := r.providerPin(model); pid != "" {
			target = pid + ":" + model
		}
	}
	providerID, model := splitProviderModel(target)
	for _, route := range r.All() {
		if route.ProviderID == providerID && route.BackendModel == model {
			return route, nil
		}
	}
	if providerID == "" {
		// Bare model name: resolve to the provider serving it. Ambiguity
		// across providers is an error listing the candidates.
		var providers []string
		for _, p := range r.cfg.Providers {
			if p.Models.Find(model) != nil {
				providers = append(providers, p.ID)
			}
		}
		if len(providers) == 1 {
			return r.resolveTarget(providers[0] + ":" + model)
		}
		if len(providers) > 1 {
			return eval.Route{}, fmt.Errorf("router: model %q served by multiple providers %v; use provider:model", model, providers)
		}
	}
	return eval.Route{}, fmt.Errorf("router: no route for %s", target)
}

// providerPin finds a configured provider pin for a model.
func (r *Router) providerPin(model string) string {
	for _, c := range r.cfg.Router.Candidates {
		if c.Model == model && c.Provider != "" {
			return c.Provider
		}
	}
	for _, p := range r.cfg.Providers {
		if p.Models.Find(model) != nil {
			return p.ID
		}
	}
	if len(r.cfg.Providers) == 1 {
		return r.cfg.Providers[0].ID
	}
	return ""
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
