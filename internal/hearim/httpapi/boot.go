// Package httpapi assembles the hearim gateway: Jev-compatible
// POST /v1/systemone, OpenAI-compatible transparent proxy endpoints, health
// and readiness, auth, budget gating, and diagnostic headers (TODO.md §3).
package httpapi

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/eval"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/metrics"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/registry"
	"hearim/internal/hearim/router"
	"hearim/internal/hearim/schedule"
	"hearim/internal/hearim/usage"
)

// Server is the assembled gateway.
type Server struct {
	Cfg        *config.Config
	Adapters   map[string]provider.Adapter
	Router     *router.Router
	Evaluator  *eval.Evaluator
	Compiler   *compile.Compiler
	Schedulers map[string]*schedule.Scheduler
	Budget     *usage.Budget
	Ledger     *usage.Ledger
	Logger     *slog.Logger
	APIKeys    map[string]bool
	Metrics    *metrics.Registry

	routesMu    sync.RWMutex
	routes      []eval.Route    // unique provider:model routes
	ready       map[string]bool // "provider:model" -> readiness
	registryDir string
}

// New builds the server from configuration, including boot-time registry
// probing (TODO.md §6.1, §6.5). Providers whose probes fail stay unhealthy
// instead of killing the server.
func New(cfg *config.Config, logger *slog.Logger, registryDir string) (*Server, error) {
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{
		Cfg:         cfg,
		Adapters:    map[string]provider.Adapter{},
		Schedulers:  map[string]*schedule.Scheduler{},
		Budget:      usage.NewBudget(cfg.Budget),
		Ledger:      usage.NewLedger(),
		Logger:      logger,
		Metrics:     metrics.New(),
		ready:       map[string]bool{},
		registryDir: registryDir,
	}
	s.Compiler = compile.New(cfg.Compiler)
	s.Evaluator = eval.New(s.Compiler, cfg)
	s.Router = router.New(cfg)

	keys, err := cfg.LoadAPIKeys()
	if err != nil {
		return nil, err
	}
	s.APIKeys = map[string]bool{}
	for _, k := range keys {
		s.APIKeys[k] = true
	}

	for _, pc := range cfg.Providers {
		adpt, err := provider.NewAdapter(pc)
		if err != nil {
			return nil, err
		}
		s.Adapters[pc.ID] = adpt
		s.Schedulers[pc.ID] = schedule.New(pc.Concurrency, 10*time.Millisecond)
	}

	// Enumerate unique alias targets and bind routes. Chain entries
	// (model_alias_chains) contribute their targets so every hop has a
	// registry-backed route before it can serve as a fallback.
	targets := map[string][]string{} // target -> aliases
	for alias, target := range cfg.ModelAliases {
		if strings.HasPrefix(target, "policy:") {
			continue
		}
		targets[target] = append(targets[target], alias)
	}
	for alias, chain := range cfg.ModelAliasChains {
		for _, target := range chain {
			if strings.HasPrefix(target, "policy:") {
				continue
			}
			targets[target] = append(targets[target], alias)
		}
	}
	for target, aliases := range targets {
		route, err := s.buildRoute(context.Background(), target)
		if err != nil {
			logger.Warn("route not ready", "target", target, "err", err)
			continue
		}
		s.routesMu.Lock()
		s.routes = append(s.routes, route)
		s.ready[target] = routeReady(route)
		s.routesMu.Unlock()
		for _, alias := range aliases {
			s.Router.Bind(alias, route)
		}
		if cfg.Gateway.DefaultModel == target {
			// Direct provider:model default resolves through aliases too.
			s.Router.Bind(target, route)
		}
	}

	// Auto policy routes pick from gated candidates (§9).
	for alias, target := range cfg.ModelAliases {
		if !strings.HasPrefix(target, "policy:") {
			continue
		}
		model, ok := s.Router.AutoTarget(target)
		if !ok {
			logger.Warn("auto policy found no gated candidate", "alias", alias)
			continue
		}
		providerID := s.providerForModel(model)
		if providerID == "" {
			logger.Warn("auto policy model has no provider", "alias", alias, "model", model)
			continue
		}
		route, err := s.buildRoute(context.Background(), providerID+":"+model)
		if err != nil {
			logger.Warn("auto route not ready", "alias", alias, "err", err)
			continue
		}
		s.routesMu.Lock()
		s.routes = append(s.routes, route)
		s.ready[providerID+":"+model] = routeReady(route)
		s.routesMu.Unlock()
		s.Router.Bind(alias, route)
	}
	return s, nil
}

// providerForModel finds the provider configured to serve a model, using the
// candidate's provider pin or the provider's model list.
func (s *Server) providerForModel(model string) string {
	for _, c := range s.Cfg.Router.Candidates {
		if c.Model == model && c.Provider != "" {
			if _, ok := s.Adapters[c.Provider]; ok {
				return c.Provider
			}
		}
	}
	for _, p := range s.Cfg.Providers {
		if p.Models.Find(model) != nil {
			return p.ID
		}
	}
	if len(s.Cfg.Providers) == 1 {
		return s.Cfg.Providers[0].ID
	}
	return ""
}

// buildRoute resolves a "provider:model" target into a full route, probing
// the token label registry at boot.
func (s *Server) buildRoute(ctx context.Context, target string) (eval.Route, error) {
	providerID, model := splitTarget(target)
	adpt, ok := s.Adapters[providerID]
	if !ok {
		return eval.Route{}, fmt.Errorf("no provider %q", providerID)
	}
	pcfg := s.providerConfig(providerID)

	// Best-effort health probe populates the engine version, which feeds the
	// tokenizer revision of the registry key.
	if _, err := adpt.Health(ctx); err != nil {
		s.Logger.Debug("provider health probe failed", "provider", providerID, "err", err)
	}

	// Resolve the exact endpoint per §3.4/§3.10 ordering.
	route := eval.Route{
		Adapter:      adpt,
		ProviderID:   providerID,
		BackendModel: model,
		Engine:       adpt.Engine(),
		Endpoint:     firstPreferred(pcfg.EndpointPreference),
		CachePlan:    pcfg.PrefixCaching,
	}
	// Model entry: upstream type (chat|thinking|vision) and per-model
	// extra parameters.
	modelCfg := pcfg.Models.Find(model)
	route.ModelCfg = modelCfg

	// Endpoint resolution order: model force > provider force >
	// capability-based resolution (§3.4) > preference head. An explicit pin
	// targets gateways that expose only one surface; probes still verify
	// logprobs on it.
	if forced := provider.ForcedEndpoint(pcfg, modelCfg); forced != nil {
		route.Endpoint = *forced
		s.Logger.Warn("endpoint force-pinned by configuration; capability resolution skipped",
			"provider", providerID, "model", model, "endpoint", *forced)
	} else if e, err := provider.ResolveExactRoute(provider.CapabilitiesForModel(adpt, model), s.Cfg.BackendPolicy); err == nil {
		route.Endpoint = e.Kind
	}
	// §3.4 veto: a model whose reasoning cannot be fully disabled has no
	// exact chat route; it must use a verified completion-style endpoint.
	// An explicit endpoint pin overrides the veto (logged) — the operator
	// owns that decision.
	forced := provider.ForcedEndpoint(pcfg, modelCfg)
	if provider.ModelIsThinking(modelCfg) && route.Endpoint == config.EndpointChatCompletion && (forced == nil || *forced != config.EndpointChatCompletion) {
		alt := config.EndpointKind("")
		for _, e := range adpt.Capabilities().Endpoints {
			if (e.Kind == config.EndpointCompletions || e.Kind == config.EndpointNativeGenerate) &&
				e.NextTokenLogprobsVerified {
				alt = e.Kind
				break
			}
		}
		if alt == "" {
			return eval.Route{}, fmt.Errorf(
				"model %q is typed thinking/reasoning and provider %s has no verified non-chat logprob endpoint: no exact route (TODO.md §3.4)",
				model, providerID)
		}
		route.Endpoint = alt
		s.Logger.Warn("thinking model routed off chat endpoint", "model", model, "endpoint", alt)
	} else if provider.ModelIsThinking(modelCfg) && route.Endpoint == config.EndpointChatCompletion {
		s.Logger.Warn("thinking model pinned to chat endpoint by configuration; §3.4 veto overridden",
			"model", model)
	}

	// Pricing lookup by model name.
	for i, cand := range s.Cfg.Router.Candidates {
		if cand.Model == model {
			route.Pricing = &s.Cfg.Router.Candidates[i]
			break
		}
	}

	// Registry: load persisted or build via probes (§6.1 step 11, §6.5).
	// The probe prompt is compiled under the model's prompt override so
	// verified boundaries transfer to production prompts, and the override
	// hash lands in the registry identity.
	var promptOverride *config.PromptConfig
	if modelCfg != nil {
		promptOverride = modelCfg.Prompt
	}
	probePlan, err := s.Compiler.CompileWith(probeRequest(), model, promptOverride)
	if err != nil {
		return eval.Route{}, err
	}
	promptBase := probePlan.Questions[0].PromptPrefix + probePlan.Questions[0].PromptSuffix

	opts := registry.Options{
		ScoringProfile:      provider.ScoringProfileKey(adpt, model),
		BackendModel:        model,
		TokenizerRevision:   tokenizerRevisionOf(adpt),
		Endpoint:            string(route.Endpoint),
		TemplateVersion:     probePlan.TemplateVersion,
		PromptBase:          promptBase,
		DelimiterCandidates: s.Cfg.Compiler.DelimiterCandidates,
		Alphabets:           s.Cfg.Compiler.LabelAlphabets,
	}
	if modelCfg != nil && modelCfg.Thinking != nil {
		if modelCfg.Thinking.CloseTag != "" && !modelCfg.Thinking.WaitClose {
			opts.CloseTag = modelCfg.Thinking.CloseTag
		}
		opts.UserSuffix = modelCfg.Thinking.UserSuffix
	}
	// Probes must measure under the production reasoning control.
	if route.Endpoint == config.EndpointChatCompletion {
		opts.NoReasoning = true
	}
	if modelCfg != nil && modelCfg.Thinking != nil && modelCfg.Thinking.DisableField != "" {
		opts.ReasoningField = modelCfg.Thinking.DisableField
		opts.ReasoningValue = modelCfg.Thinking.DisableValue
	}
	regKey := registry.OptionsKey(opts)
	if s.registryDir != "" {
		if existing, err := registry.Load(s.registryDir, regKey); err == nil && existing != nil {
			route.Registry = existing
			return route, nil
		}
	}
	if !*s.Cfg.Compiler.BuildTokenRegistryOnStartup {
		return route, nil // registry attached later by probe
	}
	reg, err := registry.Build(ctx, adpt, opts)
	if err != nil {
		return eval.Route{}, fmt.Errorf("registry build: %w", err)
	}
	route.Registry = reg
	if s.registryDir != "" {
		if path, err := reg.Save(s.registryDir); err == nil {
			s.Logger.Info("token label registry persisted", "path", path, "model", model)
		}
	}
	return route, nil
}

// probeRequest is the fixed boot probe request (deterministic prompt).
func probeRequest() *jev.ParsedRequest {
	pr, err := jev.Validate([]byte(`{"model":"probe","state":"hearim token label registry probe","questions":{"probe":{"type":"noul","instructions":"Answer with the single best label for the state.","criteria":{"true":"the proposition holds","false":"the proposition does not hold"}}}}`))
	if err != nil {
		panic(err) // fixed literal; cannot fail
	}
	return pr
}

func (s *Server) providerConfig(id string) config.ProviderConfig {
	for _, p := range s.Cfg.Providers {
		if p.ID == id {
			return p
		}
	}
	return config.ProviderConfig{}
}

func splitTarget(t string) (string, string) {
	parts := strings.SplitN(t, ":", 2)
	if len(parts) != 2 {
		return t, ""
	}
	return parts[0], parts[1]
}

func firstPreferred(prefs []config.EndpointKind) config.EndpointKind {
	if len(prefs) > 0 {
		return prefs[0]
	}
	return config.EndpointCompletions
}

func tokenizerRevisionOf(adpt provider.Adapter) string {
	caps := adpt.Capabilities()
	return caps.EngineVersion
}

func routeReady(r eval.Route) bool {
	if r.Registry == nil {
		return false
	}
	ok, _ := r.Registry.Ready(2)
	return ok
}

// Routes returns bound routes.
func (s *Server) Routes() []eval.Route {
	s.routesMu.RLock()
	defer s.routesMu.RUnlock()
	out := make([]eval.Route, len(s.routes))
	copy(out, s.routes)
	return out
}

// Ready reports whether at least one default route is ready (§6.5: if no
// default route is ready the readiness probe must fail).
func (s *Server) Ready() (bool, string) {
	if _, err := s.Router.Resolve(""); err != nil {
		return false, fmt.Sprintf("default model unresolvable: %v", err)
	}
	s.routesMu.RLock()
	defer s.routesMu.RUnlock()
	for _, r := range s.routes {
		if routeReady(r) {
			return true, ""
		}
	}
	return false, "no route has a ready token label registry"
}

// Handler returns the root HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", s.handleSystemOne)
	mux.HandleFunc("POST /v1/completions", s.handleProxy("completions"))
	mux.HandleFunc("POST /v1/chat/completions", s.handleProxy("chat"))
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /v1/routes", s.handleRoutes)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	return s.withCommon(mux)
}

// Shutdown stops schedulers.
func (s *Server) Shutdown(ctx context.Context) {
	for id, sch := range s.Schedulers {
		if err := sch.Shutdown(ctx); err != nil {
			s.Logger.Warn("scheduler shutdown", "provider", id, "err", err)
		}
	}
}
