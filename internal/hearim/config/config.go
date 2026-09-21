// Package config defines hearim's YAML configuration schema, defaults, and
// validation. The schema mirrors TODO.md §14 ("recommended initial
// configuration") and the settings referenced throughout the design.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// EngineKind identifies an inference engine. See TODO.md §3.5.
type EngineKind string

const (
	EngineOllama  EngineKind = "ollama"
	EngineLlamaPP EngineKind = "llama.cpp"
	EngineVLLM    EngineKind = "vllm"
	EngineSGLang  EngineKind = "sglang"
	EngineGeneric EngineKind = "generic-openai"
)

// EndpointKind identifies an upstream endpoint surface. See TODO.md §3.3.
type EndpointKind string

const (
	EndpointCompletions    EndpointKind = "completions"
	EndpointChatCompletion EndpointKind = "chat_completions"
	EndpointNativeGenerate EndpointKind = "native_generate"
	EndpointNativeChat     EndpointKind = "native_chat"
)

// Layout selects the prompt compilation layout. See TODO.md §5.
type Layout string

const (
	LayoutStateMajor  Layout = "state-major"
	LayoutRubricMajor Layout = "rubric-major"
)

// ScoringStrategy names a candidate-scoring strategy. See TODO.md §3.11.
type ScoringStrategy string

const (
	ScoringSelectedTokenIDs  ScoringStrategy = "selected_token_ids"
	ScoringTopK              ScoringStrategy = "top_k"
	ScoringTeacherForced     ScoringStrategy = "teacher_forced_label"
	ScoringConstrainedVocab  ScoringStrategy = "constrained_vocab"
	ScoringChoiceSequenceSum ScoringStrategy = "choice_sequence_sum"
	ScoringSequenceMean      ScoringStrategy = "choice_sequence_mean"
	ScoringSequencePMI       ScoringStrategy = "choice_pmi"
)

// Config is the root configuration.
type Config struct {
	Gateway       GatewayConfig       `yaml:"gateway" json:"gateway"`
	Server        ServerConfig        `yaml:"server" json:"server"`
	Auth          AuthConfig          `yaml:"auth" json:"auth"`
	Log           LogConfig           `yaml:"log" json:"log"`
	Ollama        OllamaDefaults      `yaml:"ollama" json:"ollama"`
	Providers     []ProviderConfig    `yaml:"providers" json:"providers"`
	BackendPolicy BackendPolicyConfig `yaml:"backend_policy" json:"backend_policy"`
	Compiler      CompilerConfig      `yaml:"compiler" json:"compiler"`
	Scoring       ScoringConfig       `yaml:"scoring" json:"scoring"`
	Router        RouterConfig        `yaml:"router" json:"router"`
	Budget        BudgetConfig        `yaml:"budget" json:"budget"`
	ModelAliases  map[string]string   `yaml:"model_aliases" json:"model_aliases"`
	// ModelAliasChains map an alias to an ordered fallback list of
	// provider:model targets (or "policy:..." references); resolution walks
	// the chain and uses the first healthy route.
	ModelAliasChains map[string][]string `yaml:"model_alias_chains" json:"model_alias_chains"`
}

// GatewayConfig controls the public Jev-compatible surface. See TODO.md §3.1.
type GatewayConfig struct {
	PublicEndpoint               string `yaml:"public_endpoint" json:"public_endpoint"`
	Compatibility                string `yaml:"compatibility" json:"compatibility"`
	DefaultModel                 string `yaml:"default_model" json:"default_model"`
	StrictCandidateProbabilities *bool  `yaml:"strict_candidate_probabilities" json:"strict_candidate_probabilities"`
	ConfidenceMethod             string `yaml:"confidence_method" json:"confidence_method"`
	// PartialAnswers enables the non-standard partial mode where per-question
	// failures do not fail the whole request. TODO.md §10: off by default.
	PartialAnswers bool `yaml:"partial_answers" json:"partial_answers"`
}

type ServerConfig struct {
	Addr                string        `yaml:"addr" json:"addr"`
	ReadHeaderTimeout   time.Duration `yaml:"read_header_timeout" json:"read_header_timeout"`
	ReadTimeout         time.Duration `yaml:"read_timeout" json:"read_timeout"`
	WriteTimeout        time.Duration `yaml:"write_timeout" json:"write_timeout"`
	ShutdownGracePeriod time.Duration `yaml:"shutdown_grace_period" json:"shutdown_grace_period"`
}

// AuthConfig gates the public API. An empty APIKeys list disables auth.
type AuthConfig struct {
	APIKeys []string `yaml:"api_keys" json:"api_keys"`
	// APIKeyFile loads keys (one per line) from a file when set.
	APIKeyFile string `yaml:"api_key_file" json:"api_key_file"`
}

type LogConfig struct {
	Level  string `yaml:"level" json:"level"`   // debug|info|warn|error
	Format string `yaml:"format" json:"format"` // text|json
	// RawState enables logging raw state content. TODO.md §11: off by
	// default; hash and token counts only.
	RawState bool `yaml:"raw_state" json:"raw_state"`
}

// OllamaDefaults supplies defaults for providers with engine "ollama",
// matching the top-level `ollama:` block in TODO.md §14.
type OllamaDefaults struct {
	BaseURL         string        `yaml:"base_url" json:"base_url"`
	Concurrency     int           `yaml:"concurrency" json:"concurrency"`
	RequestTimeout  time.Duration `yaml:"request_timeout" json:"request_timeout"`
	MaxRetries      int           `yaml:"max_retries" json:"max_retries"`
	RateLimitPerMin int           `yaml:"rate_limit_per_min" json:"rate_limit_per_min"`
}

type ProviderConfig struct {
	ID      string     `yaml:"id" json:"id"`
	Engine  EngineKind `yaml:"engine" json:"engine"`
	BaseURL string     `yaml:"base_url" json:"base_url"`
	// BaseURLs is an endpoint pool for the same engine: requests rotate
	// across hosts and fail over on transport errors and retriable statuses.
	BaseURLs []string     `yaml:"base_urls" json:"base_urls"`
	APIKey   string       `yaml:"api_key" json:"api_key"`
	Models   ModelConfigs `yaml:"models" json:"models"`
	// ExtraParams are additional JSON fields merged into every upstream
	// request for this provider (model-level extra_params win per key).
	// hearim's correctness-critical fields (logprobs, max_tokens, ...) are
	// protected and cannot be overridden.
	ExtraParams        map[string]any `yaml:"extra_params" json:"extra_params"`
	EndpointPreference []EndpointKind `yaml:"endpoint_preference" json:"endpoint_preference"`
	Concurrency        int            `yaml:"concurrency" json:"concurrency"`
	RequestTimeout     time.Duration  `yaml:"request_timeout" json:"request_timeout"`
	MaxRetries         int            `yaml:"max_retries" json:"max_retries"`
	// CachePrompt requests prompt caching where the engine exposes it
	// (llama.cpp cache_prompt). Pointer so an explicit `false` survives.
	CachePrompt *bool `yaml:"cache_prompt" json:"cache_prompt"`
	// SelectedTokenField names the provider field for direct candidate token
	// ID scoring (vLLM logprob_token_ids, SGLang token_ids_logprob).
	SelectedTokenField string `yaml:"selected_token_field" json:"selected_token_field"`
	// PrefixCaching describes the engine's prefix cache; used for telemetry.
	PrefixCaching string `yaml:"prefix_caching" json:"prefix_caching"`
	// MaxTopLogprobs caps requested top_logprobs (backend limit, 0=probe).
	MaxTopLogprobs int `yaml:"max_top_logprobs" json:"max_top_logprobs"`
	// HealthPath overrides the health probe path.
	HealthPath string `yaml:"health_path" json:"health_path"`
}

// ModelConfig is one upstream model with optional metadata. Type declares
// the upstream LLM type and steers routing:
//
//	chat    - plain instruct model (default when unset)
//	thinking / reasoning - reasoning traces cannot be fully disabled, so
//	                      chat exact routes are vetoed for it (TODO.md §3.4)
//	vision  - accepts image inputs
type ModelConfig struct {
	Name        string         `yaml:"name" json:"name"`
	Type        string         `yaml:"type" json:"type,omitempty"`
	ExtraParams map[string]any `yaml:"extra_params" json:"extra_params,omitempty"`
}

// ModelConfigs accepts either a plain string list (backward compatible) or
// a list of model entries.
type ModelConfigs []ModelConfig

// UnmarshalYAML accepts scalars ("model-name") and mappings (name/type/
// extra_params) inside a models: sequence.
func (m *ModelConfigs) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind != yaml.SequenceNode {
		return fmt.Errorf("config: models must be a list")
	}
	out := make(ModelConfigs, 0, len(node.Content))
	for _, item := range node.Content {
		switch item.Kind {
		case yaml.ScalarNode:
			var name string
			if err := item.Decode(&name); err != nil {
				return err
			}
			out = append(out, ModelConfig{Name: name})
		case yaml.MappingNode:
			var mc ModelConfig
			if err := item.Decode(&mc); err != nil {
				return err
			}
			if mc.Name == "" {
				return fmt.Errorf("config: model entry requires a name")
			}
			switch mc.Type {
			case "", "chat", "thinking", "reasoning", "vision":
			default:
				return fmt.Errorf("config: model %q: unknown type %q (chat|thinking|vision)", mc.Name, mc.Type)
			}
			out = append(out, mc)
		default:
			return fmt.Errorf("config: models entries must be strings or mappings")
		}
	}
	*m = out
	return nil
}

// Names returns just the model names.
func (m ModelConfigs) Names() []string {
	out := make([]string, 0, len(m))
	for _, mc := range m {
		out = append(out, mc.Name)
	}
	return out
}

// Find returns the entry for a model name, or nil.
func (m ModelConfigs) Find(name string) *ModelConfig {
	for i := range m {
		if m[i].Name == name {
			return &m[i]
		}
	}
	return nil
}

type BackendPolicyConfig struct {
	EndpointPreference []EndpointKind `yaml:"endpoint_preference" json:"endpoint_preference"`
	// The three policy flags below default to true (TODO.md §14); YAML cannot
	// distinguish unset from false, so they are pointers.
	RequireNoReasoningForChat            *bool `yaml:"require_no_reasoning_for_chat" json:"require_no_reasoning_for_chat"`
	RejectHiddenReasoning                *bool `yaml:"reject_hidden_reasoning" json:"reject_hidden_reasoning"`
	RejectPromptOnlyReasoningSuppression *bool `yaml:"reject_prompt_only_reasoning_suppression" json:"reject_prompt_only_reasoning_suppression"`
}

type CompilerConfig struct {
	TemplateVersion             string     `yaml:"template_version" json:"template_version"`
	DefaultLayout               Layout     `yaml:"default_layout" json:"default_layout"`
	BuildTokenRegistryOnStartup *bool      `yaml:"build_token_registry_on_startup" json:"build_token_registry_on_startup"`
	LabelAlphabets              [][]string `yaml:"label_alphabets" json:"label_alphabets"`
	// MaxExactChoiceOptions caps Choice size for the single-token fast path;
	// 0 means capability-derived from the registry (max_exact_candidates).
	MaxExactChoiceOptions int     `yaml:"max_exact_choice_options" json:"max_exact_choice_options"`
	MaxTokens             int     `yaml:"max_tokens" json:"max_tokens"`
	Temperature           float64 `yaml:"temperature" json:"temperature"`
	TopP                  float64 `yaml:"top_p" json:"top_p"`
	// DelimiterCandidates are appended between the answer marker and the
	// label, tried in order until a stable single-token boundary is found.
	DelimiterCandidates []string `yaml:"delimiter_candidates" json:"delimiter_candidates"`
	// RetryTopLogprobsFactor scales the top_logprobs retry on missing
	// candidates (TODO.md §7.3 step 3: retry once with a larger N).
	RetryTopLogprobsFactor int `yaml:"retry_top_logprobs_factor" json:"retry_top_logprobs_factor"`
}

type ScoringConfig struct {
	Primary                   ScoringStrategy   `yaml:"primary" json:"primary"`
	Fallbacks                 []ScoringStrategy `yaml:"fallbacks" json:"fallbacks"`
	ChoiceTextMode            string            `yaml:"choice_text_mode" json:"choice_text_mode"` // disabled|fallback|preferred
	ContinuationNormalization string            `yaml:"continuation_normalization" json:"continuation_normalization"`
	TokenHealing              string            `yaml:"token_healing" json:"token_healing"` // exact_lcp
	CalibrationScope          string            `yaml:"calibration_scope" json:"calibration_scope"`
	// CalibrationTau sets per-scoring-mode temperature tau (p_i =
	// softmax(score_i / tau)). Modes not listed default to 1.0.
	CalibrationTau map[string]float64 `yaml:"calibration_tau" json:"calibration_tau"`
	// MinCandidateMass is the soft floor on summed candidate probability;
	// below this the evaluation is treated as a prompt-compliance failure.
	MinCandidateMass float64 `yaml:"min_candidate_mass" json:"min_candidate_mass"`
}

// RouteCandidate is a routable backend model with pricing (USD per 1M tokens).
type RouteCandidate struct {
	Model            string  `yaml:"model" json:"model"`
	Provider         string  `yaml:"provider" json:"provider"` // optional provider pin
	InputPer1M       float64 `yaml:"input_per_1m" json:"input_per_1m"`
	CachedInputPer1M float64 `yaml:"cached_input_per_1m" json:"cached_input_per_1m"`
	OutputPer1M      float64 `yaml:"output_per_1m" json:"output_per_1m"`
	QualityGatePass  bool    `yaml:"quality_gate_pass" json:"quality_gate_pass"`
}

type RouterConfig struct {
	Candidates []RouteCandidate `yaml:"candidates" json:"candidates"`
	// CacheRatioBreakEven is the cached-input fraction at which the
	// high-cache-rate candidate becomes cheaper on input cost
	// (TODO.md §9: 0.1754 for gemma4 vs deepseek-flash).
	CacheRatioBreakEven float64 `yaml:"cache_ratio_break_even_gemma_vs_deepseek" json:"cache_ratio_break_even_gemma_vs_deepseek"`
	// ExpectedCacheRatio feeds the auto router when no live estimate exists.
	ExpectedCacheRatio float64 `yaml:"expected_cache_ratio" json:"expected_cache_ratio"`
	RequireQualityGate *bool   `yaml:"require_quality_gate" json:"require_quality_gate"`
	// Auto enables cache-ratio-based model selection for the "auto" alias.
	Auto bool `yaml:"auto" json:"auto"`
}

type BudgetConfig struct {
	IncludedUSD float64 `yaml:"included_usd" json:"included_usd"`
	WarnAt      float64 `yaml:"warn_at" json:"warn_at"`
	ThrottleAt  float64 `yaml:"throttle_at" json:"throttle_at"`
	HardStopAt  float64 `yaml:"hard_stop_at" json:"hard_stop_at"`
}

// Load reads, expands, validates, and defaults a configuration file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read %s: %w", path, err)
	}
	return Parse(raw)
}

// Parse parses configuration bytes.
func Parse(raw []byte) (*Config, error) {
	expanded := expandEnv(string(raw))
	dec := yaml.NewDecoder(strings.NewReader(expanded))
	dec.KnownFields(true)
	var c Config
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("config: parse: %w", err)
	}
	c.applyDefaults()
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// expandEnv replaces ${VAR} and ${VAR:-default} in the YAML text. Secrets stay
// out of config files without requiring a full templating language.
func expandEnv(s string) string {
	var b strings.Builder
	for {
		i := strings.Index(s, "${")
		if i < 0 {
			b.WriteString(s)
			return b.String()
		}
		b.WriteString(s[:i])
		rest := s[i+2:]
		j := strings.Index(rest, "}")
		if j < 0 {
			b.WriteString(s[i:])
			return b.String()
		}
		expr := rest[:j]
		s = rest[j+1:]
		name, def, hasDef := strings.Cut(expr, ":-")
		if v, ok := os.LookupEnv(name); ok && v != "" {
			b.WriteString(v)
		} else if hasDef {
			b.WriteString(def)
		}
	}
}

func boolPtr(b bool) *bool { return &b }

func (c *Config) applyDefaults() {
	if c.Gateway.PublicEndpoint == "" {
		c.Gateway.PublicEndpoint = "/v1/systemone"
	}
	if c.Gateway.Compatibility == "" {
		c.Gateway.Compatibility = "typesafe-systemone-v1"
	}
	if c.Gateway.ConfidenceMethod == "" {
		c.Gateway.ConfidenceMethod = "normalized-entropy-v1"
	}
	if c.Gateway.StrictCandidateProbabilities == nil {
		c.Gateway.StrictCandidateProbabilities = boolPtr(true)
	}
	if c.Server.Addr == "" {
		c.Server.Addr = ":8080"
	}
	if c.Server.ReadHeaderTimeout == 0 {
		c.Server.ReadHeaderTimeout = 10 * time.Second
	}
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 60 * time.Second
	}
	if c.Server.WriteTimeout == 0 {
		c.Server.WriteTimeout = 120 * time.Second
	}
	if c.Server.ShutdownGracePeriod == 0 {
		c.Server.ShutdownGracePeriod = 15 * time.Second
	}
	if c.Log.Level == "" {
		c.Log.Level = "info"
	}
	if c.Log.Format == "" {
		c.Log.Format = "text"
	}
	if c.Ollama.Concurrency == 0 {
		c.Ollama.Concurrency = 3 // Ollama Pro concurrent request limit
	}
	if c.Ollama.RequestTimeout == 0 {
		c.Ollama.RequestTimeout = 30 * time.Second
	}
	if c.Ollama.MaxRetries == 0 {
		c.Ollama.MaxRetries = 1
	}
	if len(c.Providers) == 0 {
		c.Providers = []ProviderConfig{{
			ID:      "ollama-cloud",
			Engine:  EngineOllama,
			BaseURL: "https://ollama.com/v1",
		}}
	}
	for i := range c.Providers {
		p := &c.Providers[i]
		if p.Concurrency == 0 {
			if p.Engine == EngineOllama {
				p.Concurrency = c.Ollama.Concurrency
			} else {
				p.Concurrency = 8
			}
		}
		if p.RequestTimeout == 0 {
			if p.Engine == EngineOllama {
				p.RequestTimeout = c.Ollama.RequestTimeout
			} else {
				p.RequestTimeout = 60 * time.Second
			}
		}
		if p.MaxRetries == 0 {
			if p.Engine == EngineOllama {
				p.MaxRetries = c.Ollama.MaxRetries
			} else {
				p.MaxRetries = 1
			}
		}
		if p.BaseURL == "" && p.Engine == EngineOllama {
			p.BaseURL = c.Ollama.BaseURL
		}
		if len(p.EndpointPreference) == 0 {
			p.EndpointPreference = defaultEndpointPreference(p.Engine)
		}
		if p.PrefixCaching == "" {
			p.PrefixCaching = defaultPrefixCaching(p.Engine)
		}
		if p.SelectedTokenField == "" {
			p.SelectedTokenField = defaultSelectedTokenField(p.Engine)
		}
	}
	if len(c.BackendPolicy.EndpointPreference) == 0 {
		c.BackendPolicy.EndpointPreference = []EndpointKind{
			EndpointCompletions, EndpointNativeGenerate, EndpointChatCompletion,
		}
	}
	c.BackendPolicy.RequireNoReasoningForChat = defaultTrue(c.BackendPolicy.RequireNoReasoningForChat)
	c.BackendPolicy.RejectHiddenReasoning = defaultTrue(c.BackendPolicy.RejectHiddenReasoning)
	c.BackendPolicy.RejectPromptOnlyReasoningSuppression = defaultTrue(c.BackendPolicy.RejectPromptOnlyReasoningSuppression)
	if c.Compiler.TemplateVersion == "" {
		c.Compiler.TemplateVersion = "systemone-v1"
	}
	if c.Compiler.DefaultLayout == "" {
		c.Compiler.DefaultLayout = LayoutStateMajor
	}
	c.Compiler.BuildTokenRegistryOnStartup = defaultTrue(c.Compiler.BuildTokenRegistryOnStartup)
	if len(c.Compiler.LabelAlphabets) == 0 {
		c.Compiler.LabelAlphabets = [][]string{
			{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
			{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
				"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"},
		}
	}
	if c.Compiler.MaxTokens == 0 {
		c.Compiler.MaxTokens = 1
	}
	if c.Compiler.Temperature == 0 {
		c.Compiler.Temperature = 1
	}
	if c.Compiler.TopP == 0 {
		c.Compiler.TopP = 1
	}
	if len(c.Compiler.DelimiterCandidates) == 0 {
		c.Compiler.DelimiterCandidates = []string{"\n", " ", ":"}
	}
	if c.Compiler.RetryTopLogprobsFactor == 0 {
		c.Compiler.RetryTopLogprobsFactor = 4
	}
	if c.Scoring.Primary == "" {
		c.Scoring.Primary = ScoringSelectedTokenIDs
	}
	if len(c.Scoring.Fallbacks) == 0 {
		c.Scoring.Fallbacks = []ScoringStrategy{
			ScoringTopK, ScoringTeacherForced, ScoringConstrainedVocab,
		}
	}
	if c.Scoring.ChoiceTextMode == "" {
		c.Scoring.ChoiceTextMode = "disabled"
	}
	if c.Scoring.ContinuationNormalization == "" {
		c.Scoring.ContinuationNormalization = string(ScoringSequencePMI)
	}
	if c.Scoring.TokenHealing == "" {
		c.Scoring.TokenHealing = "exact_lcp"
	}
	if c.Scoring.CalibrationScope == "" {
		c.Scoring.CalibrationScope = "model-engine-template-mode"
	}
	if c.Scoring.CalibrationTau == nil {
		c.Scoring.CalibrationTau = map[string]float64{}
	}
	if c.Scoring.MinCandidateMass == 0 {
		c.Scoring.MinCandidateMass = 0.01
	}
	if c.Router.CacheRatioBreakEven == 0 {
		c.Router.CacheRatioBreakEven = 0.1754
	}
	if c.Router.ExpectedCacheRatio == 0 {
		c.Router.ExpectedCacheRatio = 0.20
	}
	c.Router.RequireQualityGate = defaultTrue(c.Router.RequireQualityGate)
	if c.Budget.IncludedUSD == 0 {
		c.Budget.IncludedUSD = 60
	}
	if c.Budget.WarnAt == 0 {
		c.Budget.WarnAt = 0.70
	}
	if c.Budget.ThrottleAt == 0 {
		c.Budget.ThrottleAt = 0.90
	}
	if c.Budget.HardStopAt == 0 {
		c.Budget.HardStopAt = 1.00
	}
}

func defaultTrue(v *bool) *bool {
	if v == nil {
		return boolPtr(true)
	}
	return v
}

func defaultEndpointPreference(e EngineKind) []EndpointKind {
	switch e {
	case EngineLlamaPP:
		return []EndpointKind{EndpointNativeGenerate, EndpointCompletions}
	case EngineSGLang:
		return []EndpointKind{EndpointNativeGenerate, EndpointCompletions, EndpointChatCompletion}
	case EngineVLLM, EngineOllama:
		return []EndpointKind{EndpointCompletions, EndpointChatCompletion}
	default:
		return []EndpointKind{EndpointCompletions, EndpointChatCompletion}
	}
}

func defaultPrefixCaching(e EngineKind) string {
	switch e {
	case EngineOllama:
		return "automatic"
	case EngineLlamaPP:
		return "request-flag"
	case EngineVLLM:
		return "automatic"
	case EngineSGLang:
		return "radix"
	default:
		return "unknown"
	}
}

func defaultSelectedTokenField(e EngineKind) string {
	switch e {
	case EngineVLLM:
		return "logprob_token_ids"
	case EngineSGLang:
		return "token_ids_logprob"
	default:
		return ""
	}
}

// Validate checks cross-field constraints.
func (c *Config) Validate() error {
	if c.Gateway.DefaultModel == "" && len(c.ModelAliases) == 0 {
		return fmt.Errorf("config: gateway.default_model or model_aliases must be set")
	}
	if c.Gateway.DefaultModel != "" {
		if _, ok := c.ModelAliases[c.Gateway.DefaultModel]; !ok && !strings.Contains(c.Gateway.DefaultModel, ":") {
			return fmt.Errorf("config: gateway.default_model %q is not a model alias or provider:model route", c.Gateway.DefaultModel)
		}
	}
	seen := map[string]bool{}
	for _, p := range c.Providers {
		if p.ID == "" {
			return fmt.Errorf("config: provider without id")
		}
		if seen[p.ID] {
			return fmt.Errorf("config: duplicate provider id %q", p.ID)
		}
		seen[p.ID] = true
		switch p.Engine {
		case EngineOllama, EngineLlamaPP, EngineVLLM, EngineSGLang, EngineGeneric:
		default:
			return fmt.Errorf("config: provider %q: unknown engine %q", p.ID, p.Engine)
		}
		if p.BaseURL == "" && len(p.BaseURLs) == 0 {
			return fmt.Errorf("config: provider %q: base_url or base_urls is required", p.ID)
		}
		for _, u := range p.BaseURLs {
			if u == "" {
				return fmt.Errorf("config: provider %q: empty entry in base_urls", p.ID)
			}
		}
		if p.Concurrency < 1 {
			return fmt.Errorf("config: provider %q: concurrency must be >= 1", p.ID)
		}
	}
	for alias, target := range c.ModelAliases {
		if !strings.Contains(target, ":") {
			return fmt.Errorf("config: model_alias %q -> %q must use provider:model syntax", alias, target)
		}
		if strings.HasPrefix(target, "policy:") {
			continue // routing policy reference, resolved by the router
		}
		parts := strings.SplitN(target, ":", 2)
		if !seen[parts[0]] {
			return fmt.Errorf("config: model_alias %q references unknown provider %q", alias, parts[0])
		}
	}
	for alias, chain := range c.ModelAliasChains {
		if len(chain) == 0 {
			return fmt.Errorf("config: model_alias_chains %q is empty", alias)
		}
		for _, target := range chain {
			if strings.HasPrefix(target, "policy:") {
				continue
			}
			if !strings.Contains(target, ":") {
				return fmt.Errorf("config: model_alias_chains %q entry %q must use provider:model syntax", alias, target)
			}
			parts := strings.SplitN(target, ":", 2)
			if !seen[parts[0]] {
				return fmt.Errorf("config: model_alias_chains %q references unknown provider %q", alias, parts[0])
			}
		}
	}
	if c.Compiler.Temperature <= 0 || c.Compiler.TopP <= 0 || c.Compiler.TopP > 1 {
		return fmt.Errorf("config: compiler temperature/top_p must be > 0 and top_p <= 1 (got %v/%v)", c.Compiler.Temperature, c.Compiler.TopP)
	}
	if !c.Budget.asTiers().valid() {
		return fmt.Errorf("config: budget tiers must satisfy 0 < warn <= throttle <= hard_stop <= 1")
	}
	switch c.Scoring.ChoiceTextMode {
	case "disabled", "fallback", "preferred":
	default:
		return fmt.Errorf("config: scoring.choice_text_mode must be disabled|fallback|preferred")
	}
	return nil
}

type budgetTiers struct{ warn, throttle, stop float64 }

func (b *BudgetConfig) asTiers() budgetTiers {
	return budgetTiers{b.WarnAt, b.ThrottleAt, b.HardStopAt}
}

func (t budgetTiers) valid() bool {
	return t.warn > 0 && t.warn <= t.throttle && t.throttle <= t.stop && t.stop <= 1.0001
}

// StrictMode reports whether strict candidate probability recovery is on.
func (c *Config) StrictMode() bool { return *c.Gateway.StrictCandidateProbabilities }

// LoadAPIKeys resolves configured API keys, including the key file.
func (c *Config) LoadAPIKeys() ([]string, error) {
	keys := append([]string{}, c.Auth.APIKeys...)
	if c.Auth.APIKeyFile != "" {
		data, err := os.ReadFile(c.Auth.APIKeyFile)
		if err != nil {
			return nil, fmt.Errorf("config: read api_key_file: %w", err)
		}
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line != "" && !strings.HasPrefix(line, "#") {
				keys = append(keys, line)
			}
		}
	}
	return keys, nil
}

// ParseDurationEnv is a helper for CLI flags like "30s".
func ParseDurationEnv(s string) (time.Duration, error) {
	if d, err := time.ParseDuration(s); err == nil {
		return d, nil
	}
	if ms, err := strconv.Atoi(s); err == nil {
		return time.Duration(ms) * time.Millisecond, nil
	}
	return 0, fmt.Errorf("invalid duration %q", s)
}
