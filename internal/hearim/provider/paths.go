package provider

import (
	"hearim/internal/hearim/config"
)

// Logical path names shared by the adapters.
const (
	PathCompletions     = "completions"
	PathChatCompletions = "chat_completions"
	PathCompletion      = "completion" // llama.cpp native
	PathGenerate        = "generate"   // SGLang native
	PathTokenize        = "tokenize"
	PathHealth          = "health"
)

// defaultPaths are the engine-specific URL defaults. OpenAI-compatible
// surfaces hang off /v1; llama.cpp uses /completion; SGLang uses /generate;
// Ollama native uses /api/*. A trailing "/v1" in the configured base URL is
// stripped during joining (joinURL), so these are server-root-relative.
func defaultPaths(engine config.EngineKind) map[string]string {
	switch engine {
	case config.EngineLlamaPP:
		return map[string]string{
			PathCompletions:     "/v1/completions",
			PathChatCompletions: "/v1/chat/completions",
			PathCompletion:      "/completion",
			PathTokenize:        "/tokenize",
			PathHealth:          "/health",
		}
	case config.EngineSGLang:
		return map[string]string{
			PathCompletions:     "/v1/completions",
			PathChatCompletions: "/v1/chat/completions",
			PathGenerate:        "/generate",
			PathTokenize:        "/tokenize",
			PathHealth:          "/get_server_info",
		}
	case config.EngineOllama:
		return map[string]string{
			PathCompletions:     "/v1/completions",
			PathChatCompletions: "/v1/chat/completions",
			PathGenerate:        "/api/generate",
			PathTokenize:        "/api/tokenize",
			PathHealth:          "/api/version",
		}
	default: // vLLM, generic OpenAI
		return map[string]string{
			PathCompletions:     "/v1/completions",
			PathChatCompletions: "/v1/chat/completions",
			PathTokenize:        "/tokenize",
			PathHealth:          "/version",
		}
	}
}

// pathFor resolves a logical path against the provider's overrides.
func pathFor(cfg config.ProviderConfig, which string) string {
	if cfg.Paths != nil {
		var v string
		switch which {
		case PathCompletions:
			v = cfg.Paths.Completions
		case PathChatCompletions:
			v = cfg.Paths.ChatCompletions
		case PathCompletion:
			v = cfg.Paths.Completion
		case PathGenerate:
			v = cfg.Paths.Generate
		case PathTokenize:
			v = cfg.Paths.Tokenize
		case PathHealth:
			v = cfg.Paths.Health
		}
		if v != "" {
			return v
		}
	}
	// Backward compatibility: HealthPath used to be the single knob.
	if which == PathHealth && cfg.HealthPath != "" {
		return cfg.HealthPath
	}
	return defaultPaths(cfg.Engine)[which]
}

// ForcedEndpoint returns the explicitly pinned scoring surface, model level
// winning over provider level. Nil means capability resolution applies.
func ForcedEndpoint(pcfg config.ProviderConfig, modelCfg *config.ModelConfig) *config.EndpointKind {
	if modelCfg != nil && modelCfg.Endpoint != nil {
		return modelCfg.Endpoint
	}
	return pcfg.Endpoint
}
