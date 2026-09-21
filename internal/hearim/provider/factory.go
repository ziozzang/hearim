package provider

import (
	"fmt"

	"hearim/internal/hearim/config"
)

// NewAdapter builds the engine-specific adapter for a provider config.
func NewAdapter(cfg config.ProviderConfig) (Adapter, error) {
	switch cfg.Engine {
	case config.EngineOllama:
		return NewOllama(cfg), nil
	case config.EngineLlamaPP:
		return NewLlamaCpp(cfg), nil
	case config.EngineVLLM:
		return NewVLLM(cfg), nil
	case config.EngineSGLang:
		return NewSGLang(cfg), nil
	case config.EngineGeneric:
		return NewGeneric(cfg), nil
	default:
		return nil, fmt.Errorf("provider: unknown engine %q for %s", cfg.Engine, cfg.ID)
	}
}

// Registry resolves provider/model pairs to adapters (TODO.md §3.1: model
// aliases bind to provider:model routes).
type Registry struct {
	adapters map[string]Adapter
}

func NewRegistry() *Registry { return &Registry{adapters: map[string]Adapter{}} }

func (r *Registry) Add(a Adapter) { r.adapters[a.ID()] = a }

func (r *Registry) Get(providerID string) (Adapter, bool) {
	a, ok := r.adapters[providerID]
	return a, ok
}

func (r *Registry) All() []Adapter {
	out := make([]Adapter, 0, len(r.adapters))
	for _, a := range r.adapters {
		out = append(out, a)
	}
	return out
}
