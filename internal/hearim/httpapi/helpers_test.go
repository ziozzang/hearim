package httpapi

import (
	"context"
	"time"

	"hearim/internal/hearim/config"
)

func configParseForTest(yaml string) (*config.Config, error) {
	return config.Parse([]byte(yaml))
}

func defaultCtx() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	// Leak cancel intentionally for deferred Shutdown use in tests.
	_ = cancel
	return ctx
}
