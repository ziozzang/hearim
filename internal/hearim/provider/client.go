package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"hearim/internal/hearim/config"
)

// UpstreamError classifies adapter failures for the Jev error surface
// (TODO.md §10): 401 no-retry, 429 backoff, 5xx limited retry.
type UpstreamError struct {
	Status    int
	Code      string
	Message   string
	Retriable bool
	Body      string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream %d %s: %s", e.Status, e.Code, e.Message)
}

func classifyStatus(status int, body string) *UpstreamError {
	msg := strings.TrimSpace(body)
	if len(msg) > 500 {
		msg = msg[:500]
	}
	switch {
	case status == http.StatusUnauthorized:
		return &UpstreamError{Status: status, Code: "unauthorized", Message: msg, Retriable: false, Body: body}
	case status == http.StatusTooManyRequests:
		return &UpstreamError{Status: status, Code: "rate_limited", Message: msg, Retriable: true, Body: body}
	case status == 529:
		return &UpstreamError{Status: status, Code: "backend_overloaded", Message: msg, Retriable: true, Body: body}
	case status >= 500:
		return &UpstreamError{Status: status, Code: "backend_unavailable", Message: msg, Retriable: true, Body: body}
	case status == http.StatusNotFound:
		return &UpstreamError{Status: status, Code: "not_found", Message: msg, Retriable: false, Body: body}
	case status == http.StatusBadRequest:
		return &UpstreamError{Status: status, Code: "bad_request", Message: msg, Retriable: false, Body: body}
	default:
		return &UpstreamError{Status: status, Code: "error", Message: msg, Retriable: false, Body: body}
	}
}

// httpClient wraps one provider's base URL with auth, timeouts, retries and
// jittered backoff. Raw request/response bodies are never logged.
type httpClient struct {
	base     *url.URL
	apiKey   string
	client   *http.Client
	maxRetry int
	roots    []*url.URL
	rr       uint64
}

func newHTTPClient(cfg config.ProviderConfig) *httpClient {
	// Endpoint pool: base_url first, then any base_urls entries (deduped).
	// With a pool, retries rotate to the next host, so a dead replica is
	// bypassed instead of retried in place.
	candidates := make([]string, 0, 1+len(cfg.BaseURLs))
	if cfg.BaseURL != "" {
		candidates = append(candidates, cfg.BaseURL)
	}
	candidates = append(candidates, cfg.BaseURLs...)
	seen := map[string]bool{}
	var roots []*url.URL
	for _, raw := range candidates {
		u := strings.TrimRight(strings.TrimSpace(raw), "/")
		if u == "" || seen[u] {
			continue
		}
		seen[u] = true
		parsed, err := url.Parse(u)
		if err != nil || parsed.Host == "" {
			continue
		}
		roots = append(roots, parsed)
	}
	if len(roots) == 0 {
		roots = []*url.URL{{Scheme: "http", Host: "invalid"}}
	}
	return &httpClient{
		base:     roots[0],
		roots:    roots,
		apiKey:   cfg.APIKey,
		client:   &http.Client{Timeout: cfg.RequestTimeout},
		maxRetry: cfg.MaxRetries,
	}
}

// nextRootIdx advances the round-robin once per request so consecutive
// requests spread over replicas.
func (c *httpClient) nextRootIdx() int {
	if len(c.roots) == 1 {
		return 0
	}
	return int(atomic.AddUint64(&c.rr, 1)-1) % len(c.roots)
}

// rootAt returns the endpoint for a request's attempt n: attempt 0 uses the
// round-robin start, each retry moves to the NEXT replica, so a dead host
// is failed over instead of retried in place.
func (c *httpClient) rootAt(start, attempt int) *url.URL {
	if len(c.roots) == 1 {
		return c.roots[0]
	}
	return c.roots[(start+attempt)%len(c.roots)]
}

// do issues a request; out (when non-nil) receives the decoded JSON body.
// Retries apply only to classified retriable statuses and transport errors.
func (c *httpClient) do(ctx context.Context, method, path string, body, out any) error {
	var payload []byte
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("provider: encode request: %w", err)
		}
		payload = b
	}

	attempts := c.maxRetry + 1
	start := c.nextRootIdx()
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			// Jittered exponential backoff, capped well under request budget.
			backoff := time.Duration(1<<uint(attempt)) * 250 * time.Millisecond
			jitter := time.Duration(rand.Int63n(int64(backoff) / 2))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff + jitter):
			}
		}
		req, err := c.newRequest(ctx, method, path, payload, c.rootAt(start, attempt))
		if err != nil {
			return err
		}
		resp, err := c.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("provider: transport: %w", err)
			continue
		}
		data, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("provider: read response: %w", err)
			continue
		}
		if resp.StatusCode >= 400 {
			ue := classifyStatus(resp.StatusCode, string(data))
			if ue.Retriable && attempt < attempts-1 {
				lastErr = ue
				continue
			}
			return ue
		}
		if out == nil {
			return nil
		}
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("provider: decode response: %w", err)
		}
		return nil
	}
	return lastErr
}

func (c *httpClient) newRequest(ctx context.Context, method, path string, payload []byte, root *url.URL) (*http.Request, error) {
	u := joinURL(root, path)
	req, err := http.NewRequestWithContext(ctx, method, u, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "hearim/0.1")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	return req, nil
}

// joinURL resolves an endpoint path against a base URL.
//
// Base URLs are configured either as a server root (http://host:8000) or as
// an OpenAI-style root ending in /v1 (https://ollama.com/v1). Both OpenAI
// paths (/v1/completions) and native paths (/api/version, /generate,
// /completion, /tokenize) hang off the server root, so a trailing "/v1" in
// the base is stripped before joining — otherwise it duplicates
// (/v1/v1/completions) or misplaces native endpoints (/v1/api/version).
func joinURL(base *url.URL, path string) string {
	root := *base
	root.Path = strings.TrimSuffix(strings.TrimRight(root.Path, "/"), "/v1")
	if root.Path == "" {
		root.Path = "/"
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	merged := strings.TrimRight(root.Path, "/") + path
	if merged == "" {
		merged = "/"
	}
	root.Path = merged
	return root.String()
}

// doRaw performs a request without retry, returning status and body — used
// by capability probes that must observe raw behavior.
func (c *httpClient) doRaw(ctx context.Context, method, path string, body any) (int, []byte, error) {
	var payload []byte
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		payload = b
	}
	req, err := c.newRequest(ctx, method, path, payload, c.rootAt(c.nextRootIdx(), 0))
	if err != nil {
		return 0, nil, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	return resp.StatusCode, data, err
}

// IsNotFound reports 404-shaped upstream errors (missing endpoints/models).
func IsNotFound(err error) bool {
	var ue *UpstreamError
	return errors.As(err, &ue) && ue.Status == http.StatusNotFound
}

// IsRetriable reports whether the error class merits another attempt.
func IsRetriable(err error) bool {
	var ue *UpstreamError
	return errors.As(err, &ue) && ue.Retriable
}
