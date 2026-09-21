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
}

func newHTTPClient(cfg config.ProviderConfig) *httpClient {
	base, err := url.Parse(strings.TrimRight(cfg.BaseURL, "/"))
	if err != nil {
		base = &url.URL{Scheme: "http", Host: "invalid"}
	}
	return &httpClient{
		base:     base,
		apiKey:   cfg.APIKey,
		client:   &http.Client{Timeout: cfg.RequestTimeout},
		maxRetry: cfg.MaxRetries,
	}
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
		req, err := c.newRequest(ctx, method, path, payload)
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

func (c *httpClient) newRequest(ctx context.Context, method, path string, payload []byte) (*http.Request, error) {
	u := c.joinURL(path)
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

// joinURL resolves an endpoint path against the configured base URL.
//
// Base URLs are configured either as a server root (http://host:8000) or as
// an OpenAI-style root ending in /v1 (https://ollama.com/v1). Both OpenAI
// paths (/v1/completions) and native paths (/api/version, /generate,
// /completion, /tokenize) hang off the server root, so a trailing "/v1" in
// the base is stripped before joining — otherwise it duplicates
// (/v1/v1/completions) or misplaces native endpoints (/v1/api/version).
func (c *httpClient) joinURL(path string) string {
	root := *c.base
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
	req, err := c.newRequest(ctx, method, path, payload)
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
