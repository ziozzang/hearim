package httpapi

import (
	"net/http"
	"runtime"
	"strings"
	"time"

	"hearim/internal/hearim/usage"
)

// handleMetrics exposes the Prometheus text format. It sits behind the same
// auth as the API: scrape configs should pass the gateway key, or an
// internal-only listener should front it.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	// Refresh gauges that are cheap to compute at scrape time.
	s.refreshScrapeGauges()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(s.Metrics.Render()))
}

// refreshScrapeGauges updates budget, readiness, and process gauges.
func (s *Server) refreshScrapeGauges() {
	state, frac := s.Budget.State()
	s.Metrics.Gauge("hearim_budget_utilization_ratio",
		"Spent fraction of the included budget window.", nil).Set(frac)
	s.Metrics.Gauge("hearim_budget_state",
		"0=ok 1=warn 2=throttle 3=hard_stop.", nil).Set(float64(state))

	readyRoutes := 0
	for _, rt := range s.Routes() {
		ready := 0
		if routeReady(rt) {
			ready = 1
			readyRoutes++
		}
		s.Metrics.Gauge("hearim_route_ready",
			"1 when the route's token label registry is ready.",
			map[string]string{"provider": rt.ProviderID, "model": rt.BackendModel}).Set(float64(ready))
	}
	s.Metrics.Gauge("hearim_routes_ready", "Number of ready routes.", nil).Set(float64(readyRoutes))

	// Prefill cache utilization on a reported basis, per route (§8.5).
	for route, row := range s.Ledger.Snapshot() {
		total := row.CachedPrefillTokens + row.UncachedPrefillTokens
		ratio := 0.0
		if total > 0 {
			ratio = float64(row.CachedPrefillTokens) / float64(total)
		}
		provider, model := splitRouteKey(route)
		s.Metrics.Gauge("hearim_prefill_cache_ratio",
			"Reported cached prefill tokens over total prefill tokens, cumulative.",
			map[string]string{"provider": provider, "model": model}).Set(ratio)
	}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	s.Metrics.Gauge("go_goroutines", "Number of Goroutines currently existing.", nil).
		Set(float64(runtime.NumGoroutine()))
	s.Metrics.Gauge("go_memstats_alloc_bytes", "Number of bytes allocated in heap and currently in use.", nil).
		Set(float64(ms.Alloc))
}

// recordRequestMetric is called by the middleware for every request.
func (s *Server) recordRequestMetric(path string, status int, d time.Duration) {
	s.Metrics.Counter("hearim_requests_total",
		"HTTP requests by path and status code.",
		map[string]string{"path": normalizePath(path), "status": itoaStatus(status)}).Inc()
	s.Metrics.Observe("hearim_request_duration_seconds",
		"HTTP request duration in seconds.",
		map[string]string{"path": normalizePath(path)}, d.Seconds())
}

// recordQuestionMetrics captures per-question evaluation telemetry.
func (s *Server) recordQuestionMetrics(model, provider, method, space string, agg usage.Aggregate) {
	m := s.Metrics
	m.Counter("hearim_question_calls_total",
		"Upstream scoring calls by model and method.",
		map[string]string{"model": model, "provider": provider, "scoring_method": method, "probability_space": space}).Inc()
	m.Counter("hearim_tokens_total",
		"Upstream token consumption by kind (input|output|cached, reported only).",
		map[string]string{"model": model, "kind": "input"}).Add(uint64(agg.InputTokens))
	m.Counter("hearim_tokens_total",
		"Upstream token consumption by kind (input|output|cached, reported only).",
		map[string]string{"model": model, "kind": "output"}).Add(uint64(agg.OutputTokens))
	m.Counter("hearim_tokens_total",
		"Upstream token consumption by kind (input|output|cached, reported only).",
		map[string]string{"model": model, "kind": "cached_reported"}).Add(uint64(agg.ReportedCachedTokens))
}

func splitRouteKey(k string) (string, string) {
	for i := 0; i < len(k); i++ {
		if k[i] == '/' {
			return k[:i], k[i+1:]
		}
	}
	return k, ""
}

func normalizePath(p string) string {
	if p == "" {
		return "/"
	}
	return strings.TrimSuffix(p, "/")
}

func itoaStatus(status int) string {
	switch status {
	case 200:
		return "200"
	case 401:
		return "401"
	case 422:
		return "422"
	case 429:
		return "429"
	case 503:
		return "503"
	case 529:
		return "529"
	default:
		return itoa(status)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [12]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
