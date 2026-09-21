package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"hearim/internal/hearim/jev"
)

// handleHealth reports liveness.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// handleReady implements the §6.5 readiness rule: if no default route is
// ready, the probe fails.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	ok, detail := s.Ready()
	w.Header().Set("Content-Type", "application/json")
	status := http.StatusOK
	if !ok {
		status = http.StatusServiceUnavailable
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ready":  ok,
		"detail": detail,
	})
}

// handleRoutes exposes route and registry state for operations.
func (s *Server) handleRoutes(w http.ResponseWriter, r *http.Request) {
	type routeView struct {
		Provider  string `json:"provider"`
		Engine    string `json:"engine"`
		Model     string `json:"model"`
		Endpoint  string `json:"endpoint"`
		Ready     bool   `json:"ready"`
		Delimiter string `json:"delimiter,omitempty"`
		Labels    int    `json:"single_token_labels"`
		MaxExact  int    `json:"max_exact_candidates"`
		CachePlan string `json:"cache_plan"`
	}
	out := []routeView{}
	for _, rt := range s.Routes() {
		v := routeView{
			Provider:  rt.ProviderID,
			Engine:    string(rt.Engine),
			Model:     rt.BackendModel,
			Endpoint:  string(rt.Endpoint),
			Ready:     routeReady(rt),
			CachePlan: rt.CachePlan,
		}
		if rt.Registry != nil {
			v.Delimiter = displayDelimiter(rt.Registry.Delimiter)
			v.Labels = len(rt.Registry.Labels())
			v.MaxExact = rt.Registry.MaxExactCandidates
		}
		out = append(out, v)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"routes": out})
}

func displayDelimiter(d string) string {
	switch d {
	case "\n":
		return "\\n"
	case " ":
		return "\\s"
	default:
		return strings.TrimSpace(d)
	}
}

// withCommon adds auth and request logging.
func (s *Server) withCommon(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		reqID := newRequestID()
		w.Header().Set("x-request-id", reqID)

		if len(s.APIKeys) > 0 && !isHealthPath(r.URL.Path) {
			key := bearerToken(r)
			if key == "" || !s.APIKeys[key] {
				writeJevError(w, r, jev.NewError(jev.CodeUnauthorized, "missing or invalid API key"))
				return
			}
		}

		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		s.recordRequestMetric(r.URL.Path, sw.status, time.Since(start))

		// §11: no raw state in logs; question ids, token counts and hashes only.
		s.Logger.Info("http",
			"method", r.Method,
			"path", r.URL.Path,
			"status", sw.status,
			"duration_ms", time.Since(start).Milliseconds(),
			"request_id", reqID,
		)
	})
}

func newRequestID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return time.Now().UTC().Format("20060102T150405.000000000")
	}
	return hex.EncodeToString(b[:])
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func isHealthPath(p string) bool {
	return p == "/healthz" || p == "/readyz"
}

func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if h == "" {
		return ""
	}
	parts := strings.SplitN(h, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}
