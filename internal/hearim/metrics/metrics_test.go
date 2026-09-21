package metrics

import (
	"strings"
	"testing"
)

func TestRenderTextFormat(t *testing.T) {
	r := New()
	r.Counter("hearim_requests_total", "help", map[string]string{"path": "/v1/systemone", "status": "200"}).Add(3)
	r.Gauge("hearim_routes_ready", "ready routes", nil).Set(2)
	r.Observe("hearim_request_duration_seconds", "dur", map[string]string{"path": "/v1/systemone"}, 0.02)

	out := r.Render()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	wantSubstrings := []string{
		"# HELP hearim_requests_total help",
		"# TYPE hearim_requests_total counter",
		`hearim_requests_total{path="/v1/systemone",status="200"} 3`,
		"# TYPE hearim_routes_ready gauge",
		"hearim_routes_ready 2",
		"# TYPE hearim_request_duration_seconds histogram",
		`hearim_request_duration_seconds_bucket{path="/v1/systemone",le="0.01"} 0`,
		`hearim_request_duration_seconds_bucket{path="/v1/systemone",le="0.025"} 1`,
		`hearim_request_duration_seconds_bucket{path="/v1/systemone",le="+Inf"} 1`,
		`hearim_request_duration_seconds_count{path="/v1/systemone"} 1`,
	}
	joined := strings.Join(lines, "\n")
	for _, want := range wantSubstrings {
		if !strings.Contains(joined, want) {
			t.Errorf("rendered output missing %q\n--- output ---\n%s", want, out)
		}
	}
}

func TestLabelEscaping(t *testing.T) {
	r := New()
	r.Counter("c", "h", map[string]string{"model": `gemma"4\31b`}).Inc()
	out := r.Render()
	if !strings.Contains(out, `model="gemma\"4\\31b"`) {
		t.Errorf("escaping broken: %s", out)
	}
}

func TestConcurrentIncrement(t *testing.T) {
	r := New()
	c := r.Counter("c", "h", nil)
	done := make(chan struct{})
	for i := 0; i < 8; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 100; j++ {
				c.Inc()
			}
		}()
	}
	for i := 0; i < 8; i++ {
		<-done
	}
	if !strings.Contains(r.Render(), "c 800") {
		t.Error("lost increments")
	}
}
