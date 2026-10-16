package httpapi

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeJev is a real-Jev-shaped System One endpoint: it records the raw
// request and answers in the live wire format (index-keyed score legend).
type fakeJev struct {
	status   int
	lastBody []byte
	lastAuth string
	calls    int
}

func (f *fakeJev) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		f.lastBody, _ = io.ReadAll(r.Body)
		f.lastAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		if f.status != 0 {
			w.WriteHeader(f.status)
			fmt.Fprint(w, `{"error":{"type":"server_error","message":"Upstream request failed: Insufficient account funds"}}`)
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(f.lastBody, &req)
		fmt.Fprintf(w, `{"model":%q,"answers":{
			"dept":{"type":"choice","choice":"billing","confidence":0.9,"probabilities":{"returns":0.05,"billing":0.95}},
			"mood":{"type":"score","score":1,"confidence":0.97,"legend":{"0":"Calm","1":"Frustrated"},"probabilities":{"0":0.02,"1":0.98}},
			"urgent":{"type":"noul","noul":0.96}},
			"usage":{"input_tokens":372,"output_tokens":57}}`, req.Model)
	})
	return mux
}

func newJevTestServer(t *testing.T, upstreamURL string) http.Handler {
	t.Helper()
	cfg, err := configParseForTest(fmt.Sprintf(`
gateway:
  default_model: jev-ref
providers:
  - id: zen
    engine: jev
    base_url: %s/v1
    api_key: upstream-secret
    models: ["jev-1.13-free"]
model_aliases:
  jev-ref: zen:jev-1.13-free
`, upstreamURL))
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(&discardWriter{}, nil))
	srv, err := New(cfg, logger, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if ok, why := srv.Ready(); !ok {
		t.Fatalf("jev route not ready: %s", why)
	}
	return srv.Handler()
}

const jevPassthroughReq = `{"model":"jev-ref","state":"결제가 사흘째 실패해요","questions":{
	"dept":{"type":"choice","instructions":"Which team?","criteria":{"returns":"Refunds","billing":"Payments"}},
	"mood":{"type":"score","instructions":"How frustrated?","criteria":["Calm","Frustrated"]},
	"urgent":{"type":"noul","instructions":"Is it urgent?"}}}`

func TestJevPassthrough(t *testing.T) {
	up := &fakeJev{}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	h := newJevTestServer(t, upSrv.URL)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(jevPassthroughReq)))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	// One upstream call for all questions, backend model substituted, key
	// order of choice criteria preserved byte-for-byte.
	if up.calls != 1 {
		t.Fatalf("upstream calls = %d, want 1", up.calls)
	}
	if up.lastAuth != "Bearer upstream-secret" {
		t.Fatalf("upstream auth = %q", up.lastAuth)
	}
	sent := string(up.lastBody)
	if !strings.Contains(sent, `"model":"jev-1.13-free"`) {
		t.Fatalf("backend model not substituted: %s", sent)
	}
	if strings.Index(sent, `"returns"`) > strings.Index(sent, `"billing"`) {
		t.Fatalf("criteria order changed: %s", sent)
	}

	var resp struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "jev-ref" {
		t.Fatalf("response model = %q, want public alias", resp.Model)
	}
	if len(resp.Answers) != 3 || resp.Usage.InputTokens != 372 || resp.Usage.OutputTokens != 57 {
		t.Fatalf("unexpected response: %s", rec.Body.String())
	}
	// Answers pass through unchanged, including the upstream legend shape.
	if !strings.Contains(string(resp.Answers["mood"]), `"legend":{"0":"Calm","1":"Frustrated"}`) {
		t.Fatalf("score answer altered: %s", resp.Answers["mood"])
	}
	if got := rec.Header().Get("x-jev-implementation"); got != "hearim-jev-passthrough" {
		t.Fatalf("x-jev-implementation = %q", got)
	}
	if got := rec.Header().Get("x-jev-backend-model"); got != "zen/jev-1.13-free" {
		t.Fatalf("x-jev-backend-model = %q", got)
	}
}

func TestJevPassthroughUpstreamError(t *testing.T) {
	up := &fakeJev{status: http.StatusPaymentRequired}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	h := newJevTestServer(t, upSrv.URL)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(jevPassthroughReq)))
	if rec.Code != 529 {
		t.Fatalf("status %d, want 529: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "backend_unavailable") || !strings.Contains(rec.Body.String(), "402") {
		t.Fatalf("error body: %s", rec.Body.String())
	}
	if up.calls != 1 {
		t.Fatalf("402 must not be retried; calls = %d", up.calls)
	}
}

func TestJevPassthroughRejectsInvalidBeforeUpstream(t *testing.T) {
	up := &fakeJev{}
	upSrv := httptest.NewServer(up.handler())
	defer upSrv.Close()
	h := newJevTestServer(t, upSrv.URL)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone",
		strings.NewReader(`{"model":"jev-ref","state":"x","questions":{"q":{"type":"choice","instructions":"?"}}}`)))
	if rec.Code != 422 {
		t.Fatalf("status %d, want 422: %s", rec.Code, rec.Body.String())
	}
	if up.calls != 0 {
		t.Fatalf("invalid request reached upstream")
	}
}
