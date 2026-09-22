// Package mockinfer implements source-shaped inference API scenarios without
// loading model weights. Its byte tokenizer and fixed distribution are synthetic.
// It is a protocol regression tool, not an inference engine or compatibility proof.
package mockinfer

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type Options struct {
	Engine         string
	Scenario       string
	ModelScenarios map[string]string
}

var Scenarios = []string{"normal", "selected-rejected", "mask-rejected", "mask-ignored", "missing-candidates", "topk-missing", "processed-logprobs", "post-sampling-ignored", "no-tokenizer"}

type Request struct {
	Path string         `json:"path"`
	Body map[string]any `json:"body"`
}

type Server struct {
	opts     Options
	mu       sync.Mutex
	requests []Request
}

func New(opts Options) (*Server, error) {
	if opts.Engine != "vllm" && opts.Engine != "sglang" && opts.Engine != "llama.cpp" {
		return nil, fmt.Errorf("unknown engine %q", opts.Engine)
	}
	if opts.Scenario == "" {
		opts.Scenario = "normal"
	}
	valid := func(s string) bool {
		for _, v := range Scenarios {
			if v == s {
				return true
			}
		}
		return false
	}
	if !valid(opts.Scenario) {
		return nil, fmt.Errorf("unknown scenario %q", opts.Scenario)
	}
	models := map[string]string{}
	for m, s := range opts.ModelScenarios {
		if !valid(s) {
			return nil, fmt.Errorf("model %s: unknown scenario %q", m, s)
		}
		models[m] = s
	}
	opts.ModelScenarios = models
	return &Server{opts: opts}, nil
}

// Requests returns detached copies so tests can inspect requests concurrently.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, _ := json.Marshal(s.requests)
	var out []Request
	_ = json.Unmarshal(b, &out)
	return out
}

func TokenID(label byte) int { return int(label) + 1 }

func tokenize(text string) []int {
	ids := []int{0} // synthetic BOS; each subsequent byte is a token
	for _, b := range []byte(text) {
		ids = append(ids, TokenID(b))
	}
	return ids
}

func tokenText(id int) string {
	if id == 0 {
		return "<BOS>"
	}
	if id < 1 || id > 256 {
		return ""
	}
	return string([]byte{byte(id - 1)})
}

// Fixed normalized vocabulary, independent of the adapter and prompt labels.
// X is the most likely token, so a working {A,B} mask must change the sample.
func distribution() []float64 {
	p := make([]float64, 257)
	for i := range p {
		p[i] = 0.05 / 251
	}
	for b, v := range map[byte]float64{'X': 0.4, 'A': 0.2, 'B': 0.1, '1': 0.15, '0': 0.05, '2': 0.05} {
		p[TokenID(b)] = v
	}
	return p
}

func send(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Hearim-Mock", "synthetic-source-shaped")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func failure(w http.ResponseWriter, status int, message string) {
	send(w, status, map[string]any{"error": map[string]any{"message": message, "type": "mock_protocol_error"}})
}

func number(m map[string]any, k string) int { v, _ := m[k].(float64); return int(v) }
func str(m map[string]any, k string) string { v, _ := m[k].(string); return v }
func truth(m map[string]any, k string) bool { v, _ := m[k].(bool); return v }

func tokenIDs(v any) ([]int, error) {
	if v == nil {
		return nil, nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("token IDs must be an array")
	}
	ids := make([]int, 0, len(arr))
	for _, el := range arr {
		n, ok := el.(float64)
		if !ok || n != math.Trunc(n) || n < 0 || n > 256 {
			return nil, fmt.Errorf("token ID out of mock vocabulary [0,256]")
		}
		ids = append(ids, int(n))
	}
	return ids, nil
}

func inputTokens(body map[string]any) []int {
	for _, k := range []string{"input_ids", "prompt"} {
		if ids, err := tokenIDs(body[k]); err == nil && ids != nil {
			return ids
		}
	}
	for _, k := range []string{"prompt", "text", "content"} {
		if text, ok := body[k].(string); ok {
			return tokenize(text)
		}
	}
	if msgs, ok := body["messages"].([]any); ok {
		var text strings.Builder
		for _, m := range msgs {
			if msg, ok := m.(map[string]any); ok {
				text.WriteString(str(msg, "content"))
			}
		}
		return tokenize(text.String())
	}
	return []int{0}
}

func grammarIDs(grammar string) ([]int, error) {
	const prefix = "root ::= "
	if !strings.HasPrefix(grammar, prefix) {
		return nil, fmt.Errorf("mock supports only root label alternatives")
	}
	var ids []int
	for _, part := range strings.Split(strings.TrimPrefix(grammar, prefix), " | ") {
		label, err := strconv.Unquote(strings.TrimSpace(part))
		if err != nil || len(label) != 1 {
			return nil, fmt.Errorf("mock grammar requires single-byte quoted labels")
		}
		ids = append(ids, TokenID(label[0]))
	}
	return ids, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		switch r.URL.Path {
		case "/mock/status":
			send(w, 200, map[string]any{"engine": s.opts.Engine, "scenario": s.opts.Scenario, "models": s.opts.ModelScenarios, "synthetic": true})
			return
		case "/health", "/healthz", "/version":
			send(w, 200, map[string]any{"status": "ok", "version": "hearim-mock"})
			return
		case "/get_server_info":
			send(w, 200, map[string]any{"version": []string{"hearim-mock"}, "health": "ok"})
			return
		}
	}
	if r.Method != "POST" {
		failure(w, 404, "unknown endpoint")
		return
	}
	var body map[string]any
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	if err := dec.Decode(&body); err != nil || body == nil {
		failure(w, 400, "invalid JSON object")
		return
	}
	s.mu.Lock()
	if len(s.requests) < 1000 {
		s.requests = append(s.requests, Request{Path: r.URL.Path, Body: body})
	}
	s.mu.Unlock()
	scenario := s.opts.Scenario
	if p, ok := s.opts.ModelScenarios[str(body, "model")]; ok {
		scenario = p
	}
	if body["extra_body"] != nil {
		failure(w, 400, "extra_body is an SDK argument; flatten fields into the HTTP body")
		return
	}
	if r.URL.Path == "/tokenize" || r.URL.Path == "/v1/tokenize" {
		if scenario == "no-tokenizer" {
			failure(w, 404, "tokenizer endpoint unavailable")
			return
		}
		key := "tokens"
		if s.opts.Engine == "sglang" {
			key = "input_ids"
		}
		send(w, 200, map[string]any{key: inputTokens(body)})
		return
	}
	nativeSG := s.opts.Engine == "sglang" && r.URL.Path == "/generate"
	nativeLlama := s.opts.Engine == "llama.cpp" && r.URL.Path == "/completion"
	chat := r.URL.Path == "/v1/chat/completions"
	if !nativeSG && !nativeLlama && !chat && r.URL.Path != "/v1/completions" {
		failure(w, 404, "endpoint unavailable for engine")
		return
	}
	if truth(body, "stream") {
		failure(w, 400, "mock implements non-streaming scoring only")
		return
	}
	sp, _ := body["sampling_params"].(map[string]any)
	selectedKey, topK := "logprob_token_ids", number(body, "logprobs")
	if chat {
		topK = number(body, "top_logprobs")
	}
	if nativeSG {
		selectedKey, topK = "token_ids_logprob", number(body, "top_logprobs_num")
	}
	if nativeLlama {
		selectedKey, topK = "", number(body, "n_probs")
	}
	selected, err := tokenIDs(body[selectedKey])
	if err != nil {
		failure(w, 400, err.Error())
		return
	}
	if len(selected) > 0 && scenario == "selected-rejected" {
		failure(w, 400, "selected-token logprobs unsupported")
		return
	}
	if s.opts.Engine == "vllm" && len(selected) > 128 {
		failure(w, 400, "logprob_token_ids exceeds max allowed 128")
		return
	}
	if topK > 128 {
		failure(w, 400, "top logprobs must be between 0 and 128")
		return
	}
	maskValue := body["allowed_token_ids"]
	if nativeSG {
		maskValue = sp["allowed_token_ids"]
	}
	mask, err := tokenIDs(maskValue)
	if err != nil {
		failure(w, 400, err.Error())
		return
	}
	grammar := str(body, "grammar")
	if grammar != "" {
		mask, err = grammarIDs(grammar)
		if err != nil {
			failure(w, 400, err.Error())
			return
		}
	}
	if maskValue != nil || grammar != "" {
		if len(mask) == 0 {
			failure(w, 400, "empty candidate mask")
			return
		}
		if scenario == "mask-rejected" || (nativeSG && scenario != "mask-ignored") {
			failure(w, 400, "candidate mask unsupported")
			return
		}
	}
	raw := distribution()
	processed := append([]float64(nil), raw...)
	if len(mask) > 0 && scenario != "mask-ignored" {
		processed = make([]float64, len(raw))
		mass := 0.0
		for _, id := range mask {
			processed[id] = raw[id]
		}
		for _, p := range processed {
			mass += p
		}
		for i := range processed {
			processed[i] /= mass
		}
	}
	sample := 0
	for i, p := range processed {
		if p > processed[sample] {
			sample = i
		}
	}
	post := nativeLlama && truth(body, "post_sampling_probs") && scenario != "post-sampling-ignored"
	reported := raw
	if post || scenario == "processed-logprobs" {
		reported = processed
	}
	chosen := append([]int(nil), selected...)
	if len(chosen) == 0 {
		for id, p := range reported {
			if p > 0 {
				chosen = append(chosen, id)
			}
		}
		sort.Slice(chosen, func(i, j int) bool {
			if reported[chosen[i]] == reported[chosen[j]] {
				return chosen[i] < chosen[j]
			}
			return reported[chosen[i]] > reported[chosen[j]]
		})
		if topK < len(chosen) {
			if topK < 0 {
				topK = 0
			}
			chosen = chosen[:topK]
		}
	}
	if scenario == "missing-candidates" || (scenario == "topk-missing" && len(selected) == 0 && len(mask) == 0) {
		filtered := []int{}
		for _, id := range chosen {
			if id != TokenID('B') && id != TokenID('2') {
				filtered = append(filtered, id)
			}
		}
		chosen = filtered
	}
	inputs := inputTokens(body)
	if nativeSG {
		entries := []any{}
		for _, id := range chosen {
			var text any
			if truth(body, "return_text_in_logprobs") {
				text = tokenText(id)
			}
			entries = append(entries, []any{math.Log(reported[id]), id, text})
		}
		meta := map[string]any{"id": "mock-sglang", "prompt_tokens": len(inputs), "completion_tokens": 1, "cached_tokens": 0, "output_token_logprobs": []any{[]any{math.Log(raw[sample]), sample, nil}}}
		key := "output_top_logprobs"
		if len(selected) > 0 {
			key = "output_token_ids_logprobs"
		}
		meta[key] = []any{entries}
		if _, ok := body["logprob_start_len"]; ok {
			vals := []any{}
			for i, id := range inputs {
				var lp any
				if i > 0 {
					lp = math.Log(raw[id])
				}
				vals = append(vals, []any{lp, id, nil})
			}
			meta["input_token_logprobs"] = vals
		}
		send(w, 200, map[string]any{"text": tokenText(sample), "meta_info": meta})
		return
	}
	if nativeLlama {
		entries := []any{}
		field, topField := "logprob", "top_logprobs"
		if post {
			field, topField = "prob", "top_probs"
		}
		entry := func(id int) map[string]any {
			p := reported[id]
			if !post {
				p = math.Log(p)
			}
			return map[string]any{"id": id, "token": tokenText(id), field: p}
		}
		for _, id := range chosen {
			entries = append(entries, entry(id))
		}
		out := entry(sample)
		out[topField] = entries
		send(w, 200, map[string]any{"content": tokenText(sample), "tokens": []int{sample}, "completion_probabilities": []any{out}, "tokens_evaluated": len(inputs), "tokens_cached": 0})
		return
	}
	choice := map[string]any{"text": tokenText(sample), "index": 0, "finish_reason": "length"}
	if chat {
		entries := []any{}
		for _, id := range chosen {
			entries = append(entries, map[string]any{"token": tokenText(id), "logprob": math.Log(reported[id])})
		}
		choice["message"] = map[string]any{"role": "assistant", "content": tokenText(sample)}
		choice["logprobs"] = map[string]any{"content": []any{map[string]any{"token": tokenText(sample), "logprob": math.Log(reported[sample]), "top_logprobs": entries}}}
	} else {
		entries := map[string]float64{}
		for _, id := range chosen {
			entries[tokenText(id)] = math.Log(reported[id])
		}
		choice["logprobs"] = map[string]any{"tokens": []string{tokenText(sample)}, "token_logprobs": []float64{math.Log(reported[sample])}, "top_logprobs": []any{entries}}
	}
	if _, ok := body["prompt_logprobs"]; ok {
		vals := []any{nil}
		for _, id := range inputs[1:] {
			vals = append(vals, map[string]any{strconv.Itoa(id): map[string]any{"logprob": math.Log(raw[id]), "rank": 1}})
		}
		choice["prompt_logprobs"] = vals
	}
	send(w, 200, map[string]any{"id": "mock-" + s.opts.Engine, "choices": []any{choice}, "usage": map[string]any{"prompt_tokens": len(inputs), "completion_tokens": 1}})
}
