package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/config"
)

// openaiCompletionsRequest is the shared OpenAI-compatible completion body.
// Engine-specific extras ride in Extra.
type openaiCompletionsRequest struct {
	Model       string         `json:"model"`
	Prompt      any            `json:"prompt"` // string or []int
	MaxTokens   int            `json:"max_tokens,omitempty"`
	Temperature float64        `json:"temperature"`
	TopP        float64        `json:"top_p"`
	Logprobs    *int           `json:"logprobs,omitempty"`
	Stream      bool           `json:"stream"`
	Extra       map[string]any `json:"-"`
}

func (r openaiCompletionsRequest) MarshalJSON() ([]byte, error) {
	type alias openaiCompletionsRequest
	b, err := json.Marshal(alias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return b, nil
	}
	m := map[string]any{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		m[k] = v
	}
	return json.Marshal(m)
}

// openaiChatRequest renders the chat variant for chat-only routes (§5.1).
type openaiChatRequest struct {
	Model       string                `json:"model"`
	Messages    []compile.ChatMessage `json:"messages"`
	MaxTokens   int                   `json:"max_tokens,omitempty"`
	Temperature float64               `json:"temperature"`
	TopP        float64               `json:"top_p"`
	Logprobs    bool                  `json:"logprobs"`
	TopLogprobs int                   `json:"top_logprobs,omitempty"`
	Stream      bool                  `json:"stream"`
	Extra       map[string]any        `json:"-"`
}

func (r openaiChatRequest) MarshalJSON() ([]byte, error) {
	type alias openaiChatRequest
	b, err := json.Marshal(alias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return b, nil
	}
	m := map[string]any{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		m[k] = v
	}
	return json.Marshal(m)
}

// openaiLogprobEntry is a permissive view of one top-logprob entry across
// OpenAI-compatible servers: token text, optional bytes, logprob, and an
// optional token id. Some servers (vLLM logprob_token_ids) key entries by id
// and expose it as a numeric "token" string instead.
type openaiLogprobEntry struct {
	Token   string          `json:"token"`
	Bytes   json.RawMessage `json:"bytes"`
	Logprob float64         `json:"logprob"`
	TokenID *int64          `json:"token_id"`
	Rank    int             `json:"rank,omitempty"`
}

func (e openaiLogprobEntry) text() string {
	if b := decodeBytesField(e.Bytes); b != "" {
		return b
	}
	return e.Token
}

// decodeBytesField accepts []int (byte values) or a base64 string.
func decodeBytesField(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var ints []int
	if err := json.Unmarshal(raw, &ints); err == nil {
		b := make([]byte, 0, len(ints))
		for _, i := range ints {
			if i >= 0 && i <= 255 {
				b = append(b, byte(i))
			}
		}
		return string(b)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if dec, err := base64.StdEncoding.DecodeString(s); err == nil && utf8Printable(dec) {
			return string(dec)
		}
		return s
	}
	return ""
}

func utf8Printable(b []byte) bool {
	for _, c := range b {
		if c < 0x09 {
			return false
		}
	}
	return true
}

// openaiResponse is the permissive completions response.
type openaiResponse struct {
	Choices []struct {
		Text           string          `json:"text"`
		Index          int             `json:"index"`
		Logprobs       *openaiLogprobs `json:"logprobs"`
		PromptLogprobs json.RawMessage `json:"prompt_logprobs"` // vLLM
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	ID string `json:"id"`
}

type openaiLogprobs struct {
	Tokens      []string        `json:"tokens"`
	TopLogprobs json.RawMessage `json:"top_logprobs"`
}

// parseTopLogprobs handles the two shapes found in the wild:
//
//	OpenAI legacy: [{"A": -0.3, ...}]                (map token->logprob)
//	vLLM style:    [[{"token":"A","logprob":-0.3}]]  (array of entries)
func parseTopLogprobs(raw json.RawMessage) ([][]openaiLogprobEntry, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	// Array-of-arrays of entries (vLLM).
	var asArrArr []json.RawMessage
	if err := json.Unmarshal(raw, &asArrArr); err == nil && len(asArrArr) > 0 {
		if strings.HasPrefix(strings.TrimSpace(string(asArrArr[0])), "[") {
			out := make([][]openaiLogprobEntry, 0, len(asArrArr))
			for _, rr := range asArrArr {
				var entries []openaiLogprobEntry
				if err := json.Unmarshal(rr, &entries); err != nil {
					return nil, fmt.Errorf("provider: parse top_logprobs: %w", err)
				}
				out = append(out, entries)
			}
			return out, nil
		}
		// Array of maps (OpenAI legacy per position).
		out := make([][]openaiLogprobEntry, 0, len(asArrArr))
		for _, rr := range asArrArr {
			var m map[string]float64
			if err := json.Unmarshal(rr, &m); err != nil {
				return nil, fmt.Errorf("provider: parse top_logprobs map: %w", err)
			}
			entries := make([]openaiLogprobEntry, 0, len(m))
			for tok, lp := range m {
				entries = append(entries, openaiLogprobEntry{Token: tok, Logprob: lp})
			}
			out = append(out, entries)
		}
		return out, nil
	}
	return nil, fmt.Errorf("provider: unrecognized top_logprobs shape")
}

// matchCandidates resolves candidate logprobs from entries, preferring
// token IDs over text, then exact text, then space-trimmed text (§6.4: token
// ID first when the backend returns them; UTF-8 bytes compared otherwise).
func matchCandidates(entries []openaiLogprobEntry, ids []int, texts []string) (map[int]float64, bool) {
	found := make(map[int]float64, len(texts))
	all := true
	for i, text := range texts {
		id := -1
		if i < len(ids) {
			id = ids[i]
		}
		if id >= 0 {
			if lp, ok := matchByID(entries, id, text); ok {
				found[i] = lp
				continue
			}
		}
		if lp, ok := matchByText(entries, text); ok {
			found[i] = lp
			continue
		}
		all = false
	}
	return found, all
}

func matchByID(entries []openaiLogprobEntry, id int, text string) (float64, bool) {
	for _, e := range entries {
		if e.TokenID != nil && int(*e.TokenID) == id {
			return e.Logprob, true
		}
		// vLLM logprob_token_ids entries sometimes carry the id in "token".
		if n, err := strconv.ParseInt(e.Token, 10, 64); err == nil && int(n) == id {
			return e.Logprob, true
		}
		if n, err := strconv.ParseInt(e.text(), 10, 64); err == nil && int(n) == id && text == "" {
			return e.Logprob, true
		}
	}
	return 0, false
}

func matchByText(entries []openaiLogprobEntry, text string) (float64, bool) {
	for _, e := range entries {
		if e.Token == text || e.text() == text {
			return e.Logprob, true
		}
	}
	// Fall back to trimmed matching only when no exact entry exists.
	for _, e := range entries {
		if strings.TrimLeft(e.Token, " \t\n") == text || strings.TrimLeft(e.text(), " \t\n") == text {
			return e.Logprob, true
		}
	}
	return 0, false
}

// scoreViaCompletions is the shared scoring path for OpenAI-compatible
// servers. selectedField ("logprob_token_ids"/... ) enables direct ID
// requests when the server supports it; otherwise top-K is used.
func scoreViaCompletions(ctx context.Context, hc *httpClient, req NextTokenScoreRequest,
	model string, selectedField string, extras map[string]any) (*NextTokenScoreResult, error) {

	maxTokens := 1
	method0 := "" // set when wait-close scanning applies
	if req.WaitClose {
		maxTokens = req.MaxOutputTokens
		if maxTokens <= 0 {
			maxTokens = 256
		}
		method0 = "wait-close-tag"
	}
	body := openaiCompletionsRequest{
		Model:       model,
		Prompt:      promptValue(req),
		MaxTokens:   maxTokens,
		Temperature: tempOrOne(req.Temperature),
		TopP:        topPOrOne(req.TopP),
		Stream:      false,
		Extra:       map[string]any{},
	}
	method := "top-k"
	space := SpaceRaw
	if selectedField != "" && len(req.CandidateTokenIDs) > 0 {
		body.Extra[selectedField] = req.CandidateTokenIDs
		body.Logprobs = intptr(1) // per vLLM docs logprobs must be set
		if req.ConstrainToCandidates {
			body.Extra["allowed_token_ids"] = req.CandidateTokenIDs
			space = SpacePostMask
			method = "constrained-vocab"
		} else {
			method = "selected-token-ids"
		}
	} else {
		k := req.TopK
		if k <= 0 {
			k = 20
		}
		body.Logprobs = intptr(k)
		if req.ConstrainToCandidates {
			// Only send the mask when the server understands the field.
			body.Extra["allowed_token_ids"] = req.CandidateTokenIDs
			space = SpacePostMask
			method = "constrained-vocab"
		}
	}
	for k, v := range extras {
		if !protectedFields[k] {
			body.Extra[k] = v
		}
	}

	var out openaiResponse
	if err := hc.do(ctx, "POST", "/v1/completions", body, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 || out.Choices[0].Logprobs == nil {
		return nil, fmt.Errorf("provider: no logprobs in completions response")
	}
	positions, err := parseTopLogprobs(out.Choices[0].Logprobs.TopLogprobs)
	if err != nil {
		return nil, err
	}
	if len(positions) == 0 {
		return nil, fmt.Errorf("provider: empty top_logprobs")
	}
	pos := 0
	if req.WaitClose {
		pos, err = scanAfterCloseTag(out.Choices[0].Logprobs.Tokens, req.CloseTag)
		if err != nil {
			return nil, err
		}
		method = method0
	}
	if pos >= len(positions) {
		return nil, fmt.Errorf("provider: no logprob position %d (have %d)", pos, len(positions))
	}
	logprobs, all := matchCandidates(positions[pos], req.CandidateTokenIDs, req.CandidateTokenTexts)
	res := &NextTokenScoreResult{
		CandidateLogprobs:    logprobs,
		AllCandidatesPresent: all,
		Distribution:         "raw",
		PromptTokens:         out.Usage.PromptTokens,
		CachedPromptTokens:   cachedTokens(out),
		BackendRequestID:     out.ID,
		ScoringMethod:        method,
		ProbabilitySpace:     space,
		GeneratedText:        out.Choices[0].Text,
	}
	return res, nil
}

// scanAfterCloseTag returns the index of the first position AFTER the
// closing tag appears in the token stream (TODO.md §3.4 <think> handling).
func scanAfterCloseTag(tokens []string, closeTag string) (int, error) {
	if closeTag == "" {
		return 0, nil
	}
	cum := ""
	for i, tok := range tokens {
		cum += tok
		if strings.Contains(cum, closeTag) {
			return i + 1, nil
		}
	}
	return 0, fmt.Errorf("provider: reasoning block never closed (no %q in %d tokens)", closeTag, len(tokens))
}

func cachedTokens(out openaiResponse) int {
	if out.Usage.PromptTokensDetails != nil {
		return out.Usage.PromptTokensDetails.CachedTokens
	}
	return 0
}

// openaiChatResponse is the permissive chat response. Per the OpenAI chat
// contract (and Ollama 0.24.0's compat surface), logprobs sits at the CHOICE
// level: choices[].logprobs.content[].top_logprobs with {token, logprob,
// bytes} entries.
type openaiChatResponse struct {
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Logprobs *struct {
			Content []struct {
				Token       string               `json:"token"`
				Logprob     float64              `json:"logprob"`
				Bytes       json.RawMessage      `json:"bytes"`
				TopLogprobs []openaiLogprobEntry `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	ID string `json:"id"`
}

// scoreViaChat is the chat-completions scoring path for chat-only backends
// (TODO.md §3.4, §5.1 chat variant). The first visible assistant token's
// top_logprobs restore the candidate distribution.
func scoreViaChat(ctx context.Context, hc *httpClient, req NextTokenScoreRequest, model string, extras map[string]any) (*NextTokenScoreResult, error) {
	msgs := req.ChatMessages
	if len(msgs) == 0 && req.PromptText != "" {
		msgs = compile.PromptToMessages(req.PromptText)
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("provider: chat scoring requires messages")
	}
	k := req.TopK
	if k <= 0 {
		k = 20
	}
	maxTokens := 1
	method := "top-k"
	if req.WaitClose {
		maxTokens = req.MaxOutputTokens
		if maxTokens <= 0 {
			maxTokens = 256
		}
		method = "wait-close-tag"
	}
	body := openaiChatRequest{
		Model:       model,
		Messages:    msgs,
		MaxTokens:   maxTokens,
		Temperature: tempOrOne(req.Temperature),
		TopP:        topPOrOne(req.TopP),
		Logprobs:    true,
		TopLogprobs: k,
		Stream:      false,
		Extra:       map[string]any{},
	}
	switch {
	case req.ReasoningField != "":
		// Model-card-documented control wins (e.g. reasoning_effort: "none").
		body.Extra[req.ReasoningField] = req.ReasoningValue
	case req.NoReasoning:
		// Ollama native-style boolean control; harmless where ignored.
		body.Extra["think"] = false
	}
	for k, v := range extras {
		if !protectedFields[k] {
			body.Extra[k] = v
		}
	}
	var out openaiChatResponse
	if err := hc.do(ctx, "POST", "/v1/chat/completions", body, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 || out.Choices[0].Logprobs == nil ||
		len(out.Choices[0].Logprobs.Content) == 0 {
		return nil, fmt.Errorf("provider: no logprobs in chat response")
	}
	content := out.Choices[0].Logprobs.Content
	pos := 0
	if req.WaitClose {
		toks := make([]string, len(content))
		for i, c := range content {
			toks[i] = c.Token
		}
		var scanErr error
		pos, scanErr = scanAfterCloseTag(toks, req.CloseTag)
		if scanErr != nil {
			return nil, scanErr
		}
	}
	if pos >= len(content) {
		return nil, fmt.Errorf("provider: no position after close tag (have %d)", len(content))
	}
	first := content[pos]
	entries := append([]openaiLogprobEntry{}, first.TopLogprobs...)
	// The sampled token itself is an entry when absent from top_logprobs.
	present := false
	for _, e := range entries {
		if e.Token == first.Token {
			present = true
		}
	}
	if !present && first.Token != "" {
		entries = append(entries, openaiLogprobEntry{Token: first.Token, Logprob: first.Logprob, Bytes: first.Bytes})
	}
	logprobs, all := matchCandidates(entries, req.CandidateTokenIDs, req.CandidateTokenTexts)
	res := &NextTokenScoreResult{
		CandidateLogprobs:    logprobs,
		AllCandidatesPresent: all,
		Distribution:         "raw",
		PromptTokens:         out.Usage.PromptTokens,
		BackendRequestID:     out.ID,
		ScoringMethod:        method,
		ProbabilitySpace:     SpaceRaw,
	}
	if out.Usage.PromptTokensDetails != nil {
		res.CachedPromptTokens = out.Usage.PromptTokensDetails.CachedTokens
	}
	res.GeneratedText = out.Choices[0].Message.Content
	return res, nil
}

func promptValue(req NextTokenScoreRequest) any {
	if len(req.PromptTokenIDs) > 0 && req.Endpoint == config.EndpointNativeGenerate {
		return req.PromptTokenIDs
	}
	return req.PromptText
}

func tempOrOne(t float64) float64 {
	if t <= 0 {
		return 1
	}
	return t
}

func topPOrOne(p float64) float64 {
	if p <= 0 || p > 1 {
		return 1
	}
	return p
}

func intptr(i int) *int { return &i }

func logSafe(f float64) float64 {
	if f <= 0 {
		return math.Log(math.SmallestNonzeroFloat64)
	}
	return math.Log(f)
}
