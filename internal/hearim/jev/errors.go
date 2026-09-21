package jev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

func newBytesReader(b []byte) *bytes.Reader { return bytes.NewReader(b) }

// objectKeyOrder extracts top-level object keys in their JSON insertion
// order. Choice criteria order must be stable for label mapping (TODO.md
// §6.4); Go maps lose order, so the raw JSON is re-scanned.
func objectKeyOrder(raw json.RawMessage) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("not an object")
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		k, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("object key is not a string")
		}
		keys = append(keys, k)
		var skip any
		if err := dec.Decode(&skip); err != nil {
			return nil, err
		}
	}
	return keys, nil
}

func sortedKeys(m map[string]*string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// extractImages pulls VLM image references out of a state object. Only the
// documented top-level "image" / "images" keys are read; values must be
// http(s):// or data:image/... URLs. file:// and bare paths are rejected —
// state is data, not a license to read arbitrary local files.
func extractImages(obj map[string]any) ([]string, error) {
	var raw []string
	if v, ok := obj["image"]; ok && v != nil {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("state.image must be a string URL")
		}
		raw = append(raw, s)
	}
	if v, ok := obj["images"]; ok && v != nil {
		arr, ok := v.([]any)
		if !ok {
			return nil, fmt.Errorf("state.images must be an array of URL strings")
		}
		for _, e := range arr {
			s, ok := e.(string)
			if !ok {
				return nil, fmt.Errorf("state.images entries must be URL strings")
			}
			raw = append(raw, s)
		}
	}
	if len(raw) == 0 {
		return nil, nil
	}
	if len(raw) > MaxStateImages {
		return nil, fmt.Errorf("state declares %d images; limit is %d", len(raw), MaxStateImages)
	}
	out := make([]string, 0, len(raw))
	seen := map[string]bool{}
	for _, u := range raw {
		if !validImageRef(u) {
			return nil, fmt.Errorf("image reference must be an https?:// or data:image/... URL")
		}
		if seen[u] {
			continue
		}
		seen[u] = true
		out = append(out, u)
	}
	return out, nil
}

func validImageRef(u string) bool {
	if strings.HasPrefix(u, "data:image/") {
		if len(u) > MaxImageDataURLLen {
			return false
		}
		// data:image/<subtype>[;base64],<payload>
		rest := u[len("data:image/"):]
		semi := strings.Index(rest, ";")
		comma := strings.Index(rest, ",")
		if comma < 0 {
			return false
		}
		if semi >= 0 && semi > comma {
			return false
		}
		return true
	}
	parsed, err := url.Parse(u)
	if err != nil || parsed.Host == "" {
		return false
	}
	return parsed.Scheme == "http" || parsed.Scheme == "https"
}

// ErrorCode identifies a Jev-compatible error type (TODO.md §10).
type ErrorCode string

const (
	CodeUnauthorized                  ErrorCode = "unauthorized"
	CodeValidationFailed              ErrorCode = "validation_failed"
	CodeRateLimited                   ErrorCode = "rate_limited"
	CodeBackendOverloaded             ErrorCode = "backend_overloaded"
	CodeBackendUnavailable            ErrorCode = "backend_unavailable"
	CodeBackendProbabilityUnavailable ErrorCode = "backend_probability_unavailable"
	CodeModelNotFound                 ErrorCode = "model_not_found"
	CodeNoExactEvaluationRoute        ErrorCode = "no_exact_evaluation_route"
	CodeInternal                      ErrorCode = "internal_error"
)

// Error is the wire-level error body: {"error": {...}}.
type Error struct {
	Type       ErrorCode         `json:"type"`
	Message    string            `json:"message"`
	QuestionID string            `json:"question_id,omitempty"`
	Details    []ValidationError `json:"details,omitempty"`
}

func (e *Error) Error() string {
	if e.QuestionID != "" {
		return fmt.Sprintf("%s (question %s): %s", e.Type, e.QuestionID, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

// HTTPStatus maps an error code to its response status (TODO.md §10).
func (e *Error) HTTPStatus() int {
	switch e.Type {
	case CodeUnauthorized:
		return http.StatusUnauthorized
	case CodeValidationFailed, CodeModelNotFound:
		return http.StatusUnprocessableEntity
	case CodeRateLimited:
		return http.StatusTooManyRequests
	case CodeBackendOverloaded:
		return 529
	case CodeBackendUnavailable, CodeBackendProbabilityUnavailable, CodeNoExactEvaluationRoute:
		return 529
	default:
		return http.StatusInternalServerError
	}
}

// NewError builds a wire error.
func NewError(code ErrorCode, format string, args ...any) *Error {
	return &Error{Type: code, Message: fmt.Sprintf(format, args...)}
}

// NewQuestionError builds a wire error bound to one question.
func NewQuestionError(code ErrorCode, questionID, format string, args ...any) *Error {
	e := NewError(code, format, args...)
	e.QuestionID = questionID
	return e
}

// FromValidationError converts a validation failure into a 422 wire error.
func FromValidationError(ve *RequestValidationError) *Error {
	return &Error{
		Type:    CodeValidationFailed,
		Message: ve.Message,
		Details: ve.Details,
	}
}
