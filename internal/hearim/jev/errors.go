package jev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
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
