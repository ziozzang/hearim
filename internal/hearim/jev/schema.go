// Package jev defines the TypeSafe AI Jev System One request/response
// contract (TODO.md §2), semantic validation, and the compatible error
// surface (TODO.md §10).
package jev

import (
	"encoding/json"
	"fmt"
)

// QuestionType is one of the three System One primitives.
type QuestionType string

const (
	TypeChoice QuestionType = "choice"
	TypeScore  QuestionType = "score"
	TypeNoul   QuestionType = "noul"
)

// Limits from the official Jev specification (TODO.md §2).
const (
	MaxChoiceOptions = 255 // documented maximum for choice criteria
	MinCriteria      = 2
	MaxScoreLevels   = 10
)

// Request is the top-level POST /v1/systemone body.
type Request struct {
	Model     string               `json:"model"`
	State     json.RawMessage      `json:"state"`
	Questions map[string]*Question `json:"questions"`
}

// Question is a single typed evaluation. ID is the map key in Questions and
// is never embedded in the model prompt (TODO.md §2: ids are correlation
// keys, not evaluation content).
type Question struct {
	Type         QuestionType    `json:"type"`
	Instructions json.RawMessage `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria"`
}

// RequestValidationError carries per-question detail for 422 responses.
type RequestValidationError struct {
	Message string            `json:"message"`
	Details []ValidationError `json:"details,omitempty"`
}

// ValidationError points at one question (or the top level when QuestionID is
// empty).
type ValidationError struct {
	Path    string `json:"path,omitempty"`
	Message string `json:"message"`
}

func (e *RequestValidationError) Error() string { return e.Message }

// ParsedQuestion is the validated, Go-native form of a Question.
type ParsedQuestion struct {
	ID              string
	Type            QuestionType
	Instructions    any // string | map | slice, already canonicalizable
	HasInstructions bool
	// Choice fields (insertion order preserved).
	ChoiceNames        []string
	ChoiceDescriptions map[string]*string // nil value = null description
	// Score fields: 2..10 level descriptions, low to high.
	Levels []string
	// Noul fields; nil pointer = absent.
	TrueDesc  *string
	FalseDesc *string
}

// ParsedRequest is the validated form of a Request with canonical state.
type ParsedRequest struct {
	Model     string
	StateRaw  json.RawMessage
	State     any
	Questions []*ParsedQuestion
}

// UnmarshalJSON keeps raw criteria and validates nothing here; use Validate.
func (q *Question) UnmarshalJSON(data []byte) error {
	type alias Question
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*q = Question(a)
	return nil
}

// Validate parses and semantically validates a raw request body.
func Validate(raw []byte) (*ParsedRequest, error) {
	var req Request
	dec := json.NewDecoder(newBytesReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		return nil, &RequestValidationError{Message: fmt.Sprintf("invalid request body: %v", err)}
	}
	return ValidateRequest(&req)
}

// ValidateRequest validates an already-decoded request.
func ValidateRequest(req *Request) (*ParsedRequest, error) {
	ve := &RequestValidationError{}
	fail := func(path, format string, args ...any) {
		ve.Details = append(ve.Details, ValidationError{Path: path, Message: fmt.Sprintf(format, args...)})
	}

	if req.Model == "" {
		fail("model", "model is required")
	}

	out := &ParsedRequest{Model: req.Model, StateRaw: req.State}

	if len(req.State) == 0 {
		fail("state", "state is required")
	} else {
		var s any
		if err := json.Unmarshal(req.State, &s); err != nil {
			fail("state", "state must be valid JSON: %v", err)
		} else {
			switch s.(type) {
			case string, map[string]any, []any:
				out.State = s
			default:
				fail("state", "state must be a string, object, or array")
			}
		}
	}

	if len(req.Questions) == 0 {
		fail("questions", "at least one question is required")
	}

	for id, q := range req.Questions {
		path := "questions." + id
		if q == nil {
			fail(path, "question must be an object")
			continue
		}
		if id == "" {
			fail(path, "question id must be a non-empty string")
		}
		pq := &ParsedQuestion{ID: id, Type: q.Type}
		if q.Instructions != nil {
			var instr any
			if err := json.Unmarshal(q.Instructions, &instr); err != nil {
				fail(path+".instructions", "instructions must be valid JSON")
			} else {
				switch instr.(type) {
				case string, map[string]any, []any:
					pq.Instructions = instr
					pq.HasInstructions = true
				default:
					fail(path+".instructions", "instructions must be a string, object, or array")
				}
			}
		}

		switch q.Type {
		case TypeChoice:
			pq.ChoiceDescriptions = map[string]*string{}
			var criteria map[string]*string
			if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
				fail(path+".criteria", "choice criteria must be an object mapping option name to description or null")
				continue
			}
			if len(criteria) < MinCriteria || len(criteria) > MaxChoiceOptions {
				fail(path+".criteria", "choice criteria must contain 2 to 255 options, got %d", len(criteria))
				continue
			}
			// Preserve insertion order from the raw JSON (TODO.md §6.4).
			names, err := objectKeyOrder(q.Criteria)
			if err != nil || len(names) != len(criteria) {
				names = sortedKeys(criteria)
			}
			pq.ChoiceNames = names
			for _, n := range names {
				if n == "" {
					fail(path+".criteria", "option name must be non-empty")
				}
				pq.ChoiceDescriptions[n] = criteria[n]
			}
		case TypeScore:
			var levels []string
			if err := json.Unmarshal(q.Criteria, &levels); err != nil {
				fail(path+".criteria", "score criteria must be an array of level descriptions ordered low to high")
				continue
			}
			if len(levels) < MinCriteria || len(levels) > MaxScoreLevels {
				fail(path+".criteria", "score criteria must contain 2 to 10 levels, got %d", len(levels))
				continue
			}
			for i, lv := range levels {
				if lv == "" {
					fail(path+".criteria", "level %d description must be non-empty", i)
				}
			}
			pq.Levels = levels
		case TypeNoul:
			if len(q.Criteria) > 0 {
				var crit struct {
					True  *string `json:"true"`
					False *string `json:"false"`
				}
				if err := json.Unmarshal(q.Criteria, &crit); err != nil {
					fail(path+".criteria", "noul criteria must be an object with optional true/false descriptions")
					continue
				}
				pq.TrueDesc, pq.FalseDesc = crit.True, crit.False
			}
		default:
			fail(path+".type", "type must be one of choice, score, noul")
			continue
		}
		out.Questions = append(out.Questions, pq)
	}

	if len(ve.Details) > 0 {
		ve.Message = "request validation failed"
		return nil, ve
	}
	return out, nil
}

// Response is the Jev-compatible top-level response body.
type Response struct {
	Model   string         `json:"model"`
	Answers map[string]any `json:"answers"`
	Usage   Usage          `json:"usage"`
	// ID correlates with the request when the client supplied one
	// (non-standard, OpenAI-style convenience; empty when absent).
	ID string `json:"id,omitempty"`
}

// Usage aggregates fan-out upstream token consumption (TODO.md §8.5).
type Usage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// ChoiceAnswer is the answer payload for choice questions.
type ChoiceAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

// ScoreAnswer is the answer payload for score questions.
type ScoreAnswer struct {
	Type          string             `json:"type"`
	Score         float64            `json:"score"`
	Legend        []string           `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

// NoulAnswer is the answer payload for noul questions.
type NoulAnswer struct {
	Type string  `json:"type"`
	Noul float64 `json:"noul"`
}
