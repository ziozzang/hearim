// Package canon implements RFC 8785 (JSON Canonicalization Scheme, "JCS")
// serialization with I-JSON input validation (unique object keys).
//
// hearim embeds state, instructions, and criteria into upstream prompts.
// The canonical byte sequence is the prefix-cache key material, so the
// serialization must be deterministic across requests: sorted keys, no
// insignificant whitespace, ES6 number formatting, and JSON.stringify string
// escaping.
package canon

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// ErrNotIJSON reports an input that violates I-JSON constraints, such as
// duplicate object keys.
var ErrNotIJSON = errors.New("canon: input is not valid I-JSON")

// Decode parses data as I-JSON, rejecting duplicate object keys, and returns
// a value tree where numbers are json.Number.
func Decode(data []byte) (any, error) {
	if err := checkIJSON(data); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("canon: decode: %w", err)
	}
	return v, nil
}

// Marshal decodes data as I-JSON and returns its JCS canonical serialization.
func Marshal(data []byte) (string, error) {
	v, err := Decode(data)
	if err != nil {
		return "", err
	}
	return String(v)
}

// String returns the JCS canonical serialization of an already-decoded JSON
// value tree. Supported leaf types: nil, bool, string, json.Number, float64,
// int, int64. Maps must be map[string]any; slices []any.
func String(v any) (string, error) {
	var b strings.Builder
	if err := writeValue(&b, v); err != nil {
		return "", err
	}
	return b.String(), nil
}

func writeValue(b *strings.Builder, v any) error {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		writeString(b, x)
	case json.Number:
		s, err := numberString(string(x))
		if err != nil {
			return err
		}
		b.WriteString(s)
	case float64:
		s, err := Number(x)
		if err != nil {
			return err
		}
		b.WriteString(s)
	case int:
		return writeValue(b, json.Number(strconv.Itoa(x)))
	case int64:
		return writeValue(b, json.Number(strconv.FormatInt(x, 10)))
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool {
			return lessUTF16(keys[i], keys[j])
		})
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeString(b, k)
			b.WriteByte(':')
			if err := writeValue(b, x[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("canon: unsupported value type %T", v)
	}
	return nil
}

// lessUTF16 compares two strings by their UTF-16 code unit sequences, as
// required by RFC 8785 for object key ordering. This differs from UTF-8 byte
// order for astral characters (U+10000..), whose surrogate lead units
// (0xD800..0xDBFF) sort before some BMP characters such as U+E000..U+FFFF.
func lessUTF16(a, b string) bool {
	ua := utf16.Encode([]rune(a))
	ub := utf16.Encode([]rune(b))
	n := len(ua)
	if len(ub) < n {
		n = len(ub)
	}
	for i := 0; i < n; i++ {
		if ua[i] != ub[i] {
			return ua[i] < ub[i]
		}
	}
	return len(ua) < len(ub)
}

const hexDigits = "0123456789abcdef"

func writeString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString("\\\"")
		case '\\':
			b.WriteString("\\\\")
		case '\b':
			b.WriteString("\\b")
		case '\f':
			b.WriteString("\\f")
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		case '\t':
			b.WriteString("\\t")
		default:
			if r < 0x20 {
				b.WriteString("\\u00")
				b.WriteByte(hexDigits[(r>>4)&0xF])
				b.WriteByte(hexDigits[r&0xF])
			} else {
				// Non-ASCII code points are emitted as raw UTF-8 (RFC 8785
				// defers to JSON.stringify, which does not escape them).
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// numberString canonicalizes a JSON number literal through IEEE 754 double
// precision, per RFC 8785.
func numberString(literal string) (string, error) {
	f, err := strconv.ParseFloat(literal, 64)
	if err != nil {
		// Integer literals beyond float64 range (e.g. 1e400) error here too.
		return "", fmt.Errorf("canon: number %q: %w", literal, err)
	}
	return Number(f)
}

// Number formats x per ECMAScript Number::toString (radix 10), which RFC 8785
// references for canonical number serialization. NaN and infinities are not
// representable in JSON and return an error.
func Number(x float64) (string, error) {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return "", fmt.Errorf("canon: number %v is not JSON-serializable", x)
	}
	if x == 0 {
		// Covers +0 and -0; ECMAScript renders both as "0".
		return "0", nil
	}
	var sb strings.Builder
	if x < 0 {
		sb.WriteByte('-')
		x = -x
	}

	// Shortest round-trip digits in scientific form: "d.ddde±XX".
	sci := strconv.FormatFloat(x, 'e', -1, 64)
	mant, expPart, ok := strings.Cut(sci, "e")
	if !ok {
		return "", fmt.Errorf("canon: internal FormatFloat result %q", sci)
	}
	exp, err := strconv.Atoi(expPart)
	if err != nil {
		return "", fmt.Errorf("canon: internal exponent %q: %w", expPart, err)
	}
	digits := strings.Replace(mant, ".", "", 1)
	digits = strings.TrimRight(digits, "0")
	if digits == "" {
		digits = "0"
	}
	// ECMA-262 Number::toString: with sig significant digits s and decimal
	// position pos (value == s × 10^(pos−sig)):
	//   sig ≤ pos ≤ 21  -> digits + (pos−sig) zeros
	//   0 < pos ≤ 21    -> digits[:pos] + "." + digits[pos:]
	//   −6 < pos ≤ 0    -> "0." + (−pos) zeros + digits
	//   otherwise       -> exponential with exponent pos−1
	sig := len(digits)
	pos := exp + 1

	switch {
	case sig <= pos && pos <= 21:
		sb.WriteString(digits)
		sb.WriteString(strings.Repeat("0", pos-sig))
	case 0 < pos && pos <= 21:
		sb.WriteString(digits[:pos])
		sb.WriteByte('.')
		sb.WriteString(digits[pos:])
	case -6 < pos && pos <= 0:
		sb.WriteString("0.")
		sb.WriteString(strings.Repeat("0", -pos))
		sb.WriteString(digits)
	default:
		sb.WriteByte(digits[0])
		if sig > 1 {
			sb.WriteByte('.')
			sb.WriteString(digits[1:])
		}
		sb.WriteByte('e')
		e := pos - 1
		if e >= 0 {
			sb.WriteByte('+')
		} else {
			sb.WriteByte('-')
			e = -e
		}
		sb.WriteString(strconv.Itoa(e))
	}
	return sb.String(), nil
}

// checkIJSON walks the raw JSON and rejects duplicate object keys. It also
// rejects invalid UTF-8, matching encoding/json's replacement behavior being
// the only remaining leniency.
func checkIJSON(data []byte) error {
	if !utf8.Valid(data) {
		return fmt.Errorf("%w: invalid UTF-8", ErrNotIJSON)
	}
	if !json.Valid(data) {
		// Rejects trailing garbage and malformed values alike.
		return fmt.Errorf("%w: not a single valid JSON value", ErrNotIJSON)
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	var walk func() error
	walk = func() error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case json.Delim:
			switch t {
			case '{':
				seen := make(map[string]struct{})
				for dec.More() {
					keyTok, err := dec.Token()
					if err != nil {
						return err
					}
					key, ok := keyTok.(string)
					if !ok {
						return fmt.Errorf("canon: object key is not a string")
					}
					if _, dup := seen[key]; dup {
						return fmt.Errorf("%w: duplicate object key %q", ErrNotIJSON, key)
					}
					seen[key] = struct{}{}
					if err := walk(); err != nil {
						return err
					}
				}
				if _, err := dec.Token(); err != nil { // consume '}'
					return err
				}
			case '[':
				for dec.More() {
					if err := walk(); err != nil {
						return err
					}
				}
				if _, err := dec.Token(); err != nil { // consume ']'
					return err
				}
			}
		}
		return nil
	}
	if err := walk(); err != nil {
		return fmt.Errorf("%w: %v", ErrNotIJSON, err)
	}
	return nil
}
