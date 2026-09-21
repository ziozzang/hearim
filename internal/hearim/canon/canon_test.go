package canon

import (
	"encoding/json"
	"testing"
)

func TestNumberES6(t *testing.T) {
	cases := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{-0, "0"},
		{7, "7"},
		{-7, "-7"},
		{0.25, "0.25"},
		{0.399, "0.399"},
		{3.141592653589793, "3.141592653589793"},
		{1e-6, "0.000001"},
		{2.35e-6, "0.00000235"},
		{1e-7, "1e-7"},
		{1e20, "100000000000000000000"},
		{1e21, "1e+21"},
		{23456789012e5, "2345678901200000"}, // RFC 8785 §3.2.2.3
		{333333333.33333329, "333333333.3333333"},
		{9007199254740992, "9007199254740992"},
		{123.456, "123.456"},
		{5.5, "5.5"},
	}
	for _, c := range cases {
		got, err := Number(c.in)
		if err != nil {
			t.Errorf("Number(%v): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("Number(%v) = %q, want %q", c.in, got, c.want)
		}
		// Round-trip: the canonical string must parse back to the same double.
		back, err := json.Number(got).Float64()
		if err != nil {
			t.Errorf("canonical %q unparseable: %v", got, err)
		} else if back != c.in {
			t.Errorf("canonical %q round-trips to %v, want %v", got, back, c.in)
		}
	}
}

func TestNumberRejectsNonFinite(t *testing.T) {
	for _, x := range []float64{nan(), inf(1), inf(-1)} {
		if _, err := Number(x); err == nil {
			t.Errorf("Number(%v) should error", x)
		}
	}
}

func TestKeyOrderIsUTF16(t *testing.T) {
	// U+1D11E (𝄞) has UTF-16 lead unit 0xD834, which sorts before U+FB00 (ﬀ),
	// while UTF-8 byte order would sort ﬀ first. JCS must use UTF-16 order.
	in := []byte(`{"ﬀ": 1, "𝄞": 2}`)
	got, err := Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"𝄞":2,"ﬀ":1}`
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestCanonicalStructuralExample(t *testing.T) {
	// Same value expressed with different key order and whitespace must
	// canonicalize identically.
	a := `{"ticket":"link missing","account_age_days":820,"nested":{"z":1,"a":[true,null,{"k":"v"}]}}`
	b := "\t{\n  \"nested\" : {\"a\":[true,null,{\"k\":\"v\"}], \"z\":1},\n" +
		"  \"account_age_days\": 820, \"ticket\":\"link missing\"\n}\n"
	ca, err := Marshal([]byte(a))
	if err != nil {
		t.Fatal(err)
	}
	cb, err := Marshal([]byte(b))
	if err != nil {
		t.Fatal(err)
	}
	if ca != cb {
		t.Errorf("canonical forms differ:\n%s\n%s", ca, cb)
	}
	want := `{"account_age_days":820,"nested":{"a":[true,null,{"k":"v"}],"z":1},"ticket":"link missing"}`
	if ca != want {
		t.Errorf("got  %s\nwant %s", ca, want)
	}
}

func TestStringEscaping(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", `"plain"`},
		{"quote\"back\\slash", `"quote\"back\\slash"`},
		{"tab\tnl\n", `"tab\tnl\n"`},
		{"\u0001\u001f", `"\u0001\u001f"`},
		{"한글 𝄞", `"한글 𝄞"`}, // non-ASCII stays raw
		{"</state>", `"</state>"`},
	}
	for _, c := range cases {
		got, err := String(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("String(%q) = %s, want %s", c.in, got, c.want)
		}
	}
}

func TestDuplicateKeysRejected(t *testing.T) {
	if _, err := Marshal([]byte(`{"a":1,"a":2}`)); err == nil {
		t.Error("duplicate top-level key not rejected")
	}
	if _, err := Marshal([]byte(`{"x":{"a":1,"a":2}}`)); err == nil {
		t.Error("duplicate nested key not rejected")
	}
}

func TestTrailingGarbageRejected(t *testing.T) {
	if _, err := Marshal([]byte(`{"a":1} extra`)); err == nil {
		t.Error("trailing content not rejected")
	}
}

func TestAcceptsGoNativeValues(t *testing.T) {
	v := map[string]any{
		"n":   42,
		"f":   1.5,
		"s":   "x",
		"ok":  true,
		"arr": []any{json.Number("0.25"), nil},
	}
	got, err := String(v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"arr":[0.25,null],"f":1.5,"n":42,"ok":true,"s":"x"}`
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func nan() float64 {
	v := 0.0
	return v / v * 0 // preserve NaN without constant folding complaints
}

func inf(sign int) float64 {
	v := 1.0
	if sign < 0 {
		v = -1.0
	}
	return v / zero()
}

func zero() float64 { return 0.0 }
