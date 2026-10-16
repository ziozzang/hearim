package jev

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestScoreLegendFormats(t *testing.T) {
	base := ScoreAnswer{Type: "score", Score: 1,
		Legend:        map[string]string{"0": "Calm", "1": "Frustrated", "2": "Very angry"},
		Probabilities: map[string]float64{"0": 0.1, "1": 0.8, "2": 0.1}, Confidence: 0.5}
	cases := map[string]struct{ want, absent string }{
		"":           {`"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"}`, "legend_list"},
		LegendObject: {`"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"}`, "legend_list"},
		LegendArray:  {`"legend":["Calm","Frustrated","Very angry"]`, "legend_list"},
		LegendBoth:   {`"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"legend_list":["Calm","Frustrated","Very angry"]`, ""},
	}
	for format, c := range cases {
		a := base
		a.LegendFormat = format
		// Pointer and value encodings must agree (answers travel as *ScoreAnswer).
		for _, v := range []any{a, &a} {
			b, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			s := string(b)
			if !strings.Contains(s, c.want) || (c.absent != "" && strings.Contains(s, c.absent)) {
				t.Fatalf("format %q: %s", format, s)
			}
			if strings.Contains(s, "LegendFormat") {
				t.Fatalf("format %q leaked LegendFormat: %s", format, s)
			}
		}
	}
}
