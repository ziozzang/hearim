package jev

import (
	"strings"
	"testing"
)

func TestImageExtraction(t *testing.T) {
	pr, err := Validate([]byte(`{
	  "model": "m",
	  "state": {"ticket": "screenshot attached", "images": ["https://x/a.png", "https://x/b.jpg"]},
	  "questions": {"q": {"type": "noul"}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Images) != 2 {
		t.Fatalf("images = %v", pr.Images)
	}
}

func TestSingleImageKey(t *testing.T) {
	pr, err := Validate([]byte(`{
	  "model": "m",
	  "state": {"image": "https://x/a.png", "note": "x"},
	  "questions": {"q": {"type": "noul"}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Images) != 1 || pr.Images[0] != "https://x/a.png" {
		t.Errorf("images = %v", pr.Images)
	}
}

func TestDataURLImages(t *testing.T) {
	pr, err := Validate([]byte(`{
	  "model": "m",
	  "state": {"image": "data:image/png;base64,iVBORw0KGgo="},
	  "questions": {"q": {"type": "noul"}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Images) != 1 {
		t.Errorf("images = %v", pr.Images)
	}
}

func TestImageRejections(t *testing.T) {
	cases := []struct{ name, state string }{
		{"local file path", `{"image": "/etc/passwd"}`},
		{"file scheme", `{"image": "file:///etc/passwd"}`},
		{"ftp scheme", `{"image": "ftp://x/a.png"}`},
		{"data not image", `{"image": "data:text/plain;base64,AAAA"}`},
		{"data no comma", `{"image": "data:image/png;base64"}`},
		{"wrong type", `{"image": 42}`},
		{"images not array", `{"images": "https://x/a.png"}`},
		{"array with number", `{"images": [1, 2]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Validate([]byte(`{"model":"m","state":` + tc.state + `,"questions":{"q":{"type":"noul"}}}`))
			if err == nil {
				t.Fatalf("expected rejection for %s", tc.name)
			}
			ve, ok := err.(*RequestValidationError)
			if !ok || len(ve.Details) == 0 || ve.Details[0].Path != "state" {
				t.Errorf("expected state-level detail, got %v", err)
			}
		})
	}
}

func TestImageCountLimit(t *testing.T) {
	urls := make([]string, MaxStateImages+1)
	for i := range urls {
		urls[i] = "https://x/a.png"
	}
	state := `{"images": ["` + strings.Join(urls, `","`) + `"]}`
	_, err := Validate([]byte(`{"model":"m","state":` + state + `,"questions":{"q":{"type":"noul"}}}`))
	if err == nil {
		t.Error("count limit not enforced")
	}
}
