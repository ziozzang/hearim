package compile

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPlanCarriesImagesAndMultimodalChat(t *testing.T) {
	plan, _ := mustCompile(t, `{
	  "model": "m",
	  "state": {"ticket": "what is in this picture", "images": ["https://x/a.png", "data:image/png;base64,iVBORw0KGgo="]},
	  "questions": {"q": {"type": "noul"}}
	}`)
	if len(plan.Images) != 2 {
		t.Fatalf("plan images = %v", plan.Images)
	}
	msgs := plan.ChatMessages(plan.Questions[0])
	if len(msgs) != 3 {
		t.Fatalf("messages = %d", len(msgs))
	}
	parts, ok := msgs[1].Content.([]map[string]any)
	if !ok {
		t.Fatalf("state message content should be multimodal parts, got %T", msgs[1].Content)
	}
	if len(parts) != 3 { // 1 text + 2 images
		t.Fatalf("parts = %d", len(parts))
	}
	if parts[0]["type"] != "text" {
		t.Errorf("first part = %v", parts[0])
	}
	if parts[1]["type"] != "image_url" {
		t.Errorf("image part = %v", parts[1])
	}
	iu := parts[1]["image_url"].(map[string]string)
	if iu["url"] != "https://x/a.png" {
		t.Errorf("image url = %v", iu)
	}
	// The rendered request must survive JSON marshaling (chat wire format).
	b, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) == "" {
		t.Fatal("empty render")
	}
}

func TestNoImagesKeepsStringContent(t *testing.T) {
	plan, _ := mustCompile(t, choiceReq)
	msgs := plan.ChatMessages(plan.Questions[0])
	if _, ok := msgs[1].Content.(string); !ok {
		t.Errorf("state content should stay a string without images, got %T", msgs[1].Content)
	}
}

func TestImagePartsHelper(t *testing.T) {
	parts := ImageParts([]string{"https://a", "data:image/jpeg;base64,AA"})
	if len(parts) != 2 {
		t.Fatalf("parts = %d", len(parts))
	}
	for _, p := range parts {
		if p["type"] != "image_url" {
			t.Errorf("part = %v", p)
		}
	}
}

func TestStateTextRedactsImagePayloads(t *testing.T) {
	dataURL := "data:image/png;base64," + strings.Repeat("QUJD", 100) // 400 chars of payload
	plan, pr := mustCompile(t, `{
	  "model": "m",
	  "state": {"image": "`+dataURL+`", "note": "keep"},
	  "questions": {"q": {"type": "noul"}}
	}`)
	full := plan.Prompt(plan.Questions[0])
	if strings.Contains(full, "QUJD") {
		t.Error("base64 payload leaked into the state text block")
	}
	if !strings.Contains(full, "<image:1>") {
		t.Errorf("placeholder missing:\n%s", full)
	}
	if !strings.Contains(full, "keep") {
		t.Error("non-image state content lost")
	}
	// Identity stays on the unredacted form.
	if len(pr.Images) != 1 || pr.Images[0] != dataURL {
		t.Errorf("parsed images = %v", pr.Images)
	}
	if plan.CanonicalState != `{"image":"`+dataURL+`","note":"keep"}` {
		t.Errorf("canonical state must keep the full form: %s", plan.CanonicalState)
	}
	if !strings.Contains(full, "\"note\":\"keep\"") {
		t.Error("state JSON structure broken")
	}
}

func TestNoImagesKeepsStateVerbatim(t *testing.T) {
	plan, _ := mustCompile(t, choiceReq)
	if !strings.Contains(plan.Prompt(plan.Questions[0]), `"ticket":"결제 후 다운로드 링크를 받지 못했습니다."`) {
		t.Error("text-only state must render verbatim")
	}
}
