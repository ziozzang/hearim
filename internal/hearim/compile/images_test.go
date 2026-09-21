package compile

import (
	"encoding/json"
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
