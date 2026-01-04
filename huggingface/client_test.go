package huggingface

import (
	"encoding/json"
	"testing"
)

func TestSplitsResponseParsing(t *testing.T) {
	jsonData := `{"splits":[{"dataset":"test/dataset","config":"default","split":"train"},{"dataset":"test/dataset","config":"default","split":"test"}]}`

	var resp SplitsResponse
	if err := json.Unmarshal([]byte(jsonData), &resp); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if len(resp.Splits) != 2 {
		t.Errorf("expected 2 splits, got %d", len(resp.Splits))
	}

	if resp.Splits[0].Config != "default" {
		t.Errorf("expected config 'default', got %s", resp.Splits[0].Config)
	}
}

func TestRowsResponseParsing(t *testing.T) {
	jsonData := `{"rows":[{"row_idx":0,"row":{"text":"hello world","label":1}},{"row_idx":1,"row":{"text":"goodbye","label":0}}]}`

	var resp RowsResponse
	if err := json.Unmarshal([]byte(jsonData), &resp); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if len(resp.Rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(resp.Rows))
	}

	if resp.Rows[0].Row["text"] != "hello world" {
		t.Errorf("expected 'hello world', got %v", resp.Rows[0].Row["text"])
	}
}

func TestExtractTextsFromRows(t *testing.T) {
	rows := &RowsResponse{
		Rows: []RowWrapper{
			{RowIdx: 0, Row: map[string]interface{}{"text": "first", "label": 1}},
			{RowIdx: 1, Row: map[string]interface{}{"text": "second", "label": 0}},
			{RowIdx: 2, Row: map[string]interface{}{"text": "", "label": 0}},
			{RowIdx: 3, Row: map[string]interface{}{"other": "value"}},
		},
	}

	var texts []string
	column := "text"
	for _, wrapper := range rows.Rows {
		if val, ok := wrapper.Row[column]; ok {
			if text, ok := val.(string); ok && text != "" {
				texts = append(texts, text)
			}
		}
	}

	if len(texts) != 2 {
		t.Errorf("expected 2 texts, got %d", len(texts))
	}

	if texts[0] != "first" || texts[1] != "second" {
		t.Errorf("unexpected texts: %v", texts)
	}
}

func TestNewClient(t *testing.T) {
	client := NewClient()
	if client == nil {
		t.Error("expected non-nil client")
	}
	if client.httpClient == nil {
		t.Error("expected non-nil http client")
	}
}
