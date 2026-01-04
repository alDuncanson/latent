package dataimport

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type TextWithVector struct {
	Text   string
	Vector []float32
}

type jsonTextObject struct {
	Text   string    `json:"text"`
	Vector []float32 `json:"vector,omitempty"`
}

func LoadTexts(path string) ([]string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".csv":
		return loadCSV(path)
	case ".json":
		return loadJSON(path)
	default:
		return nil, fmt.Errorf("unsupported file extension: %s", ext)
	}
}

func LoadWithVectors(path string) ([]TextWithVector, error) {
	if strings.ToLower(filepath.Ext(path)) != ".json" {
		return nil, fmt.Errorf("LoadWithVectors only supports JSON files")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}

	var objects []jsonTextObject
	if err := json.Unmarshal(data, &objects); err != nil {
		return nil, fmt.Errorf("parsing JSON: %w", err)
	}

	results := make([]TextWithVector, 0, len(objects))
	for i, obj := range objects {
		if obj.Text == "" {
			return nil, fmt.Errorf("entry %d missing text field", i)
		}
		if len(obj.Vector) == 0 {
			return nil, fmt.Errorf("entry %d missing vector field", i)
		}
		results = append(results, TextWithVector{
			Text:   obj.Text,
			Vector: obj.Vector,
		})
	}

	return results, nil
}

func loadCSV(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening CSV file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("reading CSV: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("CSV file is empty")
	}

	textCol := -1
	for i, header := range records[0] {
		if strings.EqualFold(strings.TrimSpace(header), "text") {
			textCol = i
			break
		}
	}

	if textCol == -1 {
		return nil, fmt.Errorf("CSV missing 'text' column header")
	}

	texts := make([]string, 0, len(records)-1)
	for _, row := range records[1:] {
		if textCol < len(row) && row[textCol] != "" {
			texts = append(texts, row[textCol])
		}
	}

	return texts, nil
}

func loadJSON(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading JSON file: %w", err)
	}

	var stringArray []string
	if err := json.Unmarshal(data, &stringArray); err == nil {
		return stringArray, nil
	}

	var objectArray []jsonTextObject
	if err := json.Unmarshal(data, &objectArray); err != nil {
		return nil, fmt.Errorf("parsing JSON: expected array of strings or objects with 'text' field: %w", err)
	}

	texts := make([]string, 0, len(objectArray))
	for i, obj := range objectArray {
		if obj.Text == "" {
			return nil, fmt.Errorf("entry %d missing text field", i)
		}
		texts = append(texts, obj.Text)
	}

	return texts, nil
}
