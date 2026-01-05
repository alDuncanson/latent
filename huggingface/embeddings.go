package huggingface

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

const inferenceAPIBaseURL = "https://api-inference.huggingface.co"

// EmbeddingsClient handles HTTP communication with the Hugging Face Inference API
// for generating text embeddings.
type EmbeddingsClient struct {
	modelID    string
	token      string
	httpClient *http.Client
}

// embeddingsRequest represents the JSON payload sent to the HF Inference API.
type embeddingsRequest struct {
	Inputs  string            `json:"inputs"`
	Options map[string]bool   `json:"options,omitempty"`
}

// NewEmbeddingsClient creates a new Hugging Face embeddings client.
// If token is empty, it will attempt to read from HF_TOKEN environment variable.
func NewEmbeddingsClient(modelID, token string) *EmbeddingsClient {
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	return &EmbeddingsClient{
		modelID:    modelID,
		token:      token,
		httpClient: &http.Client{},
	}
}

// Embed converts the provided text into a vector embedding using the Hugging Face Inference API.
// It returns a slice of float32 values representing the text in embedding space,
// or an error if the embedding request fails.
func (c *EmbeddingsClient) Embed(inputText string) ([]float32, error) {
	if inputText == "" {
		return nil, nil
	}

	requestPayload := embeddingsRequest{
		Inputs:  inputText,
		Options: map[string]bool{"wait_for_model": true},
	}

	jsonBody, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/pipeline/feature-extraction/%s", inferenceAPIBaseURL, c.modelID)
	req, err := http.NewRequest("POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&errorBody)
		return nil, fmt.Errorf("API error %d: %v", resp.StatusCode, errorBody)
	}

	// The response is a nested array: [[float, float, ...]]
	// For single input, we get back [[embedding values]]
	var response [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(response) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return response[0], nil
}
