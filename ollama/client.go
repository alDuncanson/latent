// Package ollama provides an HTTP client for interacting with the Ollama API.
// It specifically handles text embedding requests, converting text strings into
// high-dimensional vector representations using Ollama's embedding models.
package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

// Client handles HTTP communication with the Ollama embedding API.
// It maintains the connection configuration and reuses an HTTP client
// for efficient request handling.
type Client struct {
	baseURL    string       // The base URL of the Ollama server (e.g., "http://localhost:11434")
	modelName  string       // The name of the embedding model to use (e.g., "nomic-embed-text")
	httpClient *http.Client // Reusable HTTP client for making requests
}

// embeddingRequest represents the JSON payload sent to the Ollama /api/embed endpoint.
type embeddingRequest struct {
	Model string `json:"model"` // The model identifier to use for embedding
	Input string `json:"input"` // The text content to be embedded
}

// embeddingResponse represents the JSON response from the Ollama /api/embed endpoint.
// Embeddings are returned as a slice of slices to support batch embedding requests,
// though this client currently only uses single-text embedding.
type embeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"` // Array of embedding vectors
}

// NewClient creates a new Ollama client configured to connect to the specified
// server and use the given embedding model.
func NewClient(baseURL, modelName string) *Client {
	return &Client{
		baseURL:    baseURL,
		modelName:  modelName,
		httpClient: &http.Client{},
	}
}

// Embed converts the provided text into a vector embedding using the Ollama API.
// It returns a slice of float32 values representing the text in embedding space,
// or an error if the embedding request fails.
//
// If the input text is empty, Embed returns nil without making an API request.
func (ollamaClient *Client) Embed(inputText string) ([]float32, error) {
	// Skip API call for empty input text
	if inputText == "" {
		return nil, nil
	}

	// Construct the embedding request payload
	requestPayload := embeddingRequest{
		Model: ollamaClient.modelName,
		Input: inputText,
	}

	// Serialize the request payload to JSON
	jsonRequestBody, marshalError := json.Marshal(requestPayload)
	if marshalError != nil {
		return nil, fmt.Errorf("marshal request: %w", marshalError)
	}

	// Send the embedding request to the Ollama API
	embeddingEndpointURL := ollamaClient.baseURL + "/api/embed"
	httpResponse, postError := ollamaClient.httpClient.Post(
		embeddingEndpointURL,
		"application/json",
		bytes.NewReader(jsonRequestBody),
	)
	if postError != nil {
		return nil, fmt.Errorf("post request: %w", postError)
	}
	defer httpResponse.Body.Close()

	// Verify the API returned a successful status code
	if httpResponse.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", httpResponse.StatusCode)
	}

	// Deserialize the JSON response into the embedding response structure
	var parsedResponse embeddingResponse
	responseDecoder := json.NewDecoder(httpResponse.Body)
	if decodeError := responseDecoder.Decode(&parsedResponse); decodeError != nil {
		return nil, fmt.Errorf("decode response: %w", decodeError)
	}

	// Validate that the response contains at least one embedding vector
	if len(parsedResponse.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Return the first embedding vector (we only requested one)
	return parsedResponse.Embeddings[0], nil
}
