// Package embedding defines the interface for text embedding providers.
// It allows the application to use different embedding backends (Ollama, Hugging Face, etc.)
// interchangeably.
package embedding

// Embedder is the interface that text embedding providers must implement.
type Embedder interface {
	// Embed converts the provided text into a vector embedding.
	// It returns a slice of float32 values representing the text in embedding space,
	// or an error if the embedding request fails.
	// If the input text is empty, Embed should return nil without error.
	Embed(text string) ([]float32, error)
}
