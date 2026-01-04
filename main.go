// Package main provides the entry point for latent, a terminal UI application
// for visualizing text embeddings. It connects to Ollama for generating embeddings
// and Qdrant for vector storage, then projects high-dimensional vectors to 2D
// using PCA for interactive visualization.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/alDuncanson/latent/dataimport"
	"github.com/alDuncanson/latent/ollama"
	"github.com/alDuncanson/latent/preload"
	"github.com/alDuncanson/latent/qdrant"
	"github.com/alDuncanson/latent/tui"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/uuid"
)

// version is set at build time via ldflags, defaults to "dev" for local builds
var version = "dev"

// Service configuration constants for connecting to backend services
const (
	// ollamaServiceURL is the HTTP endpoint for the Ollama embedding service
	ollamaServiceURL = "http://localhost:11434"

	// embeddingModelName specifies which Ollama model to use for text embeddings
	embeddingModelName = "nomic-embed-text"

	// qdrantServiceAddress is the gRPC endpoint for the Qdrant vector database
	qdrantServiceAddress = "localhost:6334"

	// vectorCollectionName is the Qdrant collection where embeddings are stored
	vectorCollectionName = "embeddings"

	// embeddingVectorDimensions is the size of vectors produced by nomic-embed-text
	embeddingVectorDimensions = 768
)

func main() {
	// Parse command-line flags for version display and demo data preloading
	showVersionFlag := flag.Bool("version", false, "print version and exit")
	preloadDemoDataFlag := flag.Bool("preload", false, "seed with demo word list")
	flag.Parse()

	// Handle version flag: print version and exit early
	if *showVersionFlag {
		fmt.Println(version)
		return
	}

	// Check for positional argument (dataset file to import)
	var datasetPath string
	if flag.NArg() > 0 {
		datasetPath = flag.Arg(0)
	}

	// Initialize the Ollama client for generating text embeddings
	ollamaEmbeddingClient := ollama.NewClient(ollamaServiceURL, embeddingModelName)

	// Initialize the Qdrant client for vector storage and retrieval
	qdrantVectorClient, connectionError := qdrant.NewClient(
		qdrantServiceAddress,
		vectorCollectionName,
		embeddingVectorDimensions,
	)
	if connectionError != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to Qdrant: %v\n", connectionError)
		fmt.Fprintln(os.Stderr, "Make sure Qdrant is running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
		os.Exit(1)
	}
	defer qdrantVectorClient.Close()

	// If preload flag is set, seed the database with demo words before starting the UI
	if *preloadDemoDataFlag {
		preloadError := runPreloadDemoWords(ollamaEmbeddingClient, qdrantVectorClient)
		if preloadError != nil {
			fmt.Fprintf(os.Stderr, "Preload failed: %v\n", preloadError)
			os.Exit(1)
		}
	}

	// If a dataset path was provided, import it
	if datasetPath != "" {
		importError := runImportDataset(ollamaEmbeddingClient, qdrantVectorClient, datasetPath)
		if importError != nil {
			fmt.Fprintf(os.Stderr, "Import failed: %v\n", importError)
			os.Exit(1)
		}
	}

	// Create and run the terminal user interface
	terminalUserInterfaceModel := tui.NewModel(ollamaEmbeddingClient, qdrantVectorClient, version)
	bubbleTeaProgram := tea.NewProgram(terminalUserInterfaceModel, tea.WithAltScreen())

	_, programRunError := bubbleTeaProgram.Run()
	if programRunError != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", programRunError)
		os.Exit(1)
	}
}

// runPreloadDemoWords seeds the Qdrant database with a predefined list of demo words.
// It generates embeddings for each word using Ollama and stores them in Qdrant.
// Progress is displayed to stdout as each word is processed.
func runPreloadDemoWords(ollamaEmbeddingClient *ollama.Client, qdrantVectorClient *qdrant.Client) error {
	demoWordList := preload.Words()
	backgroundContext := context.Background()

	fmt.Printf("Preloading %d words...\n", len(demoWordList))

	// Process each word: generate embedding and store in vector database
	for wordIndex, currentWord := range demoWordList {
		// Generate the embedding vector for the current word
		embeddingVector, embeddingError := ollamaEmbeddingClient.Embed(currentWord)
		if embeddingError != nil {
			return fmt.Errorf("embed %q: %w", currentWord, embeddingError)
		}

		// Store the word and its embedding in Qdrant with a unique identifier
		uniquePointIdentifier := uuid.New().String()
		upsertError := qdrantVectorClient.Upsert(backgroundContext, uniquePointIdentifier, currentWord, embeddingVector)
		if upsertError != nil {
			return fmt.Errorf("upsert %q: %w", currentWord, upsertError)
		}

		// Display progress on the same line using carriage return
		fmt.Printf("\r[%d/%d] %s", wordIndex+1, len(demoWordList), currentWord)
	}

	fmt.Println("\nDone.")
	return nil
}

// runImportDataset loads texts from a CSV or JSON file and embeds them into Qdrant.
func runImportDataset(ollamaEmbeddingClient *ollama.Client, qdrantVectorClient *qdrant.Client, datasetPath string) error {
	texts, loadError := dataimport.LoadTexts(datasetPath)
	if loadError != nil {
		return fmt.Errorf("loading dataset: %w", loadError)
	}

	if len(texts) == 0 {
		return fmt.Errorf("no texts found in dataset")
	}

	backgroundContext := context.Background()
	fmt.Printf("Importing %d texts from %s...\n", len(texts), datasetPath)

	for textIndex, currentText := range texts {
		embeddingVector, embeddingError := ollamaEmbeddingClient.Embed(currentText)
		if embeddingError != nil {
			return fmt.Errorf("embed %q: %w", currentText, embeddingError)
		}

		uniquePointIdentifier := uuid.New().String()
		upsertError := qdrantVectorClient.Upsert(backgroundContext, uniquePointIdentifier, currentText, embeddingVector)
		if upsertError != nil {
			return fmt.Errorf("upsert %q: %w", currentText, upsertError)
		}

		fmt.Printf("\r[%d/%d] %s", textIndex+1, len(texts), truncateForProgress(currentText, 40))
	}

	fmt.Println("\nDone.")
	return nil
}

func truncateForProgress(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
}
