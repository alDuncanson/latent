// Package main provides the entry point for latent, a terminal UI application
// for visualizing text embeddings. It connects to Ollama or Hugging Face for
// generating embeddings and Qdrant for vector storage, then projects
// high-dimensional vectors to 2D using PCA for interactive visualization.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/alDuncanson/latent/dataimport"
	"github.com/alDuncanson/latent/embedding"
	"github.com/alDuncanson/latent/huggingface"
	"github.com/alDuncanson/latent/ollama"
	"github.com/alDuncanson/latent/preload"
	"github.com/alDuncanson/latent/qdrant"
	"github.com/alDuncanson/latent/tui"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/uuid"
)

// version is set at build time via ldflags, defaults to "dev" for local builds
var version = "dev"

// Default service configuration constants
const (
	ollamaServiceURL     = "http://localhost:11434"
	defaultOllamaModel   = "nomic-embed-text"
	defaultHFModel       = "sentence-transformers/all-MiniLM-L6-v2"
	qdrantServiceAddress = "localhost:6334"
	vectorCollectionName = "embeddings"
)

func main() {
	// Parse command-line flags
	showVersionFlag := flag.Bool("version", false, "print version and exit")
	preloadDemoDataFlag := flag.Bool("preload", false, "seed with demo word list")
	hfDatasetFlag := flag.String("hf-dataset", "", "Hugging Face dataset to import (e.g., cornell-movie-review-data/rotten_tomatoes)")
	hfSplitFlag := flag.String("hf-split", "train", "dataset split to use (default: train)")
	hfColumnFlag := flag.String("hf-column", "text", "column containing text to embed (default: text)")
	hfMaxRowsFlag := flag.Int("hf-max-rows", 100, "maximum rows to fetch from Hugging Face (default: 100)")

	// Embedder configuration flags
	embedderFlag := flag.String("embedder", "ollama", "embedding provider: ollama or huggingface")
	modelFlag := flag.String("model", "", "model name (default: nomic-embed-text for ollama, sentence-transformers/all-MiniLM-L6-v2 for huggingface)")
	embeddingDimFlag := flag.Int("embedding-dim", 768, "embedding vector dimensions (must match model output)")
	hfTokenFlag := flag.String("hf-token", "", "Hugging Face API token (or set HF_TOKEN env var)")

	flag.Parse()

	if *showVersionFlag {
		fmt.Println(version)
		return
	}

	// Check for positional argument (dataset file to import)
	var datasetPath string
	if flag.NArg() > 0 {
		datasetPath = flag.Arg(0)
	}

	// Initialize the embedder based on the selected provider
	var embedder embedding.Embedder
	switch *embedderFlag {
	case "ollama":
		modelName := *modelFlag
		if modelName == "" {
			modelName = defaultOllamaModel
		}
		embedder = ollama.NewClient(ollamaServiceURL, modelName)
	case "huggingface", "hf":
		modelName := *modelFlag
		if modelName == "" {
			modelName = defaultHFModel
		}
		embedder = huggingface.NewEmbeddingsClient(modelName, *hfTokenFlag)
	default:
		fmt.Fprintf(os.Stderr, "Unknown embedder: %s (use 'ollama' or 'huggingface')\n", *embedderFlag)
		os.Exit(1)
	}

	// Initialize the Qdrant client for vector storage and retrieval
	qdrantVectorClient, connectionError := qdrant.NewClient(
		qdrantServiceAddress,
		vectorCollectionName,
		uint64(*embeddingDimFlag),
	)
	if connectionError != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to Qdrant: %v\n", connectionError)
		fmt.Fprintln(os.Stderr, "Make sure Qdrant is running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
		os.Exit(1)
	}
	defer qdrantVectorClient.Close()

	// If preload flag is set, seed the database with demo words before starting the UI
	if *preloadDemoDataFlag {
		preloadError := runPreloadDemoWords(embedder, qdrantVectorClient)
		if preloadError != nil {
			fmt.Fprintf(os.Stderr, "Preload failed: %v\n", preloadError)
			os.Exit(1)
		}
	}

	// If a dataset path was provided, import it
	if datasetPath != "" {
		importError := runImportDataset(embedder, qdrantVectorClient, datasetPath)
		if importError != nil {
			fmt.Fprintf(os.Stderr, "Import failed: %v\n", importError)
			os.Exit(1)
		}
	}

	// If a Hugging Face dataset was specified, fetch and import it
	if *hfDatasetFlag != "" {
		importError := runImportHuggingFace(
			embedder,
			qdrantVectorClient,
			*hfDatasetFlag,
			*hfSplitFlag,
			*hfColumnFlag,
			*hfMaxRowsFlag,
		)
		if importError != nil {
			fmt.Fprintf(os.Stderr, "Hugging Face import failed: %v\n", importError)
			os.Exit(1)
		}
	}

	// Create and run the terminal user interface
	terminalUserInterfaceModel := tui.NewModel(embedder, qdrantVectorClient, version)
	bubbleTeaProgram := tea.NewProgram(terminalUserInterfaceModel, tea.WithAltScreen())

	_, programRunError := bubbleTeaProgram.Run()
	if programRunError != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", programRunError)
		os.Exit(1)
	}
}

// runPreloadDemoWords seeds the Qdrant database with a predefined list of demo words.
func runPreloadDemoWords(embedder embedding.Embedder, qdrantVectorClient *qdrant.Client) error {
	demoWordList := preload.Words()
	backgroundContext := context.Background()

	fmt.Printf("Preloading %d words...\n", len(demoWordList))

	for wordIndex, currentWord := range demoWordList {
		embeddingVector, embeddingError := embedder.Embed(currentWord)
		if embeddingError != nil {
			return fmt.Errorf("embed %q: %w", currentWord, embeddingError)
		}

		uniquePointIdentifier := uuid.New().String()
		upsertError := qdrantVectorClient.Upsert(backgroundContext, uniquePointIdentifier, currentWord, embeddingVector)
		if upsertError != nil {
			return fmt.Errorf("upsert %q: %w", currentWord, upsertError)
		}

		fmt.Printf("\r[%d/%d] %s", wordIndex+1, len(demoWordList), currentWord)
	}

	fmt.Println("\nDone.")
	return nil
}

// runImportDataset loads texts from a CSV or JSON file and embeds them into Qdrant.
func runImportDataset(embedder embedding.Embedder, qdrantVectorClient *qdrant.Client, datasetPath string) error {
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
		embeddingVector, embeddingError := embedder.Embed(currentText)
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

// runImportHuggingFace fetches texts from a Hugging Face dataset and embeds them into Qdrant.
func runImportHuggingFace(embedder embedding.Embedder, qdrantVectorClient *qdrant.Client, dataset, split, column string, maxRows int) error {
	hfClient := huggingface.NewClient()

	splits, splitsError := hfClient.GetSplits(dataset)
	if splitsError != nil {
		return fmt.Errorf("fetching splits: %w", splitsError)
	}

	if len(splits.Splits) == 0 {
		return fmt.Errorf("no splits found for dataset %s", dataset)
	}

	var config string
	for _, s := range splits.Splits {
		if s.Split == split {
			config = s.Config
			break
		}
	}
	if config == "" {
		config = splits.Splits[0].Config
		split = splits.Splits[0].Split
		fmt.Printf("Split not found, using %s/%s\n", config, split)
	}

	fmt.Printf("Fetching from Hugging Face: %s (config=%s, split=%s, column=%s)\n", dataset, config, split, column)

	texts, fetchError := hfClient.FetchTexts(dataset, config, split, column, maxRows)
	if fetchError != nil {
		return fmt.Errorf("fetching texts: %w", fetchError)
	}

	if len(texts) == 0 {
		return fmt.Errorf("no texts found in column %q", column)
	}

	backgroundContext := context.Background()
	fmt.Printf("Importing %d texts...\n", len(texts))

	for textIndex, currentText := range texts {
		embeddingVector, embeddingError := embedder.Embed(currentText)
		if embeddingError != nil {
			return fmt.Errorf("embed %q: %w", truncateForProgress(currentText, 20), embeddingError)
		}

		uniquePointIdentifier := uuid.New().String()
		upsertError := qdrantVectorClient.Upsert(backgroundContext, uniquePointIdentifier, currentText, embeddingVector)
		if upsertError != nil {
			return fmt.Errorf("upsert: %w", upsertError)
		}

		fmt.Printf("\r[%d/%d] %s", textIndex+1, len(texts), truncateForProgress(currentText, 40))
	}

	fmt.Println("\nDone.")
	return nil
}
