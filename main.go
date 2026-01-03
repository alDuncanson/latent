package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"latent/ollama"
	"latent/preload"
	"latent/qdrant"
	"latent/tui"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/uuid"
)

var version = "dev"

const (
	ollamaURL      = "http://localhost:11434"
	ollamaModel    = "nomic-embed-text"
	qdrantAddr     = "localhost:6334"
	collectionName = "embeddings"
	vectorSize     = 768
)

func main() {
	showVersion := flag.Bool("version", false, "print version and exit")
	doPreload := flag.Bool("preload", false, "seed with demo word list")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}

	ollamaClient := ollama.NewClient(ollamaURL, ollamaModel)

	qdrantClient, err := qdrant.NewClient(qdrantAddr, collectionName, vectorSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to Qdrant: %v\n", err)
		fmt.Fprintln(os.Stderr, "Make sure Qdrant is running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
		os.Exit(1)
	}
	defer qdrantClient.Close()

	if *doPreload {
		if err := runPreload(ollamaClient, qdrantClient); err != nil {
			fmt.Fprintf(os.Stderr, "Preload failed: %v\n", err)
			os.Exit(1)
		}
	}

	model := tui.NewModel(ollamaClient, qdrantClient)
	p := tea.NewProgram(model, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", err)
		os.Exit(1)
	}
}

func runPreload(ollamaClient *ollama.Client, qdrantClient *qdrant.Client) error {
	words := preload.Words()
	ctx := context.Background()

	fmt.Printf("Preloading %d words...\n", len(words))
	for i, word := range words {
		vec, err := ollamaClient.Embed(word)
		if err != nil {
			return fmt.Errorf("embed %q: %w", word, err)
		}
		if err := qdrantClient.Upsert(ctx, uuid.New().String(), word, vec); err != nil {
			return fmt.Errorf("upsert %q: %w", word, err)
		}
		fmt.Printf("\r[%d/%d] %s", i+1, len(words), word)
	}
	fmt.Println("\nDone.")
	return nil
}
