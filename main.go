package main

import (
	"flag"
	"fmt"
	"os"

	"latent/ollama"
	"latent/qdrant"
	"latent/tui"

	tea "github.com/charmbracelet/bubbletea"
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

	model := tui.NewModel(ollamaClient, qdrantClient)
	p := tea.NewProgram(model, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", err)
		os.Exit(1)
	}
}
