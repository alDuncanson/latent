# latent

TUI for visualizing text embeddings using Ollama and Qdrant.

## Commands

```bash
go build ./...                    # Build
go run .                          # Run
go test ./...                     # Test all
go test ./projection -run TestPCA # Single test
```

## Dependencies

- Ollama at localhost:11434 with nomic-embed-text model
- Qdrant at localhost:6334 (gRPC)

## Architecture

```
main.go           Entry point, config, client init
ollama/client.go  HTTP client for Ollama /api/embed
qdrant/client.go  gRPC client (github.com/qdrant/go-client)
projection/pca.go SVD-based PCA for 768D -> 2D
tui/model.go      Bubble Tea model, lipgloss rendering
```

## Code Style

- Standard Go: gofmt, stdlib imports first, then external, then local (latent/...)
- Wrap errors: fmt.Errorf("context: %w", err)
- No emojis in UI; use lipgloss for styling
- Shift+key for destructive actions (D = delete), / for toggles
