# latent

A terminal UI for exploring text embeddings. Type text, save embeddings, and visualize semantic relationships in real-time.

## Requirements

- [Ollama](https://ollama.ai) with `nomic-embed-text`:
  ```bash
  ollama pull nomic-embed-text
  ollama serve
  ```

- [Qdrant](https://qdrant.tech) vector database:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

## Install

```bash
go install github.com/yourusername/latent@latest
```

Or build from source:

```bash
go build -o latent .
./latent
```

## Controls

| Key | Action |
|-----|--------|
| Type | Generate embeddings in real-time |
| Enter | Save current text and embedding |
| Tab / Shift+Tab | Cycle through saved points |
| Up / Down | Navigate saved points |
| / | Toggle metadata panel |
| D | Delete selected point |
| Esc | Quit |

## How it works

1. Text is embedded via Ollama as you type (debounced)
2. Saved embeddings are projected from 768D to 2D using PCA
3. Similar texts cluster together in the visualization
4. The metadata panel shows vector stats and nearest neighbors

## Symbols

- `●` Current input
- `○` Saved embedding
- `◉` Selected point
