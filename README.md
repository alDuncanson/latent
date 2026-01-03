# latent

[![Build](https://github.com/alDuncanson/latent/actions/workflows/build.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/build.yml)
[![Release](https://github.com/alDuncanson/latent/actions/workflows/release.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/release.yml)

A terminal UI for exploring text embeddings. Type text, save embeddings, and visualize semantic relationships in real-time.

## Requirements

- [Ollama](https://ollama.ai) running locally with the `nomic-embed-text` model
- [Qdrant](https://qdrant.tech) vector database

```bash
ollama pull nomic-embed-text
ollama serve
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Install

**Quick install** (Linux/macOS):

```bash
curl -sSL https://raw.githubusercontent.com/alDuncanson/latent/main/install.sh | bash
```

**From release**:

Download the binary for your platform from [Releases](https://github.com/alDuncanson/latent/releases), then:

```bash
chmod +x latent-*
sudo mv latent-* /usr/local/bin/latent
```

**With Go**:

```bash
go install github.com/alDuncanson/latent@latest
```

**From source**:

```bash
git clone https://github.com/alDuncanson/latent.git
cd latent
go build -o latent .
sudo mv latent /usr/local/bin/
```

## Usage

```bash
latent           # start the TUI
latent -version  # print version
```

---

<details>
<summary>Technical Details</summary>

### How it works

1. Text is embedded via Ollama's `/api/embed` endpoint as you type (debounced)
2. 768-dimensional embeddings are stored in Qdrant via gRPC
3. PCA (via SVD) projects all saved embeddings to 2D for visualization
4. Similar texts cluster together; the metadata panel shows nearest neighbors

### Architecture

```
main.go           Entry point, config, client initialization
ollama/client.go  HTTP client for Ollama embedding API
qdrant/client.go  gRPC client using github.com/qdrant/go-client
projection/pca.go SVD-based dimensionality reduction (768D → 2D)
tui/model.go      Bubble Tea model with lipgloss rendering
```

</details>
