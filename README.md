# latent

[![Go Version](https://img.shields.io/github/go-mod/go-version/alDuncanson/latent)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/alDuncanson/latent.svg)](https://pkg.go.dev/github.com/alDuncanson/latent)
[![Version](https://img.shields.io/github/v/release/alDuncanson/latent)](https://github.com/alDuncanson/latent/releases)
[![Build](https://github.com/alDuncanson/latent/actions/workflows/build.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/build.yml)
[![Release](https://github.com/alDuncanson/latent/actions/workflows/release.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/release.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/alDuncanson/latent)](https://goreportcard.com/report/github.com/alDuncanson/latent)
[![License](https://img.shields.io/github/license/alDuncanson/latent)](https://github.com/alDuncanson/latent/blob/main/LICENSE)

Peer into latent space.

![demo](assets/demo.gif)

Terminal UI for visualizing high-dimensional text embeddings via dimensionality reduction. Embeds text using [Ollama](https://ollama.ai)'s `nomic-embed-text` model (768D vectors), persists to [Qdrant](https://qdrant.tech) vector database over gRPC, and projects to 2D using PCA (SVD-based) or UMAP for nonlinear manifold approximation. Clustering via HDBSCAN reveals semantic structure without specifying k. Built with [Bubble Tea](https://github.com/charmbracelet/bubbletea) and [Lipgloss](https://github.com/charmbracelet/lipgloss).

## Prerequisites

- Ollama serving `nomic-embed-text` on `localhost:11434`
- Qdrant running on `localhost:6334` (gRPC)

## Install

```bash
curl -sSL https://raw.githubusercontent.com/alDuncanson/latent/main/install.sh | bash
```

or

```bash
go install github.com/alDuncanson/latent@latest
```

## Usage

```bash
latent                    # Start TUI
latent dataset.csv        # Import from CSV (requires `text` column)
latent dataset.json       # Import from JSON (array of strings or {text: ...} objects)
latent --preload          # Seed with demo word list
```

### Hugging Face Datasets

```bash
latent --hf-dataset stanfordnlp/imdb --hf-split test --hf-max-rows 50
latent --hf-dataset rajpurkar/squad --hf-column question --hf-max-rows 200
```

Flags: `--hf-dataset`, `--hf-split` (default: train), `--hf-column` (default: text), `--hf-max-rows` (default: 100)
