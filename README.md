# latent

[![Build](https://github.com/alDuncanson/latent/actions/workflows/build.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/build.yml)
[![Release](https://github.com/alDuncanson/latent/actions/workflows/release.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/release.yml)

A terminal UI for exploring text embeddings. Type text, watch it get embedded into 768-dimensional vectors via [Ollama](https://ollama.ai)'s `nomic-embed-text` model, and save points to a local [Qdrant](https://qdrant.tech) database. PCA projects stored embeddings down to 2D for real-time visualization—similar texts cluster together, and a metadata panel shows nearest neighbors.

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
latent
```
