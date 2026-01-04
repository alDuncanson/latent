# latent

[![Build](https://github.com/alDuncanson/latent/actions/workflows/build.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/build.yml)
[![Release](https://github.com/alDuncanson/latent/actions/workflows/release.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/release.yml)

Peer into latent space.

![demo](assets/demo.gif)

Terminal UI for text embedding visualization. Embeds text into 768-dimensional
vectors via [Ollama](https://ollama.ai)'s `nomic-embed-text`, persists them to a
local [Qdrant](https://qdrant.tech) vector database, and projects the collection
into two-dimensional Euclidean space using principal component analysis.
Semantically similar texts cluster spatially; nearest neighbors surface in a
metadata panel.

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
