# latent

[![Go Version](https://img.shields.io/github/go-mod/go-version/alDuncanson/latent)](https://go.dev/)
[![Build](https://github.com/alDuncanson/latent/actions/workflows/build.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/build.yml)
[![Release](https://github.com/alDuncanson/latent/actions/workflows/release.yml/badge.svg)](https://github.com/alDuncanson/latent/actions/workflows/release.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/alDuncanson/latent)](https://goreportcard.com/report/github.com/alDuncanson/latent)
[![License](https://img.shields.io/github/license/alDuncanson/latent)](https://github.com/alDuncanson/latent/blob/main/LICENSE)

Peer into latent space.

![demo](assets/demo.gif)

A terminal UI for visualizing high-dimensional embedding vectors through
dimensionality reduction. Latent explores the structure of vector embeddings by
projecting them from their native high-dimensional space onto a two-dimensional
manifold, revealing clusters and relationships that emerge from semantic
similarity.

Supports multiple projection methods including principal component analysis
(PCA) via singular value decomposition for fast linear projections, and UMAP
(Uniform Manifold Approximation and Projection) for nonlinear dimensionality
reduction that better preserves local and global topological structure.

Currently supports text embeddings via [Ollama](https://ollama.ai)'s
`nomic-embed-text` model, with vectors persisted to a local
[Qdrant](https://qdrant.tech) vector database. Nearest neighbors surface in a
metadata panel for interactive exploration.

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
latent                    # Start the TUI
latent dataset.csv        # Import texts from CSV then start TUI
latent dataset.json       # Import texts from JSON then start TUI
latent --preload          # Seed with demo word list
latent --version          # Print version
```

### Batch Import

Import many texts at once from CSV or JSON files:

**CSV format** - requires a `text` column header:

```csv
text
hello world
machine learning
neural networks
```

**JSON format** - array of strings or objects:

```json
["hello world", "machine learning", "neural networks"]
```

or

```json
[{ "text": "hello world" }, { "text": "machine learning" }]
```

### Keyboard Controls

| Key          | Action                                              |
| ------------ | --------------------------------------------------- |
| `Up/Down`    | Select previous/next point                          |
| `Left/Right` | Navigate among nearest neighbors                    |
| `Tab`        | Cycle through points                                |
| `/`          | Toggle metadata panel                               |
| `F`          | Toggle focus mode (show only selected + neighbors)  |
| `P`          | Toggle projection method (PCA / UMAP)               |
| `C`          | Toggle HDBSCAN clustering (color points by cluster) |
| `D`          | Delete selected point                               |
| `Enter`      | Save current input as embedding                     |
| `Esc`        | Quit                                                |

### Clustering

Press `C` to enable automatic cluster detection using HDBSCAN (Hierarchical
Density-Based Spatial Clustering of Applications with Noise). Points are colored
by their cluster assignment, with noise points shown in gray. The metadata panel
shows the cluster ID for the selected point.

HDBSCAN finds clusters of varying densities and automatically determines the
number of clusters—no need to specify k as with k-means.

Clustering works best with:

- **Diverse categories** - distinct semantic groups (colors vs animals vs
  emotions) create natural separation in embedding space
- **Sufficient density** - each category needs multiple terms to form dense
  regions that HDBSCAN can detect

Datasets with semantically similar terms (e.g., all physics terms) may produce
fewer clusters since the points are spread evenly rather than forming dense
pockets.

## Test Data

The `testdata/` directory contains sample files demonstrating supported import
formats:

| File                  | Format            | Description                               |
| --------------------- | ----------------- | ----------------------------------------- |
| `vehicles.json`       | JSON string array | Transportation terms (cars, planes, etc.) |
| `science.csv`         | CSV               | Physics and biology terms                 |
| `professions.json`    | JSON object array | Jobs and careers                          |
| `geography.csv`       | CSV               | Landforms and terrain                     |
| `mixed_categories.json` | JSON string array | 5 distinct groups for visible clustering  |

Example usage:

```bash
latent testdata/mixed_categories.json
```

The `mixed_categories.json` file works best for demonstrating clustering since
it contains 5 semantically distinct groups (tools, flowers, planets, clothing,
coffee drinks) with 8 terms each.
