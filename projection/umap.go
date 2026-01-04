// Package projection provides dimensionality reduction for high-dimensional embedding vectors.
//
// # UMAP (Uniform Manifold Approximation and Projection) Overview
//
// UMAP is a nonlinear dimensionality reduction technique that preserves both local and global
// structure better than linear methods like PCA. It works by:
//
//  1. Constructing a k-nearest neighbor graph in high-dimensional space
//  2. Converting distances to fuzzy membership strengths (fuzzy simplicial set)
//  3. Initializing a low-dimensional embedding via spectral methods
//  4. Optimizing the embedding via stochastic gradient descent with negative sampling
//
// Reference: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
// Approximation and Projection for Dimension Reduction. https://arxiv.org/abs/1802.03426
//
// This is a Go port of the Python umap-learn library: https://github.com/lmcinnes/umap
package projection

import (
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// UMAPConfig holds hyperparameters for UMAP dimensionality reduction.
type UMAPConfig struct {
	NNeighbors         int     // Number of nearest neighbors (default: 15)
	MinDist            float64 // Minimum distance in low-dim space (default: 0.1)
	Spread             float64 // Effective scale of embedded points (default: 1.0)
	NEpochs            int     // Number of optimization epochs (default: 200)
	LearningRate       float64 // Initial learning rate (default: 1.0)
	NegativeSampleRate float64 // Negative samples per positive (default: 5.0)
	RandomSeed         int64   // Random seed for reproducibility
}

// DefaultUMAPConfig returns sensible default hyperparameters.
func DefaultUMAPConfig() UMAPConfig {
	return UMAPConfig{
		NNeighbors:         15,
		MinDist:            0.1,
		Spread:             1.0,
		NEpochs:            200,
		LearningRate:       1.0,
		NegativeSampleRate: 5.0,
		RandomSeed:         42,
	}
}

// COOMatrix represents a sparse matrix in coordinate (COO) format.
type COOMatrix struct {
	Rows []int
	Cols []int
	Data []float64
	NRow int
	NCol int
}

// knnResult holds k-nearest neighbor indices and distances for all points.
type knnResult struct {
	Indices [][]int     // [nSamples][k] neighbor indices
	Dists   [][]float64 // [nSamples][k] distances to neighbors
}

// ProjectTo2DUMAP reduces high-dimensional embedding vectors to 2D points using UMAP.
// This provides an alternative to PCA that better preserves nonlinear manifold structure.
func ProjectTo2DUMAP(embeddingVectors [][]float32, textLabels []string) []Point2D {
	return ProjectTo2DUMAPWithConfig(embeddingVectors, textLabels, DefaultUMAPConfig())
}

// ProjectTo2DUMAPWithConfig allows customizing UMAP hyperparameters.
func ProjectTo2DUMAPWithConfig(embeddingVectors [][]float32, textLabels []string, config UMAPConfig) []Point2D {
	if len(embeddingVectors) == 0 {
		return nil
	}

	nSamples := len(embeddingVectors)
	nDims := len(embeddingVectors[0])

	if nDims < 2 || nSamples < 2 {
		return createFallbackProjection(embeddingVectors, textLabels)
	}

	// Adjust k for small datasets
	k := config.NNeighbors
	if k >= nSamples {
		k = nSamples - 1
	}
	if k < 2 {
		return createFallbackProjection(embeddingVectors, textLabels)
	}

	// Convert to float64 for numerical precision
	data := convertToFloat64(embeddingVectors)

	// Step 1: Build k-NN graph
	knn := computeKNN(data, k)

	// Step 2: Compute fuzzy simplicial set
	sigmas, rhos := smoothKNNDist(knn.Dists, float64(k))
	graph := computeFuzzySimplicialSet(knn, sigmas, rhos, nSamples)

	// Step 3: Find output manifold parameters
	a, b := findABParams(config.Spread, config.MinDist)

	// Step 4: Initialize embedding (spectral or random)
	embedding := initializeEmbedding(graph, nSamples, 2, config.RandomSeed)

	// Step 5: Optimize via SGD - create fresh RNG from seed for reproducibility
	rng := rand.New(rand.NewSource(config.RandomSeed + 1))
	embedding = optimizeLayout(
		embedding, graph, a, b,
		config.NEpochs, config.LearningRate,
		config.NegativeSampleRate, rng,
	)

	// Convert to Point2D
	return embeddingToPoints(embedding, textLabels)
}

// convertToFloat64 converts float32 vectors to float64 for numerical precision.
func convertToFloat64(vectors [][]float32) [][]float64 {
	result := make([][]float64, len(vectors))
	for i, v := range vectors {
		result[i] = make([]float64, len(v))
		for j, val := range v {
			result[i][j] = float64(val)
		}
	}
	return result
}

// computeKNN computes k-nearest neighbors using brute force (O(n²)).
// For production use with large datasets, replace with approximate NN (e.g., NN-Descent).
func computeKNN(data [][]float64, k int) knnResult {
	n := len(data)
	indices := make([][]int, n)
	dists := make([][]float64, n)

	type distIdx struct {
		dist float64
		idx  int
	}

	for i := 0; i < n; i++ {
		neighbors := make([]distIdx, n)
		for j := 0; j < n; j++ {
			neighbors[j] = distIdx{
				dist: euclideanDistance(data[i], data[j]),
				idx:  j,
			}
		}
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].dist < neighbors[b].dist
		})

		// Take k+1 to include self, then skip self
		indices[i] = make([]int, k)
		dists[i] = make([]float64, k)
		idx := 0
		for j := 0; j < len(neighbors) && idx < k; j++ {
			if neighbors[j].idx == i {
				continue // Skip self
			}
			indices[i][idx] = neighbors[j].idx
			dists[i][idx] = neighbors[j].dist
			idx++
		}
	}

	return knnResult{Indices: indices, Dists: dists}
}

// euclideanDistance computes the Euclidean distance between two vectors.
func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// smoothKNNDist computes sigma (bandwidth) and rho (local connectivity distance) for each point.
// Uses binary search to find sigma such that the sum of fuzzy memberships equals log2(k).
func smoothKNNDist(distances [][]float64, k float64) (sigmas, rhos []float64) {
	const (
		nIter             = 64
		localConnectivity = 1.0
		smoothKTolerance  = 1e-5
		minKDistScale     = 1e-3
	)

	n := len(distances)
	sigmas = make([]float64, n)
	rhos = make([]float64, n)
	target := math.Log2(k)

	for i := 0; i < n; i++ {
		dists := distances[i]

		// Compute rho: distance to the local_connectivity-th neighbor
		nonZeroDists := make([]float64, 0, len(dists))
		for _, d := range dists {
			if d > 0 {
				nonZeroDists = append(nonZeroDists, d)
			}
		}

		if len(nonZeroDists) >= int(localConnectivity) {
			idx := int(math.Floor(localConnectivity))
			interp := localConnectivity - float64(idx)
			if idx > 0 {
				rhos[i] = nonZeroDists[idx-1]
				if interp > smoothKTolerance {
					rhos[i] += interp * (nonZeroDists[idx] - nonZeroDists[idx-1])
				}
			} else {
				rhos[i] = interp * nonZeroDists[0]
			}
		} else if len(nonZeroDists) > 0 {
			rhos[i] = nonZeroDists[len(nonZeroDists)-1]
		}

		// Binary search for sigma
		lo, hi, mid := 0.0, math.Inf(1), 1.0

		for iter := 0; iter < nIter; iter++ {
			psum := 0.0
			for j := 0; j < len(dists); j++ {
				d := dists[j] - rhos[i]
				if d > 0 {
					psum += math.Exp(-d / mid)
				} else {
					psum += 1.0
				}
			}

			if math.Abs(psum-target) < smoothKTolerance {
				break
			}

			if psum > target {
				hi = mid
			} else {
				lo = mid
			}

			if math.IsInf(hi, 1) {
				mid *= 2
			} else {
				mid = (lo + hi) / 2
			}
		}

		sigmas[i] = mid

		// Enforce minimum sigma
		meanDist := mean(dists)
		minSigma := minKDistScale * meanDist
		if sigmas[i] < minSigma {
			sigmas[i] = minSigma
		}
	}

	return sigmas, rhos
}

// computeFuzzySimplicialSet constructs the fuzzy graph from k-NN data.
func computeFuzzySimplicialSet(knn knnResult, sigmas, rhos []float64, nSamples int) COOMatrix {
	// Compute membership strengths
	rows, cols, vals := computeMembershipStrengths(knn, sigmas, rhos)

	// Build sparse matrix
	graph := COOMatrix{
		Rows: rows,
		Cols: cols,
		Data: vals,
		NRow: nSamples,
		NCol: nSamples,
	}

	// Apply fuzzy set union: result + transpose - product
	graph = fuzzySetUnion(graph)

	return graph
}

// computeMembershipStrengths computes fuzzy membership values for each edge.
func computeMembershipStrengths(knn knnResult, sigmas, rhos []float64) (rows, cols []int, vals []float64) {
	n := len(knn.Indices)
	k := len(knn.Indices[0])

	rows = make([]int, 0, n*k)
	cols = make([]int, 0, n*k)
	vals = make([]float64, 0, n*k)

	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			neighbor := knn.Indices[i][j]
			dist := knn.Dists[i][j]

			var val float64
			if dist-rhos[i] <= 0 || sigmas[i] == 0 {
				val = 1.0
			} else {
				val = math.Exp(-(dist - rhos[i]) / sigmas[i])
			}

			rows = append(rows, i)
			cols = append(cols, neighbor)
			vals = append(vals, val)
		}
	}

	return rows, cols, vals
}

// fuzzySetUnion symmetrizes the graph using fuzzy set union.
// Union formula: P(A ∪ B) = P(A) + P(B) - P(A)P(B)
func fuzzySetUnion(graph COOMatrix) COOMatrix {
	// Build a map for fast lookup
	type edge struct{ r, c int }
	edgeMap := make(map[edge]float64)

	for i := range graph.Rows {
		e := edge{graph.Rows[i], graph.Cols[i]}
		edgeMap[e] = graph.Data[i]
	}

	// Compute union with transpose - iterate in deterministic order
	resultMap := make(map[edge]float64)
	for i := range graph.Rows {
		e := edge{graph.Rows[i], graph.Cols[i]}
		v := graph.Data[i]
		transpose := edge{e.c, e.r}
		vt := edgeMap[transpose]

		// Fuzzy union: v + vt - v*vt
		union := v + vt - v*vt
		if union > 0 {
			resultMap[e] = union
		}
	}

	// Convert back to COO in deterministic order
	// Sort edges for reproducibility
	edges := make([]edge, 0, len(resultMap))
	for e := range resultMap {
		edges = append(edges, e)
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].r != edges[j].r {
			return edges[i].r < edges[j].r
		}
		return edges[i].c < edges[j].c
	})

	rows := make([]int, len(edges))
	cols := make([]int, len(edges))
	vals := make([]float64, len(edges))

	for i, e := range edges {
		rows[i] = e.r
		cols[i] = e.c
		vals[i] = resultMap[e]
	}

	return COOMatrix{
		Rows: rows,
		Cols: cols,
		Data: vals,
		NRow: graph.NRow,
		NCol: graph.NCol,
	}
}

// findABParams fits curve parameters for the low-dimensional membership function.
// Fits: f(x) = 1 / (1 + a * x^(2b)) to approximate the target distribution.
func findABParams(spread, minDist float64) (a, b float64) {
	// Generate target curve
	const nPoints = 300
	xv := make([]float64, nPoints)
	yv := make([]float64, nPoints)

	for i := 0; i < nPoints; i++ {
		xv[i] = float64(i) / float64(nPoints-1) * spread * 3
		if xv[i] < minDist {
			yv[i] = 1.0
		} else {
			yv[i] = math.Exp(-(xv[i] - minDist) / spread)
		}
	}

	// Simple curve fitting via grid search
	// For production, use proper nonlinear least squares
	bestA, bestB := 1.0, 1.0
	bestError := math.Inf(1)

	for aTest := 0.1; aTest <= 10.0; aTest += 0.1 {
		for bTest := 0.1; bTest <= 2.0; bTest += 0.05 {
			err := 0.0
			for i := 0; i < nPoints; i++ {
				pred := 1.0 / (1.0 + aTest*math.Pow(xv[i], 2*bTest))
				diff := pred - yv[i]
				err += diff * diff
			}
			if err < bestError {
				bestError = err
				bestA, bestB = aTest, bTest
			}
		}
	}

	return bestA, bestB
}

// initializeEmbedding creates the initial low-dimensional embedding.
// Uses spectral initialization when possible, falls back to random.
func initializeEmbedding(graph COOMatrix, nSamples, nDims int, seed int64) [][]float64 {
	// Create a fresh RNG for initialization to ensure reproducibility
	rng := rand.New(rand.NewSource(seed))

	// Try spectral initialization
	embedding := spectralLayout(graph, nSamples, nDims)
	if embedding != nil {
		// Add small noise
		for i := range embedding {
			for j := range embedding[i] {
				embedding[i][j] += (rng.Float64() - 0.5) * 0.0001
			}
		}
		return embedding
	}

	// Fallback to random initialization
	embedding = make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		embedding[i] = make([]float64, nDims)
		for j := 0; j < nDims; j++ {
			embedding[i][j] = (rng.Float64() - 0.5) * 10
		}
	}
	return embedding
}

// spectralLayout computes initial embedding using graph Laplacian eigenvectors.
// For small datasets (< 50 points), returns nil to use random initialization instead,
// as spectral layout is more valuable for preserving global structure in larger datasets.
func spectralLayout(graph COOMatrix, nSamples, nDims int) [][]float64 {
	// Skip spectral for small datasets - random init works fine
	if nSamples < 50 {
		return nil
	}

	// Build adjacency matrix
	adj := mat.NewDense(nSamples, nSamples, nil)
	for i := range graph.Rows {
		adj.Set(graph.Rows[i], graph.Cols[i], graph.Data[i])
	}

	// Compute degree matrix
	degrees := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nSamples; j++ {
			degrees[i] += adj.At(i, j)
		}
	}

	// Build normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
	laplacian := mat.NewDense(nSamples, nSamples, nil)
	for i := 0; i < nSamples; i++ {
		laplacian.Set(i, i, 1.0) // Identity diagonal
		for j := 0; j < nSamples; j++ {
			if degrees[i] > 0 && degrees[j] > 0 {
				normalized := adj.At(i, j) / math.Sqrt(degrees[i]*degrees[j])
				laplacian.Set(i, j, laplacian.At(i, j)-normalized)
			}
		}
	}

	// Compute eigendecomposition
	var eigen mat.Eigen
	ok := eigen.Factorize(laplacian, mat.EigenRight)
	if !ok {
		return nil
	}

	values := eigen.Values(nil)
	vectors := mat.CDense{}
	eigen.VectorsTo(&vectors)

	// Sort eigenvalues and get indices of smallest (skip first trivial one)
	type eigenPair struct {
		val float64
		idx int
	}
	pairs := make([]eigenPair, len(values))
	for i, v := range values {
		pairs[i] = eigenPair{real(v), i}
	}
	sort.Slice(pairs, func(a, b int) bool {
		return pairs[a].val < pairs[b].val
	})

	// Extract eigenvectors for smallest non-trivial eigenvalues
	embedding := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		embedding[i] = make([]float64, nDims)
		for j := 0; j < nDims; j++ {
			if j+1 < len(pairs) {
				embedding[i][j] = real(vectors.At(i, pairs[j+1].idx))
			}
		}
	}

	// Scale to reasonable range
	for d := 0; d < nDims; d++ {
		minVal, maxVal := math.Inf(1), math.Inf(-1)
		for i := 0; i < nSamples; i++ {
			if embedding[i][d] < minVal {
				minVal = embedding[i][d]
			}
			if embedding[i][d] > maxVal {
				maxVal = embedding[i][d]
			}
		}
		scale := maxVal - minVal
		if scale > 0 {
			for i := 0; i < nSamples; i++ {
				embedding[i][d] = (embedding[i][d] - minVal) / scale * 10
			}
		}
	}

	return embedding
}

// optimizeLayout performs SGD optimization to refine the embedding.
func optimizeLayout(
	embedding [][]float64,
	graph COOMatrix,
	a, b float64,
	nEpochs int,
	initialAlpha float64,
	negativeSampleRate float64,
	rng *rand.Rand,
) [][]float64 {
	nSamples := len(embedding)
	nEdges := len(graph.Rows)

	if nEdges == 0 || nSamples < 2 {
		return embedding
	}

	// Compute epochs per sample based on edge weight
	maxWeight := 0.0
	for _, w := range graph.Data {
		if w > maxWeight {
			maxWeight = w
		}
	}
	if maxWeight == 0 {
		maxWeight = 1.0
	}

	// Each edge is sampled proportionally to its weight
	// Higher weight edges are sampled more frequently
	epochsPerSample := make([]float64, nEdges)
	for i, w := range graph.Data {
		if w > 0 {
			// Inverse: stronger edges sampled more often (smaller epoch interval)
			epochsPerSample[i] = float64(nEpochs) / (float64(nEpochs) * (w / maxWeight))
			if epochsPerSample[i] < 1 {
				epochsPerSample[i] = 1
			}
		} else {
			epochsPerSample[i] = float64(nEpochs) + 1 // Never sample
		}
	}

	epochOfNextSample := make([]float64, nEdges)
	for i := range epochOfNextSample {
		epochOfNextSample[i] = epochsPerSample[i]
	}

	// Limit negative samples per positive sample
	nNegPerPos := int(negativeSampleRate)
	if nNegPerPos < 1 {
		nNegPerPos = 1
	}

	// SGD optimization loop
	for epoch := 0; epoch < nEpochs; epoch++ {
		alpha := initialAlpha * (1.0 - float64(epoch)/float64(nEpochs))
		if alpha < 0.0001 {
			alpha = 0.0001
		}

		for i := 0; i < nEdges; i++ {
			if epochOfNextSample[i] > float64(epoch) {
				continue
			}

			j := graph.Rows[i]
			k := graph.Cols[i]
			if j >= nSamples || k >= nSamples {
				continue
			}

			// Positive sample (attraction)
			current := embedding[j]
			other := embedding[k]

			distSq := squaredEuclidean(current, other)
			if distSq > 0 {
				gradCoeff := -2.0 * a * b * math.Pow(distSq, b-1.0)
				gradCoeff /= a*math.Pow(distSq, b) + 1.0

				for d := range current {
					grad := clip(gradCoeff * (current[d] - other[d]))
					embedding[j][d] += grad * alpha
				}
			}

			// Negative samples (repulsion)
			for p := 0; p < nNegPerPos; p++ {
				negIdx := rng.Intn(nSamples)
				if negIdx == j {
					continue
				}

				negPoint := embedding[negIdx]
				distSq := squaredEuclidean(current, negPoint)

				var gradCoeff float64
				if distSq > 0.001 {
					gradCoeff = 2.0 * b
					gradCoeff /= (0.001 + distSq) * (a*math.Pow(distSq, b) + 1)
				}

				if gradCoeff > 0 {
					for d := range current {
						grad := clip(gradCoeff * (current[d] - negPoint[d]))
						embedding[j][d] += grad * alpha
					}
				}
			}

			epochOfNextSample[i] += epochsPerSample[i]
		}
	}

	return embedding
}

// squaredEuclidean computes the squared Euclidean distance.
func squaredEuclidean(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// clip constrains gradient values to prevent explosive updates.
func clip(val float64) float64 {
	if val > 4.0 {
		return 4.0
	}
	if val < -4.0 {
		return -4.0
	}
	return val
}

// mean computes the arithmetic mean of a slice.
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// embeddingToPoints converts the final embedding to Point2D structs.
func embeddingToPoints(embedding [][]float64, textLabels []string) []Point2D {
	points := make([]Point2D, len(embedding))
	for i, coords := range embedding {
		label := ""
		if i < len(textLabels) {
			label = textLabels[i]
		}
		points[i] = Point2D{
			X:    coords[0],
			Y:    coords[1],
			Text: label,
		}
	}
	return points
}
