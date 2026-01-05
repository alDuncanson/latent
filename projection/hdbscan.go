// Package projection provides dimensionality reduction and clustering for high-dimensional embedding vectors.
//
// # HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) Overview
//
// HDBSCAN is a density-based clustering algorithm that finds clusters of varying densities without
// requiring the number of clusters to be specified in advance. Unlike K-means, it can discover
// arbitrarily shaped clusters and automatically identifies noise points (outliers).
//
// The algorithm works in five main steps:
//
//  1. Compute Core Distances: For each point, find the distance to its k-th nearest neighbor.
//     This measures local density—points in dense regions have small core distances.
//
//  2. Build Mutual Reachability Graph: Transform the distance metric to account for density.
//     The mutual reachability distance between points a and b is:
//     max(core_dist(a), core_dist(b), dist(a, b))
//     This makes sparse regions "farther apart" even if Euclidean distance is small.
//
//  3. Construct Minimum Spanning Tree: Build an MST using mutual reachability distances.
//     This captures the hierarchical cluster structure—edges with small weights connect
//     dense regions, while edges with large weights span sparse regions.
//
//  4. Build Condensed Tree: Walk the MST from longest to shortest edges, tracking when
//     clusters split. Small splits (fewer than MinClusterSize points) are treated as
//     points "falling out" of a cluster rather than true splits.
//
//  5. Extract Clusters: Use cluster stability (how long points persist in a cluster)
//     to select the most prominent clusters. Points not in any stable cluster are noise.
//
// Key advantages over other clustering methods:
//   - No need to specify number of clusters
//   - Robust to noise and outliers (labels them as -1)
//   - Finds clusters of varying densities
//   - Produces a hierarchy that can be cut at different levels
//
// Reference: Campello, R.J.G.B., Moulavi, D., & Sander, J. (2013). Density-Based Clustering
// Based on Hierarchical Density Estimates. https://doi.org/10.1007/978-3-642-37456-2_14
package projection

import (
	"math"
	"sort"
)

// HDBSCANConfig holds hyperparameters for HDBSCAN clustering.
type HDBSCANConfig struct {
	MinClusterSize int // Minimum points required to form a cluster (default: 5)
	MinSamples     int // Points used to estimate density; affects core distance (default: MinClusterSize)
}

// DefaultHDBSCANConfig returns sensible default hyperparameters.
// MinClusterSize=5 works well for most datasets; increase for larger datasets
// or when you want to ignore small clusters.
func DefaultHDBSCANConfig() HDBSCANConfig {
	return HDBSCANConfig{
		MinClusterSize: 5,
		MinSamples:     0, // 0 means use MinClusterSize
	}
}

// ClusterResult contains the output of HDBSCAN clustering.
type ClusterResult struct {
	Labels        []int     // Cluster assignment for each point (-1 = noise)
	Probabilities []float64 // Confidence score (0-1) for each point's cluster membership
}

// ClusterLabels is a convenience function that returns only cluster labels.
// Points labeled -1 are noise (not part of any cluster).
func ClusterLabels(vectors [][]float32, config HDBSCANConfig) []int {
	return Cluster(vectors, config).Labels
}

// Cluster performs HDBSCAN clustering on high-dimensional vectors.
// Returns cluster labels and membership probabilities for each point.
//
// The algorithm pipeline:
//  1. Compute core distances (local density estimation)
//  2. Build minimum spanning tree using mutual reachability distance
//  3. Convert MST to single-linkage dendrogram
//  4. Condense the tree by removing spurious splits
//  5. Extract stable clusters using persistence-based selection
//  6. Compute membership probabilities based on cluster lifetime
func Cluster(vectors [][]float32, config HDBSCANConfig) ClusterResult {
	n := len(vectors)
	if n == 0 {
		return ClusterResult{}
	}

	// Ensure valid configuration
	if config.MinClusterSize < 2 {
		config.MinClusterSize = 5
	}
	if config.MinSamples <= 0 {
		config.MinSamples = config.MinClusterSize
	}

	// Not enough points to form even one cluster
	if n < config.MinClusterSize {
		labels := make([]int, n)
		probs := make([]float64, n)
		for i := range labels {
			labels[i] = -1 // All points are noise
		}
		return ClusterResult{Labels: labels, Probabilities: probs}
	}

	// Convert to float64 for numerical precision
	data := convertToFloat64(vectors)

	// Step 1: Compute core distances (k-th nearest neighbor distance for each point)
	coreDistances := computeCoreDistances(data, config.MinSamples)

	// Step 2: Build MST using mutual reachability distance
	mstEdges := computeMutualReachabilityMST(data, coreDistances)

	// Step 3: Convert MST edges to single-linkage dendrogram
	linkage := singleLinkageTree(mstEdges, n)

	// Step 4: Condense tree by merging small clusters into parents
	condensed := condenseTree(linkage, config.MinClusterSize, n)

	// Step 5: Compute stability scores and extract clusters
	stability := computeStability(condensed)
	labels := extractClusters(condensed, stability, n)

	// Step 6: Compute membership probabilities
	probs := computeProbabilities(condensed, labels)

	return ClusterResult{Labels: labels, Probabilities: probs}
}

// mstEdge represents an edge in the minimum spanning tree.
// From and To are point indices, Weight is the mutual reachability distance.
type mstEdge struct {
	From, To int
	Weight   float64
}

// linkageNode represents a merge in the single-linkage dendrogram.
// Left and Right are the merged cluster indices, Distance is the merge distance,
// Size is the total number of points in the resulting cluster.
type linkageNode struct {
	Left, Right int
	Distance    float64
	Size        int
}

// condensedEdge represents a parent-child relationship in the condensed tree.
// Lambda (1/distance) represents density—higher lambda means denser regions.
// ChildSize=1 indicates a single point falling out of a cluster.
type condensedEdge struct {
	Parent    int
	Child     int
	Lambda    float64 // 1/distance at which this edge was created
	ChildSize int     // 1 for single points, >1 for sub-clusters
}

// condensedTree is the simplified cluster hierarchy after removing spurious splits.
// Only splits where both children have at least MinClusterSize points are kept as true splits.
type condensedTree struct {
	Edges       []condensedEdge
	RootCluster int
}

// computeCoreDistances calculates the core distance for each point.
// The core distance is the distance to the k-th nearest neighbor (where k = minSamples).
// Points in dense regions have small core distances; isolated points have large ones.
//
// This is a key concept in density-based clustering: it measures how "crowded"
// the neighborhood around each point is.
func computeCoreDistances(data [][]float64, minSamples int) []float64 {
	n := len(data)
	coreDistances := make([]float64, n)

	if minSamples > n {
		minSamples = n
	}

	for i := 0; i < n; i++ {
		// Compute distances from point i to all other points
		dists := make([]float64, n)
		for j := 0; j < n; j++ {
			dists[j] = euclideanDistance(data[i], data[j])
		}
		sort.Float64s(dists)

		// Core distance is distance to k-th neighbor (0th is self with distance 0)
		k := minSamples
		if k >= n {
			k = n - 1
		}
		coreDistances[i] = dists[k]
	}

	return coreDistances
}

// computeMutualReachabilityMST builds a minimum spanning tree using mutual reachability distance.
//
// Mutual reachability distance between points a and b is:
//
//	MRD(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
//
// This transformation makes points in sparse regions effectively farther apart,
// ensuring that clusters in dense regions are connected before sparse regions.
//
// Uses Prim's algorithm for MST construction (O(n²) complexity).
func computeMutualReachabilityMST(data [][]float64, coreDistances []float64) []mstEdge {
	n := len(data)
	if n < 2 {
		return nil
	}

	// Prim's algorithm state
	inTree := make([]bool, n)       // Points already in the MST
	minDist := make([]float64, n)   // Minimum MRD to reach each point
	minEdge := make([]int, n)       // Source point for minimum MRD edge

	for i := range minDist {
		minDist[i] = math.Inf(1)
		minEdge[i] = -1
	}

	edges := make([]mstEdge, 0, n-1)
	current := 0
	inTree[current] = true

	// Add n-1 edges to complete the tree
	for added := 1; added < n; added++ {
		// Update minimum distances from current point to all non-tree points
		for j := 0; j < n; j++ {
			if inTree[j] {
				continue
			}

			dist := euclideanDistance(data[current], data[j])
			// Mutual reachability distance considers both core distances
			mrd := math.Max(coreDistances[current], math.Max(coreDistances[j], dist))

			if mrd < minDist[j] {
				minDist[j] = mrd
				minEdge[j] = current
			}
		}

		// Find the non-tree point with minimum distance
		minIdx := -1
		minVal := math.Inf(1)
		for j := 0; j < n; j++ {
			if !inTree[j] && minDist[j] < minVal {
				minVal = minDist[j]
				minIdx = j
			}
		}

		if minIdx < 0 {
			break // Disconnected graph (shouldn't happen with valid data)
		}

		edges = append(edges, mstEdge{
			From:   minEdge[minIdx],
			To:     minIdx,
			Weight: minVal,
		})
		inTree[minIdx] = true
		current = minIdx
	}

	// Sort edges by weight for hierarchical processing
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].Weight < edges[j].Weight
	})

	return edges
}

// unionFind implements a disjoint-set data structure with path compression.
// Used to efficiently track cluster merges when building the dendrogram.
// Nodes 0 to n-1 are initial singleton clusters; nodes n+ are merged clusters.
type unionFind struct {
	parent []int // Parent pointer (self-loop for roots)
	size   []int // Size of each cluster
	next   int   // Next available node ID for merged clusters
}

// newUnionFind creates a union-find structure for n initial singletons.
// Allocates 2n-1 slots to accommodate n-1 merge operations.
func newUnionFind(n int) *unionFind {
	parent := make([]int, 2*n-1)
	size := make([]int, 2*n-1)
	for i := 0; i < n; i++ {
		parent[i] = i
		size[i] = 1
	}
	return &unionFind{parent: parent, size: size, next: n}
}

// find returns the root of the cluster containing x, with path compression.
func (uf *unionFind) find(x int) int {
	root := x
	for uf.parent[root] != root {
		root = uf.parent[root]
	}
	// Path compression: point all nodes directly to root
	for uf.parent[x] != root {
		next := uf.parent[x]
		uf.parent[x] = root
		x = next
	}
	return root
}

// union merges clusters x and y into a new cluster node.
// Returns the ID of the newly created merged cluster.
func (uf *unionFind) union(x, y int) int {
	newNode := uf.next
	uf.parent[x] = newNode
	uf.parent[y] = newNode
	uf.parent[newNode] = newNode
	uf.size[newNode] = uf.size[x] + uf.size[y]
	uf.next++
	return newNode
}

// singleLinkageTree converts MST edges into a single-linkage dendrogram.
// Processes edges in order of increasing weight, merging clusters as edges are added.
// The result is a sequence of merge operations forming a hierarchical tree.
func singleLinkageTree(edges []mstEdge, nSamples int) []linkageNode {
	if len(edges) == 0 {
		return nil
	}

	uf := newUnionFind(nSamples)
	tree := make([]linkageNode, len(edges))

	for i, edge := range edges {
		left := uf.find(edge.From)
		right := uf.find(edge.To)

		leftSize := uf.size[left]
		rightSize := uf.size[right]
		newSize := leftSize + rightSize

		uf.union(left, right)

		tree[i] = linkageNode{
			Left:     left,
			Right:    right,
			Distance: edge.Weight,
			Size:     newSize,
		}
	}

	return tree
}

// condenseTree simplifies the dendrogram by removing spurious splits.
// A split is considered "real" only if both children have at least MinClusterSize points.
// Smaller splits are treated as individual points falling out of the parent cluster.
//
// This produces a tree where each node is either:
//   - A cluster (size > 1) that persists across a range of density thresholds
//   - A single point that fell out of its parent cluster at some density
//
// Lambda (1/distance) is used instead of distance because it represents density:
// higher lambda = denser region = points stayed together longer.
func condenseTree(tree []linkageNode, minClusterSize int, nSamples int) condensedTree {
	if len(tree) == 0 {
		return condensedTree{RootCluster: nSamples}
	}

	rootCluster := nSamples + len(tree) - 1
	relabel := make(map[int]int)
	nextLabel := nSamples

	var edges []condensedEdge

	// Recursive function to process the dendrogram
	var condense func(node int, parent int, lambdaVal float64)
	condense = func(node int, parent int, lambdaVal float64) {
		// Base case: leaf node (original data point)
		if node < nSamples {
			edges = append(edges, condensedEdge{
				Parent:    parent,
				Child:     node,
				Lambda:    lambdaVal,
				ChildSize: 1,
			})
			return
		}

		// Get the linkage node for this merge
		linkIdx := node - nSamples
		if linkIdx < 0 || linkIdx >= len(tree) {
			return
		}
		linkNode := tree[linkIdx]

		left := linkNode.Left
		right := linkNode.Right

		// Compute sizes of left and right subtrees
		leftSize := 1
		rightSize := 1
		if left >= nSamples {
			leftIdx := left - nSamples
			if leftIdx >= 0 && leftIdx < len(tree) {
				leftSize = tree[leftIdx].Size
			}
		}
		if right >= nSamples {
			rightIdx := right - nSamples
			if rightIdx >= 0 && rightIdx < len(tree) {
				rightSize = tree[rightIdx].Size
			}
		}

		// Lambda = 1/distance (density measure)
		newLambda := 0.0
		if linkNode.Distance > 0 {
			newLambda = 1.0 / linkNode.Distance
		}

		leftValid := leftSize >= minClusterSize
		rightValid := rightSize >= minClusterSize

		if leftValid && rightValid {
			// True split: both children are valid clusters
			leftLabel := nextLabel
			nextLabel++
			rightLabel := nextLabel
			nextLabel++

			relabel[left] = leftLabel
			relabel[right] = rightLabel

			edges = append(edges, condensedEdge{
				Parent:    parent,
				Child:     leftLabel,
				Lambda:    newLambda,
				ChildSize: leftSize,
			})
			edges = append(edges, condensedEdge{
				Parent:    parent,
				Child:     rightLabel,
				Lambda:    newLambda,
				ChildSize: rightSize,
			})

			condense(left, leftLabel, newLambda)
			condense(right, rightLabel, newLambda)
		} else if leftValid {
			// Right child too small: points fall out, left continues as parent
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		} else if rightValid {
			// Left child too small: points fall out, right continues as parent
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		} else {
			// Both too small: all points fall out into parent
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		}
	}

	// Start from root
	rootLabel := nextLabel
	nextLabel++
	relabel[rootCluster] = rootLabel

	var rootLambda float64
	if len(tree) > 0 && tree[len(tree)-1].Distance > 0 {
		rootLambda = 1.0 / tree[len(tree)-1].Distance
	}
	condense(rootCluster, rootLabel, rootLambda)

	return condensedTree{Edges: edges, RootCluster: rootLabel}
}

// computeStability calculates the stability score for each cluster.
// Stability measures how long points persist in a cluster as density threshold increases.
//
// For each cluster, stability = sum over all points of (lambda_death - lambda_birth)
// where lambda_death is when the point falls out and lambda_birth is when the cluster formed.
//
// Higher stability indicates a more "real" cluster that persists across density levels.
func computeStability(tree condensedTree) map[int]float64 {
	stability := make(map[int]float64)

	// Track when each cluster was born (first appeared in the tree)
	birthLambda := make(map[int]float64)
	for _, e := range tree.Edges {
		if e.Child >= len(tree.Edges)+tree.RootCluster-len(tree.Edges) {
			continue
		}
		if e.ChildSize > 1 {
			if _, ok := birthLambda[e.Child]; !ok {
				birthLambda[e.Child] = e.Lambda
			}
		}
	}

	// Identify all clusters in the tree
	clusters := make(map[int]bool)
	for _, e := range tree.Edges {
		if e.ChildSize > 1 {
			clusters[e.Child] = true
		}
		clusters[e.Parent] = true
	}

	// Compute stability: sum of (death - birth) for each point
	for cluster := range clusters {
		var stab float64
		for _, e := range tree.Edges {
			if e.Parent == cluster && e.ChildSize == 1 {
				birth := birthLambda[cluster]
				stab += (e.Lambda - birth) * float64(e.ChildSize)
			}
		}
		stability[cluster] = stab
	}

	return stability
}

// extractClusters selects the final set of clusters using excess of mass.
// Works bottom-up: for each cluster, compare its stability to the sum of its
// children's stabilities. If the cluster is more stable, select it and deselect children.
//
// This produces a flat clustering from the hierarchical tree by cutting at the
// level that maximizes cluster persistence.
func extractClusters(tree condensedTree, stability map[int]float64, nSamples int) []int {
	labels := make([]int, nSamples)
	for i := range labels {
		labels[i] = -1 // Default: noise
	}

	if len(tree.Edges) == 0 {
		return labels
	}

	// Build parent-to-children mapping (only for clusters, not points)
	children := make(map[int][]int)
	for _, e := range tree.Edges {
		if e.ChildSize > 1 {
			children[e.Parent] = append(children[e.Parent], e.Child)
		}
	}

	// Find leaf clusters (clusters with no cluster children)
	leaves := make(map[int]bool)
	allClusters := make(map[int]bool)
	for _, e := range tree.Edges {
		if e.ChildSize > 1 {
			allClusters[e.Child] = true
		}
		allClusters[e.Parent] = true
	}
	for c := range allClusters {
		if len(children[c]) == 0 && c >= nSamples {
			leaves[c] = true
		}
	}

	// Track selected clusters and subtree stability
	selected := make(map[int]bool)
	subtreeStability := make(map[int]float64)
	for k, v := range stability {
		subtreeStability[k] = v
	}

	// Process clusters bottom-up (highest IDs first = deepest in tree)
	var clusters []int
	for c := range allClusters {
		clusters = append(clusters, c)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(clusters)))

	for _, cluster := range clusters {
		// Sum stability of all children
		childStab := 0.0
		for _, child := range children[cluster] {
			childStab += subtreeStability[child]
		}

		if stability[cluster] >= childStab {
			// This cluster is more stable: select it, deselect descendants
			selected[cluster] = true
			for _, child := range children[cluster] {
				delete(selected, child)
				var removeDescendants func(int)
				removeDescendants = func(c int) {
					for _, desc := range children[c] {
						delete(selected, desc)
						removeDescendants(desc)
					}
				}
				removeDescendants(child)
			}
			subtreeStability[cluster] = stability[cluster]
		} else {
			// Children are more stable: propagate their stability up
			subtreeStability[cluster] = childStab
		}
	}

	// Map points in clusters to point indices (unused but kept for reference)
	clusterPoints := make(map[int][]int)
	for _, e := range tree.Edges {
		if e.ChildSize == 1 {
			clusterPoints[e.Parent] = append(clusterPoints[e.Parent], e.Child)
		}
	}

	// Assign labels to points in selected clusters
	labelID := 0
	for cluster := range selected {
		points := collectClusterPoints(tree, cluster, nSamples)
		for _, pt := range points {
			if pt >= 0 && pt < nSamples {
				labels[pt] = labelID
			}
		}
		labelID++
	}

	return labels
}

// collectClusterPoints recursively gathers all data points belonging to a cluster.
// Traverses the condensed tree to find all leaf points (ChildSize == 1) under the given cluster.
func collectClusterPoints(tree condensedTree, cluster int, nSamples int) []int {
	var points []int

	// Build adjacency: parent -> list of edges
	children := make(map[int][]condensedEdge)
	for _, e := range tree.Edges {
		children[e.Parent] = append(children[e.Parent], e)
	}

	var collect func(c int)
	collect = func(c int) {
		for _, e := range children[c] {
			if e.ChildSize == 1 {
				points = append(points, e.Child)
			} else {
				collect(e.Child)
			}
		}
	}

	collect(cluster)
	return points
}

// computeProbabilities calculates cluster membership confidence for each point.
// Probability is based on how long a point persisted in its cluster relative to
// the longest-persisting point in that cluster.
//
// Points that joined early and stayed late have probability close to 1.
// Points that fell out early have lower probability.
// Noise points (label == -1) have probability 0.
func computeProbabilities(tree condensedTree, labels []int) []float64 {
	n := len(labels)
	probs := make([]float64, n)

	if len(tree.Edges) == 0 {
		return probs
	}

	// Track each point's lambda and cluster assignment
	maxLambdaPerCluster := make(map[int]float64)
	pointLambda := make(map[int]float64)
	pointCluster := make(map[int]int)

	for _, e := range tree.Edges {
		if e.ChildSize == 1 {
			pointLambda[e.Child] = e.Lambda
			pointCluster[e.Child] = e.Parent
			if e.Lambda > maxLambdaPerCluster[e.Parent] {
				maxLambdaPerCluster[e.Parent] = e.Lambda
			}
		}
	}

	// Probability = point's lambda / max lambda in cluster
	for i := 0; i < n; i++ {
		if labels[i] == -1 {
			probs[i] = 0
			continue
		}

		cluster := pointCluster[i]
		maxLambda := maxLambdaPerCluster[cluster]
		if maxLambda > 0 {
			probs[i] = pointLambda[i] / maxLambda
		} else {
			probs[i] = 1.0
		}

		if probs[i] > 1.0 {
			probs[i] = 1.0
		}
	}

	return probs
}
