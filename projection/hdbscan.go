package projection

import (
	"math"
	"sort"
)

type HDBSCANConfig struct {
	MinClusterSize int
	MinSamples     int
}

func DefaultHDBSCANConfig() HDBSCANConfig {
	return HDBSCANConfig{
		MinClusterSize: 5,
		MinSamples:     0,
	}
}

type ClusterResult struct {
	Labels        []int
	Probabilities []float64
}

func ClusterLabels(vectors [][]float32, config HDBSCANConfig) []int {
	return Cluster(vectors, config).Labels
}

func Cluster(vectors [][]float32, config HDBSCANConfig) ClusterResult {
	n := len(vectors)
	if n == 0 {
		return ClusterResult{}
	}

	if config.MinClusterSize < 2 {
		config.MinClusterSize = 5
	}
	if config.MinSamples <= 0 {
		config.MinSamples = config.MinClusterSize
	}

	if n < config.MinClusterSize {
		labels := make([]int, n)
		probs := make([]float64, n)
		for i := range labels {
			labels[i] = -1
		}
		return ClusterResult{Labels: labels, Probabilities: probs}
	}

	data := convertToFloat64(vectors)

	coreDistances := computeCoreDistances(data, config.MinSamples)
	mstEdges := computeMutualReachabilityMST(data, coreDistances)
	linkage := singleLinkageTree(mstEdges, n)
	condensed := condenseTree(linkage, config.MinClusterSize, n)
	stability := computeStability(condensed)
	labels := extractClusters(condensed, stability, n)
	probs := computeProbabilities(condensed, labels)

	return ClusterResult{Labels: labels, Probabilities: probs}
}

type mstEdge struct {
	From, To int
	Weight   float64
}

type linkageNode struct {
	Left, Right int
	Distance    float64
	Size        int
}

type condensedEdge struct {
	Parent    int
	Child     int
	Lambda    float64
	ChildSize int
}

type condensedTree struct {
	Edges       []condensedEdge
	RootCluster int
}

func computeCoreDistances(data [][]float64, minSamples int) []float64 {
	n := len(data)
	coreDistances := make([]float64, n)

	if minSamples > n {
		minSamples = n
	}

	for i := 0; i < n; i++ {
		dists := make([]float64, n)
		for j := 0; j < n; j++ {
			dists[j] = euclideanDistance(data[i], data[j])
		}
		sort.Float64s(dists)

		k := minSamples
		if k >= n {
			k = n - 1
		}
		coreDistances[i] = dists[k]
	}

	return coreDistances
}

func computeMutualReachabilityMST(data [][]float64, coreDistances []float64) []mstEdge {
	n := len(data)
	if n < 2 {
		return nil
	}

	inTree := make([]bool, n)
	minDist := make([]float64, n)
	minEdge := make([]int, n)

	for i := range minDist {
		minDist[i] = math.Inf(1)
		minEdge[i] = -1
	}

	edges := make([]mstEdge, 0, n-1)
	current := 0
	inTree[current] = true

	for added := 1; added < n; added++ {
		for j := 0; j < n; j++ {
			if inTree[j] {
				continue
			}

			dist := euclideanDistance(data[current], data[j])
			mrd := math.Max(coreDistances[current], math.Max(coreDistances[j], dist))

			if mrd < minDist[j] {
				minDist[j] = mrd
				minEdge[j] = current
			}
		}

		minIdx := -1
		minVal := math.Inf(1)
		for j := 0; j < n; j++ {
			if !inTree[j] && minDist[j] < minVal {
				minVal = minDist[j]
				minIdx = j
			}
		}

		if minIdx < 0 {
			break
		}

		edges = append(edges, mstEdge{
			From:   minEdge[minIdx],
			To:     minIdx,
			Weight: minVal,
		})
		inTree[minIdx] = true
		current = minIdx
	}

	sort.Slice(edges, func(i, j int) bool {
		return edges[i].Weight < edges[j].Weight
	})

	return edges
}

type unionFind struct {
	parent []int
	size   []int
	next   int
}

func newUnionFind(n int) *unionFind {
	parent := make([]int, 2*n-1)
	size := make([]int, 2*n-1)
	for i := 0; i < n; i++ {
		parent[i] = i
		size[i] = 1
	}
	return &unionFind{parent: parent, size: size, next: n}
}

func (uf *unionFind) find(x int) int {
	root := x
	for uf.parent[root] != root {
		root = uf.parent[root]
	}
	for uf.parent[x] != root {
		next := uf.parent[x]
		uf.parent[x] = root
		x = next
	}
	return root
}

func (uf *unionFind) union(x, y int) int {
	newNode := uf.next
	uf.parent[x] = newNode
	uf.parent[y] = newNode
	uf.parent[newNode] = newNode
	uf.size[newNode] = uf.size[x] + uf.size[y]
	uf.next++
	return newNode
}

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

func condenseTree(tree []linkageNode, minClusterSize int, nSamples int) condensedTree {
	if len(tree) == 0 {
		return condensedTree{RootCluster: nSamples}
	}

	rootCluster := nSamples + len(tree) - 1
	relabel := make(map[int]int)
	nextLabel := nSamples

	var edges []condensedEdge

	var condense func(node int, parent int, lambdaVal float64)
	condense = func(node int, parent int, lambdaVal float64) {
		if node < nSamples {
			edges = append(edges, condensedEdge{
				Parent:    parent,
				Child:     node,
				Lambda:    lambdaVal,
				ChildSize: 1,
			})
			return
		}

		treeIdx := node - nSamples
		if treeIdx < 0 || treeIdx >= len(tree) {
			return
		}

		linkNode := tree[treeIdx]
		left := linkNode.Left
		right := linkNode.Right

		leftSize := 1
		rightSize := 1
		if left >= nSamples {
			idx := left - nSamples
			if idx >= 0 && idx < len(tree) {
				leftSize = tree[idx].Size
			}
		}
		if right >= nSamples {
			idx := right - nSamples
			if idx >= 0 && idx < len(tree) {
				rightSize = tree[idx].Size
			}
		}

		newLambda := 0.0
		if linkNode.Distance > 0 {
			newLambda = 1.0 / linkNode.Distance
		}

		leftValid := leftSize >= minClusterSize
		rightValid := rightSize >= minClusterSize

		if leftValid && rightValid {
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
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		} else if rightValid {
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		} else {
			condense(left, parent, lambdaVal)
			condense(right, parent, lambdaVal)
		}
	}

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

func computeStability(tree condensedTree) map[int]float64 {
	stability := make(map[int]float64)

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

	clusters := make(map[int]bool)
	for _, e := range tree.Edges {
		if e.ChildSize > 1 {
			clusters[e.Child] = true
		}
		clusters[e.Parent] = true
	}

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

func extractClusters(tree condensedTree, stability map[int]float64, nSamples int) []int {
	labels := make([]int, nSamples)
	for i := range labels {
		labels[i] = -1
	}

	if len(tree.Edges) == 0 {
		return labels
	}

	children := make(map[int][]int)
	for _, e := range tree.Edges {
		if e.ChildSize > 1 {
			children[e.Parent] = append(children[e.Parent], e.Child)
		}
	}

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

	selected := make(map[int]bool)
	subtreeStability := make(map[int]float64)
	for k, v := range stability {
		subtreeStability[k] = v
	}

	var clusters []int
	for c := range allClusters {
		clusters = append(clusters, c)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(clusters)))

	for _, cluster := range clusters {
		childStab := 0.0
		for _, child := range children[cluster] {
			childStab += subtreeStability[child]
		}

		if stability[cluster] >= childStab {
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
			subtreeStability[cluster] = childStab
		}
	}

	clusterPoints := make(map[int][]int)
	for _, e := range tree.Edges {
		if e.ChildSize == 1 {
			clusterPoints[e.Parent] = append(clusterPoints[e.Parent], e.Child)
		}
	}

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

func collectClusterPoints(tree condensedTree, cluster int, nSamples int) []int {
	var points []int

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

func computeProbabilities(tree condensedTree, labels []int) []float64 {
	n := len(labels)
	probs := make([]float64, n)

	if len(tree.Edges) == 0 {
		return probs
	}

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
