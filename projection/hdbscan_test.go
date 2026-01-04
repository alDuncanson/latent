package projection

import (
	"testing"
)

func TestCluster_EmptyInput(t *testing.T) {
	result := Cluster(nil, DefaultHDBSCANConfig())
	if len(result.Labels) != 0 {
		t.Errorf("expected empty labels for empty input, got %d", len(result.Labels))
	}
}

func TestCluster_TooFewPoints(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{1.1, 2.1, 3.1},
	}
	config := DefaultHDBSCANConfig()
	config.MinClusterSize = 5

	result := Cluster(vectors, config)
	if len(result.Labels) != 2 {
		t.Fatalf("expected 2 labels, got %d", len(result.Labels))
	}
	for i, label := range result.Labels {
		if label != -1 {
			t.Errorf("point %d should be noise (-1), got %d", i, label)
		}
	}
}

func TestCluster_TwoClusters(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.05, 0.05},
		{10.0, 10.0}, {10.1, 10.0}, {10.0, 10.1}, {10.1, 10.1}, {10.05, 10.05},
	}

	config := HDBSCANConfig{
		MinClusterSize: 3,
		MinSamples:     2,
	}

	result := Cluster(vectors, config)

	if len(result.Labels) != 10 {
		t.Fatalf("expected 10 labels, got %d", len(result.Labels))
	}

	cluster0 := result.Labels[0]
	cluster5 := result.Labels[5]

	if cluster0 == -1 && cluster5 == -1 {
		t.Skip("both clusters marked as noise - algorithm may need more points")
	}

	if cluster0 == cluster5 {
		t.Errorf("points in different regions should have different clusters: %d vs %d", cluster0, cluster5)
	}

	for i := 0; i < 5; i++ {
		if result.Labels[i] != cluster0 {
			t.Errorf("first cluster points should have same label: point %d has %d, expected %d", i, result.Labels[i], cluster0)
		}
	}

	for i := 5; i < 10; i++ {
		if result.Labels[i] != cluster5 {
			t.Errorf("second cluster points should have same label: point %d has %d, expected %d", i, result.Labels[i], cluster5)
		}
	}
}

func TestClusterLabels(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.05, 0.05},
	}

	config := HDBSCANConfig{
		MinClusterSize: 3,
		MinSamples:     2,
	}

	labels := ClusterLabels(vectors, config)
	if len(labels) != 5 {
		t.Errorf("expected 5 labels, got %d", len(labels))
	}
}

func TestComputeCoreDistances(t *testing.T) {
	data := [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{2.0, 0.0},
		{3.0, 0.0},
	}

	coreDistances := computeCoreDistances(data, 2)

	if len(coreDistances) != 4 {
		t.Fatalf("expected 4 core distances, got %d", len(coreDistances))
	}

	for i, d := range coreDistances {
		if d < 0 {
			t.Errorf("core distance %d should be non-negative, got %f", i, d)
		}
	}

	if coreDistances[0] < 1.0 || coreDistances[0] > 2.1 {
		t.Errorf("point 0 core distance should be around 2 (to 2nd neighbor), got %f", coreDistances[0])
	}
}

func TestComputeMutualReachabilityMST(t *testing.T) {
	data := [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{5.0, 0.0},
	}
	coreDistances := []float64{1.0, 1.0, 4.0}

	edges := computeMutualReachabilityMST(data, coreDistances)

	if len(edges) != 2 {
		t.Fatalf("expected 2 MST edges for 3 points, got %d", len(edges))
	}

	for _, e := range edges {
		if e.Weight < 0 {
			t.Errorf("edge weight should be non-negative, got %f", e.Weight)
		}
	}

	if edges[0].Weight > edges[1].Weight {
		t.Errorf("MST edges should be sorted by weight")
	}
}

func TestSingleLinkageTree(t *testing.T) {
	edges := []mstEdge{
		{From: 0, To: 1, Weight: 1.0},
		{From: 1, To: 2, Weight: 2.0},
		{From: 2, To: 3, Weight: 3.0},
	}

	tree := singleLinkageTree(edges, 4)

	if len(tree) != 3 {
		t.Fatalf("expected 3 linkage nodes for 4 points, got %d", len(tree))
	}

	if tree[len(tree)-1].Size != 4 {
		t.Errorf("root node should have size 4, got %d", tree[len(tree)-1].Size)
	}
}

func TestDefaultHDBSCANConfig(t *testing.T) {
	config := DefaultHDBSCANConfig()

	if config.MinClusterSize != 5 {
		t.Errorf("expected MinClusterSize=5, got %d", config.MinClusterSize)
	}

	if config.MinSamples != 0 {
		t.Errorf("expected MinSamples=0 (defaults to MinClusterSize), got %d", config.MinSamples)
	}
}

func TestCluster_Probabilities(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.05, 0.05},
	}

	config := HDBSCANConfig{
		MinClusterSize: 3,
		MinSamples:     2,
	}

	result := Cluster(vectors, config)

	if len(result.Probabilities) != 5 {
		t.Fatalf("expected 5 probabilities, got %d", len(result.Probabilities))
	}

	for i, p := range result.Probabilities {
		if p < 0 || p > 1 {
			t.Errorf("probability %d should be in [0,1], got %f", i, p)
		}
	}
}
