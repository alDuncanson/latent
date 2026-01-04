package projection

import (
	"math"
	"testing"
)

func TestProjectTo2DUMAP_EmptyInput(t *testing.T) {
	result := ProjectTo2DUMAP(nil, nil)
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
}

func TestProjectTo2DUMAP_SinglePoint(t *testing.T) {
	vectors := [][]float32{{1.0, 2.0, 3.0}}
	labels := []string{"point1"}

	result := ProjectTo2DUMAP(vectors, labels)
	if len(result) != 1 {
		t.Fatalf("expected 1 point, got %d", len(result))
	}
}

func TestProjectTo2DUMAP_SmallCluster(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0, 0.0},
		{0.1, 0.1, 0.1},
		{0.2, 0.2, 0.2},
		{10.0, 10.0, 10.0},
		{10.1, 10.1, 10.1},
		{10.2, 10.2, 10.2},
	}
	labels := []string{"a1", "a2", "a3", "b1", "b2", "b3"}

	config := DefaultUMAPConfig()
	config.NNeighbors = 2
	config.NEpochs = 50

	result := ProjectTo2DUMAPWithConfig(vectors, labels, config)

	if len(result) != 6 {
		t.Fatalf("expected 6 points, got %d", len(result))
	}

	for i, p := range result {
		if math.IsNaN(p.X) || math.IsNaN(p.Y) {
			t.Errorf("point %d has NaN coordinates", i)
		}
		if math.IsInf(p.X, 0) || math.IsInf(p.Y, 0) {
			t.Errorf("point %d has Inf coordinates", i)
		}
	}

	for i, p := range result {
		if p.Text != labels[i] {
			t.Errorf("point %d has wrong label: expected %s, got %s", i, labels[i], p.Text)
		}
	}
}

func TestComputeKNN(t *testing.T) {
	data := [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{2.0, 0.0},
		{3.0, 0.0},
	}

	knn := computeKNN(data, 2)

	if len(knn.Indices) != 4 {
		t.Fatalf("expected 4 index sets, got %d", len(knn.Indices))
	}

	// Point 0's nearest neighbor should be point 1
	if knn.Indices[0][0] != 1 {
		t.Errorf("point 0's nearest should be 1, got %d", knn.Indices[0][0])
	}
}

func TestSmoothKNNDist(t *testing.T) {
	distances := [][]float64{
		{1.0, 2.0, 3.0},
		{0.5, 1.5, 2.5},
		{2.0, 4.0, 6.0},
	}

	sigmas, rhos := smoothKNNDist(distances, 3.0)

	if len(sigmas) != 3 || len(rhos) != 3 {
		t.Fatalf("wrong output lengths")
	}

	for i, s := range sigmas {
		if s <= 0 {
			t.Errorf("sigma[%d] should be positive, got %f", i, s)
		}
	}
}

func TestFindABParams(t *testing.T) {
	a, b := findABParams(1.0, 0.1)

	if a <= 0 || b <= 0 {
		t.Errorf("a and b should be positive, got a=%f, b=%f", a, b)
	}

	// Check the curve at min_dist should be close to 1
	atMinDist := 1.0 / (1.0 + a*math.Pow(0.1, 2*b))
	if atMinDist < 0.8 {
		t.Errorf("curve at min_dist should be close to 1, got %f", atMinDist)
	}
}

func TestEuclideanDistance(t *testing.T) {
	a := []float64{0.0, 0.0, 0.0}
	b := []float64{3.0, 4.0, 0.0}

	dist := euclideanDistance(a, b)
	if math.Abs(dist-5.0) > 1e-10 {
		t.Errorf("expected distance 5.0, got %f", dist)
	}
}

func TestClip(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.0},
		{3.0, 3.0},
		{5.0, 4.0},
		{-5.0, -4.0},
		{100.0, 4.0},
	}

	for _, tc := range tests {
		result := clip(tc.input)
		if result != tc.expected {
			t.Errorf("clip(%f) = %f, expected %f", tc.input, result, tc.expected)
		}
	}
}

func TestProjectTo2DUMAP_Reproducibility(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{1.1, 2.1, 3.1, 4.1},
		{5.0, 6.0, 7.0, 8.0},
		{5.1, 6.1, 7.1, 8.1},
	}
	labels := []string{"a", "b", "c", "d"}

	config := DefaultUMAPConfig()
	config.NEpochs = 20
	config.NNeighbors = 2

	result1 := ProjectTo2DUMAPWithConfig(vectors, labels, config)
	result2 := ProjectTo2DUMAPWithConfig(vectors, labels, config)

	for i := range result1 {
		if result1[i].X != result2[i].X || result1[i].Y != result2[i].Y {
			t.Errorf("point %d differs between runs: (%f,%f) vs (%f,%f)",
				i, result1[i].X, result1[i].Y, result2[i].X, result2[i].Y)
		}
	}
}

func BenchmarkProjectTo2DUMAP(b *testing.B) {
	vectors := make([][]float32, 100)
	labels := make([]string, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i*128 + j)
		}
		labels[i] = "label"
	}

	config := DefaultUMAPConfig()
	config.NEpochs = 50
	config.NNeighbors = 10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ProjectTo2DUMAPWithConfig(vectors, labels, config)
	}
}
