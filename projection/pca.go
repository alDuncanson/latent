package projection

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type Point2D struct {
	X, Y float64
	Text string
}

func ProjectTo2D(vectors [][]float32, texts []string) []Point2D {
	if len(vectors) == 0 {
		return nil
	}

	n := len(vectors)
	dim := len(vectors[0])

	if dim < 2 {
		return nil
	}

	for _, v := range vectors {
		if len(v) != dim {
			return fallbackProjection(vectors, texts)
		}
	}

	data := make([]float64, n*dim)
	for i, v := range vectors {
		for j, val := range v {
			data[i*dim+j] = float64(val)
		}
	}

	X := mat.NewDense(n, dim, data)

	means := make([]float64, dim)
	for j := 0; j < dim; j++ {
		col := mat.Col(nil, j, X)
		means[j] = stat.Mean(col, nil)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			X.Set(i, j, X.At(i, j)-means[j])
		}
	}

	var svd mat.SVD
	ok := svd.Factorize(X, mat.SVDThin)
	if !ok {
		return fallbackProjection(vectors, texts)
	}

	var vt mat.Dense
	svd.VTo(&vt)

	vtr, vtc := vt.Dims()
	if vtr < 2 || vtc < dim {
		return fallbackProjection(vectors, texts)
	}

	pc := mat.NewDense(dim, 2, nil)
	for i := 0; i < dim && i < vtc; i++ {
		pc.Set(i, 0, vt.At(0, i))
		pc.Set(i, 1, vt.At(1, i))
	}

	var projected mat.Dense
	projected.Mul(X, pc)

	points := make([]Point2D, n)
	for i := 0; i < n; i++ {
		text := ""
		if i < len(texts) {
			text = texts[i]
		}
		points[i] = Point2D{
			X:    projected.At(i, 0),
			Y:    projected.At(i, 1),
			Text: text,
		}
	}

	return points
}

func fallbackProjection(vectors [][]float32, texts []string) []Point2D {
	points := make([]Point2D, len(vectors))
	for i, v := range vectors {
		x, y := 0.0, 0.0
		if len(v) > 0 {
			x = float64(v[0])
		}
		if len(v) > 1 {
			y = float64(v[1])
		}
		text := ""
		if i < len(texts) {
			text = texts[i]
		}
		points[i] = Point2D{X: x, Y: y, Text: text}
	}
	return points
}
