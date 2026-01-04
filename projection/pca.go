// Package projection provides dimensionality reduction for high-dimensional embedding vectors.
//
// # Principal Component Analysis (PCA) Overview
//
// PCA is a technique that reduces high-dimensional data (like 768-dimensional text embeddings)
// down to fewer dimensions (like 2D for visualization) while preserving as much variance as possible.
//
// The key insight is that most high-dimensional data lies on or near a lower-dimensional subspace.
// PCA finds this subspace by identifying the directions (principal components) along which the data
// varies the most.
//
// # Why We Use Singular Value Decomposition (SVD)
//
// While PCA can be computed by finding eigenvectors of the covariance matrix, SVD is numerically
// more stable and efficient. For a centered data matrix X, the right singular vectors (V) give us
// the principal components directly, without needing to compute X^T * X explicitly.
//
// The mathematical relationship is:
//   - X = U * Σ * V^T  (SVD decomposition)
//   - The columns of V are the principal components (directions of maximum variance)
//   - The singular values in Σ indicate how much variance each component captures
//   - Projecting data: X_projected = X * V[:, 0:k] gives us the k-dimensional representation
package projection

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Point2D represents a single data point projected into 2D space for visualization.
// It preserves the original text label for display in the UI.
type Point2D struct {
	X, Y float64
	Text string
}

// ProjectTo2D reduces high-dimensional embedding vectors to 2D points using PCA.
// Each input vector (typically 768 dimensions from text embeddings) is transformed
// into a 2D point that can be plotted, while preserving the relative distances
// and clustering structure of the original high-dimensional space.
//
// Parameters:
//   - embeddingVectors: slice of high-dimensional vectors (e.g., from Ollama embeddings)
//   - textLabels: corresponding text labels for each vector
//
// Returns:
//   - slice of Point2D structs ready for 2D visualization
func ProjectTo2D(embeddingVectors [][]float32, textLabels []string) []Point2D {
	if len(embeddingVectors) == 0 {
		return nil
	}

	numberOfVectors := len(embeddingVectors)
	embeddingDimension := len(embeddingVectors[0])

	// We need at least 2 dimensions to project to 2D
	if embeddingDimension < 2 {
		return nil
	}

	// Validate that all vectors have the same dimensionality
	if !allVectorsHaveSameDimension(embeddingVectors, embeddingDimension) {
		return createFallbackProjection(embeddingVectors, textLabels)
	}

	// Step 1: Convert input vectors to a matrix format suitable for linear algebra operations
	dataMatrix := convertVectorsToMatrix(embeddingVectors, numberOfVectors, embeddingDimension)

	// Step 2: Center the data by subtracting the mean of each dimension
	// This is a crucial preprocessing step for PCA - it ensures that the first
	// principal component passes through the centroid of the data
	centerDataMatrixBySubtractingColumnMeans(dataMatrix, numberOfVectors, embeddingDimension)

	// Step 3: Compute SVD and extract the top 2 principal components
	principalComponentMatrix, svdSucceeded := computePrincipalComponentsUsingSVD(dataMatrix, embeddingDimension)
	if !svdSucceeded {
		return createFallbackProjection(embeddingVectors, textLabels)
	}

	// Step 4: Project the centered data onto the 2D subspace defined by the principal components
	projectedCoordinates := projectDataOntoPrincipalComponents(dataMatrix, principalComponentMatrix)

	// Step 5: Convert the projected matrix back to Point2D structs with labels
	return convertProjectedMatrixToPoints(projectedCoordinates, textLabels, numberOfVectors)
}

// allVectorsHaveSameDimension checks that every vector has the expected dimensionality.
// Inconsistent dimensions would cause matrix operations to fail or produce incorrect results.
func allVectorsHaveSameDimension(vectors [][]float32, expectedDimension int) bool {
	for _, vector := range vectors {
		if len(vector) != expectedDimension {
			return false
		}
	}
	return true
}

// convertVectorsToMatrix transforms a slice of float32 vectors into a gonum Dense matrix.
// The matrix has shape (numberOfVectors x embeddingDimension), where each row is one embedding.
// We convert from float32 to float64 because gonum operates on float64 for numerical precision.
func convertVectorsToMatrix(vectors [][]float32, numberOfVectors int, embeddingDimension int) *mat.Dense {
	// Allocate a flat slice to hold all matrix data in row-major order
	flattenedMatrixData := make([]float64, numberOfVectors*embeddingDimension)

	for rowIndex, vector := range vectors {
		for columnIndex, value := range vector {
			flatIndex := rowIndex*embeddingDimension + columnIndex
			flattenedMatrixData[flatIndex] = float64(value)
		}
	}

	return mat.NewDense(numberOfVectors, embeddingDimension, flattenedMatrixData)
}

// centerDataMatrixBySubtractingColumnMeans modifies the matrix in-place to have zero mean
// for each column (dimension).
//
// Why centering matters:
// PCA finds directions of maximum variance. If data isn't centered, the first principal
// component might just point toward the data's center rather than capturing the direction
// of maximum spread. Centering ensures we're measuring variance around the mean.
//
// Example: If all embeddings have dimension[0] ≈ 0.5, centering shifts them so dimension[0] ≈ 0,
// allowing PCA to focus on how points differ from each other rather than their absolute positions.
func centerDataMatrixBySubtractingColumnMeans(dataMatrix *mat.Dense, numberOfVectors int, embeddingDimension int) {
	// Calculate the mean value for each dimension across all vectors
	columnMeans := calculateColumnMeans(dataMatrix, embeddingDimension)

	// Subtract the mean from each element to center the data
	for rowIndex := 0; rowIndex < numberOfVectors; rowIndex++ {
		for columnIndex := 0; columnIndex < embeddingDimension; columnIndex++ {
			originalValue := dataMatrix.At(rowIndex, columnIndex)
			centeredValue := originalValue - columnMeans[columnIndex]
			dataMatrix.Set(rowIndex, columnIndex, centeredValue)
		}
	}
}

// calculateColumnMeans computes the arithmetic mean of each column (dimension) in the matrix.
func calculateColumnMeans(dataMatrix *mat.Dense, embeddingDimension int) []float64 {
	columnMeans := make([]float64, embeddingDimension)

	for columnIndex := 0; columnIndex < embeddingDimension; columnIndex++ {
		columnValues := mat.Col(nil, columnIndex, dataMatrix)
		columnMeans[columnIndex] = stat.Mean(columnValues, nil)
	}

	return columnMeans
}

// computePrincipalComponentsUsingSVD performs Singular Value Decomposition on the centered
// data matrix and extracts the first two principal components.
//
// SVD decomposes our data matrix X into: X = U * Σ * V^T
//
// Where:
//   - U (left singular vectors): represents how each data point relates to each component
//   - Σ (singular values): diagonal matrix showing the "importance" of each component
//   - V^T (right singular vectors): the principal component directions we want
//
// The columns of V (rows of V^T) are the principal components, ordered by the amount
// of variance they capture. We take the first two columns to get a 2D projection.
//
// Returns:
//   - principalComponentMatrix: (embeddingDimension x 2) matrix where each column is a principal component
//   - success: whether SVD computation succeeded
func computePrincipalComponentsUsingSVD(centeredDataMatrix *mat.Dense, embeddingDimension int) (*mat.Dense, bool) {
	var svdDecomposition mat.SVD

	// Compute the thin SVD (more efficient than full SVD when we only need top components)
	svdSucceeded := svdDecomposition.Factorize(centeredDataMatrix, mat.SVDThin)
	if !svdSucceeded {
		return nil, false
	}

	// Extract V^T (the right singular vectors, transposed)
	var rightSingularVectorsTransposed mat.Dense
	svdDecomposition.VTo(&rightSingularVectorsTransposed)

	// Validate that we have enough components for 2D projection
	numberOfRows, numberOfColumns := rightSingularVectorsTransposed.Dims()
	if numberOfRows < embeddingDimension || numberOfColumns < 2 {
		return nil, false
	}

	// Extract the first two columns of V (first two principal components)
	// These are the directions along which the data varies most
	return extractFirstTwoPrincipalComponents(rightSingularVectorsTransposed, embeddingDimension), true
}

// extractFirstTwoPrincipalComponents creates a (embeddingDimension x 2) matrix containing
// only the first two principal components from the full V matrix.
//
// The first principal component (column 0) captures the direction of maximum variance.
// The second principal component (column 1) captures the direction of maximum variance
// that is orthogonal (perpendicular) to the first component.
func extractFirstTwoPrincipalComponents(rightSingularVectors mat.Dense, embeddingDimension int) *mat.Dense {
	// Create a matrix to hold just the first two principal components
	principalComponentMatrix := mat.NewDense(embeddingDimension, 2, nil)

	for dimensionIndex := 0; dimensionIndex < embeddingDimension; dimensionIndex++ {
		firstComponentValue := rightSingularVectors.At(dimensionIndex, 0)
		secondComponentValue := rightSingularVectors.At(dimensionIndex, 1)

		principalComponentMatrix.Set(dimensionIndex, 0, firstComponentValue)
		principalComponentMatrix.Set(dimensionIndex, 1, secondComponentValue)
	}

	return principalComponentMatrix
}

// projectDataOntoPrincipalComponents multiplies the centered data matrix by the principal
// component matrix to get 2D coordinates for each data point.
//
// Mathematically: ProjectedData = CenteredData × PrincipalComponents
//
// Where:
//   - CenteredData is (numberOfVectors x embeddingDimension)
//   - PrincipalComponents is (embeddingDimension x 2)
//   - ProjectedData is (numberOfVectors x 2) - our 2D coordinates!
//
// Each row of the result contains the (x, y) coordinates for one embedding in 2D space.
func projectDataOntoPrincipalComponents(centeredDataMatrix *mat.Dense, principalComponentMatrix *mat.Dense) *mat.Dense {
	var projectedCoordinates mat.Dense
	projectedCoordinates.Mul(centeredDataMatrix, principalComponentMatrix)
	return &projectedCoordinates
}

// convertProjectedMatrixToPoints transforms the projected coordinate matrix into a slice
// of Point2D structs, attaching the original text labels for UI display.
func convertProjectedMatrixToPoints(projectedCoordinates *mat.Dense, textLabels []string, numberOfVectors int) []Point2D {
	points := make([]Point2D, numberOfVectors)

	for vectorIndex := 0; vectorIndex < numberOfVectors; vectorIndex++ {
		xCoordinate := projectedCoordinates.At(vectorIndex, 0)
		yCoordinate := projectedCoordinates.At(vectorIndex, 1)

		labelText := getTextLabelAtIndex(textLabels, vectorIndex)

		points[vectorIndex] = Point2D{
			X:    xCoordinate,
			Y:    yCoordinate,
			Text: labelText,
		}
	}

	return points
}

// getTextLabelAtIndex safely retrieves a text label, returning empty string if index is out of bounds.
func getTextLabelAtIndex(textLabels []string, index int) string {
	if index < len(textLabels) {
		return textLabels[index]
	}
	return ""
}

// createFallbackProjection provides a simple projection when PCA fails.
// It simply uses the first two dimensions of each vector as x and y coordinates.
// This is a naive approach but ensures we always return some visualization
// rather than failing completely.
//
// This fallback might be triggered when:
//   - Vectors have inconsistent dimensions
//   - SVD fails to converge (rare with well-formed data)
//   - There aren't enough dimensions in the V matrix
func createFallbackProjection(embeddingVectors [][]float32, textLabels []string) []Point2D {
	points := make([]Point2D, len(embeddingVectors))

	for vectorIndex, vector := range embeddingVectors {
		xCoordinate := getVectorComponentOrZero(vector, 0)
		yCoordinate := getVectorComponentOrZero(vector, 1)
		labelText := getTextLabelAtIndex(textLabels, vectorIndex)

		points[vectorIndex] = Point2D{
			X:    xCoordinate,
			Y:    yCoordinate,
			Text: labelText,
		}
	}

	return points
}

// getVectorComponentOrZero safely retrieves a component from a vector,
// returning 0.0 if the index is out of bounds.
func getVectorComponentOrZero(vector []float32, componentIndex int) float64 {
	if componentIndex < len(vector) {
		return float64(vector[componentIndex])
	}
	return 0.0
}
