package tui

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/alDuncanson/latent/ollama"
	"github.com/alDuncanson/latent/projection"
	"github.com/alDuncanson/latent/qdrant"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/google/uuid"
)

// Model represents the main application state for the TUI embedding visualization.
type Model struct {
	width, height  int
	input          string
	cursorPos      int
	points         []projection.Point2D
	storedPoints   []qdrant.Point
	currentVec     []float32
	ollama         *ollama.Client
	qdrant         *qdrant.Client
	err            error
	debounceTimer  *time.Timer
	embedding      bool
	savedTexts     []string
	selectedIndex  int
	showMetadata   bool
	focusMode      bool
	version        string
}

// embeddingResult is the message returned after computing an embedding vector.
type embeddingResult struct {
	vector []float32
	err    error
}

// pointsUpdated is the message returned after points have been recalculated or loaded.
type pointsUpdated struct {
	points       []projection.Point2D
	storedPoints []qdrant.Point
}

// NewModel creates and initializes a new TUI Model with the given clients and version.
func NewModel(ollamaClient *ollama.Client, qdrantClient *qdrant.Client, version string) Model {
	return Model{
		ollama:        ollamaClient,
		qdrant:        qdrantClient,
		width:         80,
		height:        24,
		selectedIndex: -1,
		showMetadata:  true,
		version:       version,
	}
}

// Init initializes the model by loading existing points from the database.
func (model Model) Init() tea.Cmd {
	return model.loadPoints()
}

// loadPoints fetches all stored embeddings from Qdrant and projects them to 2D coordinates.
func (model *Model) loadPoints() tea.Cmd {
	qdrantClient := model.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		storedPoints, err := qdrantClient.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		// Extract vectors and text labels from stored points
		var embeddingVectors [][]float32
		var textLabels []string
		for _, storedPoint := range storedPoints {
			embeddingVectors = append(embeddingVectors, storedPoint.Vector)
			textLabels = append(textLabels, storedPoint.Text)
		}

		// Project high-dimensional vectors to 2D for visualization
		projectedPoints := projection.ProjectTo2D(embeddingVectors, textLabels)
		return pointsUpdated{points: projectedPoints, storedPoints: storedPoints}
	}
}

// Update handles all incoming messages and updates the model state accordingly.
func (model Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch message := msg.(type) {
	case tea.KeyMsg:
		return model.handleKeyPress(message)

	case tea.WindowSizeMsg:
		model.width = message.Width
		model.height = message.Height

	case embeddingResult:
		return model.handleEmbeddingResult(message)

	case pointsUpdated:
		return model.handlePointsUpdated(message)
	}

	return model, nil
}

// handleKeyPress processes keyboard input and returns the updated model and any commands.
func (model Model) handleKeyPress(keyMessage tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch keyMessage.String() {
	case "ctrl+c", "esc":
		return model, tea.Quit

	case "enter":
		return model.handleEnterKey()

	case "backspace":
		return model.handleBackspace()

	case "left":
		if model.cursorPos > 0 {
			model.cursorPos--
		}

	case "right":
		if model.cursorPos < len(model.input) {
			model.cursorPos++
		}

	case "tab":
		model.selectNextPoint()

	case "shift+tab":
		model.selectPreviousPoint()

	case "up":
		model.selectPreviousPoint()

	case "down":
		model.selectNextPoint()

	case "/":
		model.showMetadata = !model.showMetadata

	case "D":
		if model.selectedIndex >= 0 && model.selectedIndex < len(model.storedPoints) {
			return model, model.deleteSelected()
		}

	default:
		return model.handleCharacterInput(keyMessage)
	}

	return model, nil
}

// handleEnterKey saves the current embedding when Enter is pressed.
func (model Model) handleEnterKey() (tea.Model, tea.Cmd) {
	if model.input != "" && model.currentVec != nil {
		saveCommand := model.saveCurrentEmbedding()
		model.input = ""
		model.cursorPos = 0
		model.currentVec = nil
		return model, saveCommand
	}
	return model, nil
}

// handleBackspace removes the character before the cursor and triggers re-embedding.
func (model Model) handleBackspace() (tea.Model, tea.Cmd) {
	if model.cursorPos > 0 {
		model.input = model.input[:model.cursorPos-1] + model.input[model.cursorPos:]
		model.cursorPos--
		return model, model.debounceEmbed()
	}
	return model, nil
}

// selectNextPoint moves the selection to the next point in the list.
func (model *Model) selectNextPoint() {
	if len(model.storedPoints) > 0 {
		model.selectedIndex = (model.selectedIndex + 1) % len(model.storedPoints)
	}
}

// selectPreviousPoint moves the selection to the previous point in the list.
func (model *Model) selectPreviousPoint() {
	if len(model.storedPoints) > 0 {
		model.selectedIndex--
		if model.selectedIndex < 0 {
			model.selectedIndex = len(model.storedPoints) - 1
		}
	}
}

// handleCharacterInput inserts a typed character at the cursor position.
func (model Model) handleCharacterInput(keyMessage tea.KeyMsg) (tea.Model, tea.Cmd) {
	keyString := keyMessage.String()
	if len(keyString) == 1 {
		model.input = model.input[:model.cursorPos] + keyString + model.input[model.cursorPos:]
		model.cursorPos++
		return model, model.debounceEmbed()
	}
	return model, nil
}

// handleEmbeddingResult processes the result of an embedding computation.
func (model Model) handleEmbeddingResult(result embeddingResult) (tea.Model, tea.Cmd) {
	model.embedding = false
	if result.err != nil {
		model.err = result.err
		return model, nil
	}
	model.currentVec = result.vector
	model.err = nil
	if model.currentVec != nil {
		return model, model.updateVisualization()
	}
	return model, nil
}

// handlePointsUpdated processes updated point data and refreshes the visualization.
func (model Model) handlePointsUpdated(update pointsUpdated) (tea.Model, tea.Cmd) {
	model.points = update.points
	if update.storedPoints != nil {
		model.storedPoints = update.storedPoints
	}

	// Extract text labels from all points
	var textLabels []string
	for _, point := range model.points {
		textLabels = append(textLabels, point.Text)
	}
	model.savedTexts = textLabels

	// Ensure selected index stays within bounds
	if model.selectedIndex >= len(model.storedPoints) {
		model.selectedIndex = len(model.storedPoints) - 1
	}

	return model, nil
}

// debounceEmbed waits briefly before computing an embedding to avoid excessive API calls.
func (model *Model) debounceEmbed() tea.Cmd {
	inputText := model.input
	ollamaClient := model.ollama
	return func() tea.Msg {
		time.Sleep(150 * time.Millisecond)
		if inputText == "" {
			return embeddingResult{}
		}
		embeddingVector, err := ollamaClient.Embed(inputText)
		return embeddingResult{vector: embeddingVector, err: err}
	}
}

// saveCurrentEmbedding persists the current input and its embedding to the database.
func (model *Model) saveCurrentEmbedding() tea.Cmd {
	textToSave := model.input
	vectorToSave := model.currentVec
	qdrantClient := model.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		pointID := uuid.New().String()
		if err := qdrantClient.Upsert(ctx, pointID, textToSave, vectorToSave); err != nil {
			return embeddingResult{err: err}
		}

		// Reload all points after saving
		storedPoints, err := qdrantClient.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		// Extract vectors and text labels for projection
		var embeddingVectors [][]float32
		var textLabels []string
		for _, storedPoint := range storedPoints {
			embeddingVectors = append(embeddingVectors, storedPoint.Vector)
			textLabels = append(textLabels, storedPoint.Text)
		}

		projectedPoints := projection.ProjectTo2D(embeddingVectors, textLabels)
		return pointsUpdated{points: projectedPoints, storedPoints: storedPoints}
	}
}

// deleteSelected removes the currently selected point from the database.
func (model *Model) deleteSelected() tea.Cmd {
	if model.selectedIndex < 0 || model.selectedIndex >= len(model.storedPoints) {
		return nil
	}
	pointIDToDelete := model.storedPoints[model.selectedIndex].ID
	qdrantClient := model.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		if err := qdrantClient.Delete(ctx, pointIDToDelete); err != nil {
			return embeddingResult{err: err}
		}

		// Reload all points after deletion
		storedPoints, err := qdrantClient.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		// Extract vectors and text labels for projection
		var embeddingVectors [][]float32
		var textLabels []string
		for _, storedPoint := range storedPoints {
			embeddingVectors = append(embeddingVectors, storedPoint.Vector)
			textLabels = append(textLabels, storedPoint.Text)
		}

		projectedPoints := projection.ProjectTo2D(embeddingVectors, textLabels)
		return pointsUpdated{points: projectedPoints, storedPoints: storedPoints}
	}
}

// updateVisualization recalculates the 2D projection including the current input vector.
func (model *Model) updateVisualization() tea.Cmd {
	qdrantClient := model.qdrant
	currentVector := model.currentVec
	currentInput := model.input
	return func() tea.Msg {
		ctx := context.Background()
		storedPoints, err := qdrantClient.GetAll(ctx)
		if err != nil {
			return pointsUpdated{}
		}

		// Collect all stored vectors and their labels
		var embeddingVectors [][]float32
		var textLabels []string
		for _, storedPoint := range storedPoints {
			embeddingVectors = append(embeddingVectors, storedPoint.Vector)
			textLabels = append(textLabels, storedPoint.Text)
		}

		// Add the current input vector (not yet saved) to the visualization
		if currentVector != nil {
			embeddingVectors = append(embeddingVectors, currentVector)
			textLabels = append(textLabels, currentInput+" ●")
		}

		projectedPoints := projection.ProjectTo2D(embeddingVectors, textLabels)
		return pointsUpdated{points: projectedPoints}
	}
}

// View renders the complete UI as a string.
func (model Model) View() string {
	marginSize := 1
	gapSize := 1
	totalWidth := model.width - marginSize*2

	// Define UI styles
	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	helpStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	inputBorderStyle := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63")).Padding(0, 1)
	canvasBorderStyle := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("240"))
	metadataBorderStyle := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63")).Padding(0, 1)

	var outputBuilder strings.Builder

	// Render title
	outputBuilder.WriteString(titleStyle.Render("latent"))
	outputBuilder.WriteString("\n")

	// Prepare input display with optional embedding indicator
	inputDisplayText := model.input
	if model.embedding {
		inputDisplayText += " (embedding...)"
	}

	// Calculate canvas dimensions
	canvasHeight := model.height - 9
	if canvasHeight < 10 {
		canvasHeight = 10
	}

	// Determine whether to show the metadata panel
	shouldShowMetadata := model.showMetadata && model.selectedIndex >= 0 && model.selectedIndex < len(model.storedPoints)

	if shouldShowMetadata {
		// Layout with metadata panel on the right
		metadataPanelOuterWidth := 26
		metadataPanelInnerWidth := metadataPanelOuterWidth - 4
		canvasInnerWidth := totalWidth - metadataPanelOuterWidth - gapSize - 2

		outputBuilder.WriteString(inputBorderStyle.Width(totalWidth - 4).Render(inputDisplayText))
		outputBuilder.WriteString("\n")

		canvasContent := model.renderCanvas(canvasInnerWidth, canvasHeight)
		metadataContent := model.renderMetadata(metadataPanelInnerWidth, canvasHeight-1)

		leftPanel := canvasBorderStyle.Width(canvasInnerWidth).Render(canvasContent)
		rightPanel := metadataBorderStyle.Width(metadataPanelInnerWidth).Height(canvasHeight - 1).Render(metadataContent)

		outputBuilder.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, strings.Repeat(" ", gapSize), rightPanel))
	} else {
		// Full-width canvas layout without metadata panel
		canvasInnerWidth := totalWidth - 4

		outputBuilder.WriteString(inputBorderStyle.Width(canvasInnerWidth).Render(inputDisplayText))
		outputBuilder.WriteString("\n")

		canvasContent := model.renderCanvas(canvasInnerWidth, canvasHeight)
		outputBuilder.WriteString(canvasBorderStyle.Width(canvasInnerWidth).Render(canvasContent))
	}
	outputBuilder.WriteString("\n")

	// Render error message if present
	if model.err != nil {
		errorStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
		outputBuilder.WriteString(errorStyle.Render("Error: "+model.err.Error()) + "\n")
	}

	help := "Up/Down: select | /: metadata | F: focus | D: delete | Enter: save | Esc: quit"
	versionLabel := model.version
	padding := totalWidth - len(help) - len(versionLabel)
	if padding < 1 {
		padding = 1
	}
	outputBuilder.WriteString(helpStyle.Render(help + strings.Repeat(" ", padding) + versionLabel))

	return lipgloss.NewStyle().Padding(1, marginSize).Render(outputBuilder.String())
}

// renderMetadata generates the metadata panel content for the selected point.
func (model Model) renderMetadata(panelWidth, panelHeight int) string {
	if model.selectedIndex < 0 || model.selectedIndex >= len(model.storedPoints) {
		return ""
	}

	selectedPoint := model.storedPoints[model.selectedIndex]
	var contentLines []string

	// Define metadata panel styles
	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	valueStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("255"))

	// Section: Selected point info
	contentLines = append(contentLines, headerStyle.Render("Selected"))
	contentLines = append(contentLines, valueStyle.Render(truncateString(selectedPoint.Text, panelWidth)))
	contentLines = append(contentLines, "")

	// Section: Vector statistics
	if len(selectedPoint.Vector) > 0 {
		contentLines = append(contentLines, labelStyle.Render("Dim: ")+valueStyle.Render(fmt.Sprintf("%d", len(selectedPoint.Vector))))

		minimumValue, maximumValue, meanValue := computeVectorStatistics(selectedPoint.Vector)
		contentLines = append(contentLines, labelStyle.Render("Min/Max: ")+valueStyle.Render(fmt.Sprintf("%.3f / %.3f", minimumValue, maximumValue)))
		contentLines = append(contentLines, labelStyle.Render("Mean: ")+valueStyle.Render(fmt.Sprintf("%.4f", meanValue)))

		euclideanNorm := computeVectorNorm(selectedPoint.Vector)
		contentLines = append(contentLines, labelStyle.Render("L2 norm: ")+valueStyle.Render(fmt.Sprintf("%.4f", euclideanNorm)))
		contentLines = append(contentLines, "")
	}

	// Section: Nearest neighbors by cosine similarity
	nearestNeighbors := model.findNearestNeighbors(model.selectedIndex, 5)
	if len(nearestNeighbors) > 0 {
		contentLines = append(contentLines, headerStyle.Render("Nearest"))
		for _, neighborEntry := range nearestNeighbors {
			neighborLine := fmt.Sprintf("%.3f %s", neighborEntry.similarity, truncateString(neighborEntry.text, panelWidth-7))
			contentLines = append(contentLines, neighborLine)
		}
	}

	// Pad or truncate to fit panel height
	for len(contentLines) < panelHeight {
		contentLines = append(contentLines, "")
	}
	if len(contentLines) > panelHeight {
		contentLines = contentLines[:panelHeight]
	}

	return strings.Join(contentLines, "\n")
}

// neighbor represents a neighboring point with its similarity score.
type neighbor struct {
	text       string
	similarity float64
}

// findNearestNeighbors returns the k most similar points to the selected point.
func (model Model) findNearestNeighbors(selectedPointIndex int, maxNeighbors int) []neighbor {
	if selectedPointIndex < 0 || selectedPointIndex >= len(model.storedPoints) {
		return nil
	}

	selectedPoint := model.storedPoints[selectedPointIndex]
	var neighborList []neighbor

	// Calculate cosine similarity with all other points
	for pointIndex, candidatePoint := range model.storedPoints {
		if pointIndex == selectedPointIndex {
			continue
		}
		similarityScore := computeCosineSimilarity(selectedPoint.Vector, candidatePoint.Vector)
		neighborList = append(neighborList, neighbor{text: candidatePoint.Text, similarity: similarityScore})
	}

	// Sort by similarity in descending order (most similar first)
	sort.Slice(neighborList, func(firstIndex, secondIndex int) bool {
		return neighborList[firstIndex].similarity > neighborList[secondIndex].similarity
	})

	// Return only the top k neighbors
	if len(neighborList) > maxNeighbors {
		neighborList = neighborList[:maxNeighbors]
	}
	return neighborList
}

// findNearestNeighborIndices returns the indices of the k most similar points.
func (model Model) findNearestNeighborIndices(selectedPointIndex int, maxNeighbors int) []int {
	if selectedPointIndex < 0 || selectedPointIndex >= len(model.storedPoints) {
		return nil
	}

	selectedPoint := model.storedPoints[selectedPointIndex]

	// indexedNeighbor pairs an index with its similarity score for sorting
	type indexedNeighbor struct {
		pointIndex int
		similarity float64
	}
	var neighborList []indexedNeighbor

	// Calculate cosine similarity with all other points
	for pointIndex, candidatePoint := range model.storedPoints {
		if pointIndex == selectedPointIndex {
			continue
		}
		similarityScore := computeCosineSimilarity(selectedPoint.Vector, candidatePoint.Vector)
		neighborList = append(neighborList, indexedNeighbor{pointIndex: pointIndex, similarity: similarityScore})
	}

	// Sort by similarity in descending order
	sort.Slice(neighborList, func(firstIndex, secondIndex int) bool {
		return neighborList[firstIndex].similarity > neighborList[secondIndex].similarity
	})

	// Return only the top k neighbor indices
	if len(neighborList) > maxNeighbors {
		neighborList = neighborList[:maxNeighbors]
	}

	var neighborIndices []int
	for _, neighborEntry := range neighborList {
		neighborIndices = append(neighborIndices, neighborEntry.pointIndex)
	}
	return neighborIndices
}

// computeCosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical direction.
func computeCosineSimilarity(vectorA, vectorB []float32) float64 {
	if len(vectorA) != len(vectorB) || len(vectorA) == 0 {
		return 0
	}

	var dotProduct float64
	var normSquaredA float64
	var normSquaredB float64

	for dimensionIndex := range vectorA {
		componentA := float64(vectorA[dimensionIndex])
		componentB := float64(vectorB[dimensionIndex])
		dotProduct += componentA * componentB
		normSquaredA += componentA * componentA
		normSquaredB += componentB * componentB
	}

	if normSquaredA == 0 || normSquaredB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normSquaredA) * math.Sqrt(normSquaredB))
}

// computeVectorStatistics returns the minimum, maximum, and mean values of a vector.
func computeVectorStatistics(vector []float32) (minimumValue, maximumValue, meanValue float64) {
	if len(vector) == 0 {
		return
	}

	minimumValue = float64(vector[0])
	maximumValue = float64(vector[0])
	var sumOfValues float64

	for _, component := range vector {
		componentAsFloat := float64(component)
		if componentAsFloat < minimumValue {
			minimumValue = componentAsFloat
		}
		if componentAsFloat > maximumValue {
			maximumValue = componentAsFloat
		}
		sumOfValues += componentAsFloat
	}

	meanValue = sumOfValues / float64(len(vector))
	return
}

// computeVectorNorm calculates the L2 (Euclidean) norm of a vector.
func computeVectorNorm(vector []float32) float64 {
	var sumOfSquares float64
	for _, component := range vector {
		sumOfSquares += float64(component) * float64(component)
	}
	return math.Sqrt(sumOfSquares)
}

// truncateString shortens a string to maxLength, adding ellipsis if truncated.
func truncateString(text string, maxLength int) string {
	if len(text) <= maxLength {
		return text
	}
	if maxLength < 3 {
		return text[:maxLength]
	}
	return text[:maxLength-3] + "..."
}

// canvasCell represents a single cell in the rendering grid with its character and styling.
type canvasCell struct {
	char       rune
	style      lipgloss.Style
	isSelected bool
	isCurrent  bool
	isLine     bool
}

// renderCanvas generates the 2D visualization of all embedding points.
func (model Model) renderCanvas(canvasWidth, canvasHeight int) string {
	// Initialize the canvas grid with empty cells
	canvasGrid := model.initializeCanvasGrid(canvasWidth, canvasHeight)

	// Define all rendering styles
	styles := model.defineCanvasStyles()

	// Handle empty state - show placeholder message
	if len(model.points) == 0 {
		model.renderEmptyCanvasMessage(canvasGrid, canvasWidth, canvasHeight)
	} else {
		model.renderPointsOnCanvas(canvasGrid, canvasWidth, canvasHeight, styles)
	}

	// Convert the grid to a renderable string
	return model.canvasGridToString(canvasGrid)
}

// canvasStyles holds all the lipgloss styles used for canvas rendering.
type canvasStyles struct {
	selectedDotStyle   lipgloss.Style
	selectedLabelStyle lipgloss.Style
	currentStyle       lipgloss.Style
	normalStyle        lipgloss.Style
	lineStyle          lipgloss.Style
	neighborDotStyle   lipgloss.Style
	neighborLabelStyle lipgloss.Style
}

// initializeCanvasGrid creates a 2D grid of empty canvas cells.
func (model Model) initializeCanvasGrid(canvasWidth, canvasHeight int) [][]canvasCell {
	canvasGrid := make([][]canvasCell, canvasHeight)
	for rowIndex := range canvasGrid {
		canvasGrid[rowIndex] = make([]canvasCell, canvasWidth)
		for columnIndex := range canvasGrid[rowIndex] {
			canvasGrid[rowIndex][columnIndex] = canvasCell{char: ' ', style: lipgloss.NewStyle()}
		}
	}
	return canvasGrid
}

// defineCanvasStyles creates all the styles used for rendering the canvas.
func (model Model) defineCanvasStyles() canvasStyles {
	return canvasStyles{
		selectedDotStyle:   lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true),   // Orange for selected dot
		selectedLabelStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("118")).Bold(true),   // Green for selected word
		currentStyle:       lipgloss.NewStyle().Foreground(lipgloss.Color("213")),              // Pink for current input
		normalStyle:        lipgloss.NewStyle().Foreground(lipgloss.Color("239")),              // Dim gray for unselected
		lineStyle:          lipgloss.NewStyle().Foreground(lipgloss.Color("117")),              // Light blue for connector dots
		neighborDotStyle:   lipgloss.NewStyle().Foreground(lipgloss.Color("213")).Bold(true),   // Pink for neighbor dots
		neighborLabelStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("228")).Bold(true),   // Yellow for neighbor words
	}
}

// renderEmptyCanvasMessage displays a placeholder when no points exist.
func (model Model) renderEmptyCanvasMessage(canvasGrid [][]canvasCell, canvasWidth, canvasHeight int) {
	centerRowIndex := canvasHeight / 2
	placeholderMessage := "No embeddings yet - start typing!"
	startColumnIndex := (canvasWidth - len(placeholderMessage)) / 2
	if startColumnIndex < 0 {
		startColumnIndex = 0
	}
	for characterOffset, character := range placeholderMessage {
		if startColumnIndex+characterOffset < canvasWidth {
			canvasGrid[centerRowIndex][startColumnIndex+characterOffset] = canvasCell{char: character, style: lipgloss.NewStyle()}
		}
	}
}

// renderPointsOnCanvas draws all points, labels, and connector lines on the canvas.
func (model Model) renderPointsOnCanvas(canvasGrid [][]canvasCell, canvasWidth, canvasHeight int, styles canvasStyles) {
	// Calculate the bounding box of all points
	minimumX, maximumX, minimumY, maximumY := model.calculatePointBounds()

	// Calculate coordinate ranges, avoiding division by zero
	rangeX := maximumX - minimumX
	rangeY := maximumY - minimumY
	if rangeX == 0 {
		rangeX = 1
	}
	if rangeY == 0 {
		rangeY = 1
	}

	// Define the plotting area with padding
	paddingSize := 2
	plotAreaWidth := canvasWidth - 2*paddingSize
	plotAreaHeight := canvasHeight - 2*paddingSize

	// Convert 2D projection coordinates to grid positions
	gridPoints := model.convertPointsToGridPositions(paddingSize, plotAreaWidth, plotAreaHeight, minimumX, rangeX, minimumY, rangeY, canvasWidth, canvasHeight)

	// Identify neighbor points for the selected point
	neighborPointIndices := model.identifyNeighborIndices(gridPoints)

	// Draw connector lines from selected point to its neighbors
	model.drawNeighborConnectorLines(canvasGrid, gridPoints, neighborPointIndices, styles.lineStyle)

	// Sort points so highlighted ones render on top (last in draw order)
	sortedGridPoints := model.sortGridPointsByRenderPriority(gridPoints, neighborPointIndices)

	// Render each point with its marker and label
	model.renderGridPointsWithLabels(canvasGrid, sortedGridPoints, neighborPointIndices, canvasWidth, styles)
}

// calculatePointBounds finds the min/max X and Y coordinates across all points.
func (model Model) calculatePointBounds() (minimumX, maximumX, minimumY, maximumY float64) {
	minimumX, maximumX = model.points[0].X, model.points[0].X
	minimumY, maximumY = model.points[0].Y, model.points[0].Y

	for _, point := range model.points {
		if point.X < minimumX {
			minimumX = point.X
		}
		if point.X > maximumX {
			maximumX = point.X
		}
		if point.Y < minimumY {
			minimumY = point.Y
		}
		if point.Y > maximumY {
			maximumY = point.Y
		}
	}
	return
}

// gridPoint represents a point positioned on the canvas grid.
type gridPoint struct {
	rowIndex    int
	columnIndex int
	pointIndex  int
	label       string
	isCurrent   bool
	isSelected  bool
}

// convertPointsToGridPositions maps 2D projection coordinates to grid cell positions.
func (model Model) convertPointsToGridPositions(paddingSize, plotAreaWidth, plotAreaHeight int, minimumX, rangeX, minimumY, rangeY float64, canvasWidth, canvasHeight int) []gridPoint {
	var gridPoints []gridPoint

	for pointIndex, point := range model.points {
		// Map normalized coordinates to grid positions
		columnIndex := paddingSize + int((point.X-minimumX)/rangeX*float64(plotAreaWidth-1))
		rowIndex := paddingSize + int((point.Y-minimumY)/rangeY*float64(plotAreaHeight-1))

		// Clamp to canvas bounds
		if columnIndex < 0 {
			columnIndex = 0
		}
		if columnIndex >= canvasWidth {
			columnIndex = canvasWidth - 1
		}
		if rowIndex < 0 {
			rowIndex = 0
		}
		if rowIndex >= canvasHeight {
			rowIndex = canvasHeight - 1
		}

		// Determine if this is the current input point (unsaved, marked with ●)
		isCurrentInputPoint := pointIndex == len(model.points)-1 && strings.HasSuffix(point.Text, " ●")
		isSelectedPoint := pointIndex == model.selectedIndex

		gridPoints = append(gridPoints, gridPoint{
			rowIndex:    rowIndex,
			columnIndex: columnIndex,
			pointIndex:  pointIndex,
			label:       point.Text,
			isCurrent:   isCurrentInputPoint,
			isSelected:  isSelectedPoint,
		})
	}

	return gridPoints
}

// identifyNeighborIndices finds which point indices are neighbors of the selected point.
func (model Model) identifyNeighborIndices(gridPoints []gridPoint) map[int]bool {
	neighborPointIndices := make(map[int]bool)

	for gridPointIndex := range gridPoints {
		if gridPoints[gridPointIndex].isSelected {
			neighborIndices := model.findNearestNeighborIndices(model.selectedIndex, 5)
			for _, neighborIndex := range neighborIndices {
				neighborPointIndices[neighborIndex] = true
			}
			break
		}
	}

	return neighborPointIndices
}

// drawNeighborConnectorLines draws lines from the selected point to its neighbors.
func (model Model) drawNeighborConnectorLines(canvasGrid [][]canvasCell, gridPoints []gridPoint, neighborPointIndices map[int]bool, lineStyle lipgloss.Style) {
	// Find the selected grid point
	var selectedGridPoint *gridPoint
	for gridPointIndex := range gridPoints {
		if gridPoints[gridPointIndex].isSelected {
			selectedGridPoint = &gridPoints[gridPointIndex]
			break
		}
	}

	// Draw lines to each neighbor point
	if selectedGridPoint != nil {
		for _, targetGridPoint := range gridPoints {
			if neighborPointIndices[targetGridPoint.pointIndex] {
				drawLineOnCanvas(canvasGrid, selectedGridPoint.columnIndex, selectedGridPoint.rowIndex, targetGridPoint.columnIndex, targetGridPoint.rowIndex, lineStyle)
			}
		}
	}
}

// sortGridPointsByRenderPriority orders points so highlighted ones draw last (on top).
// Order: unselected -> neighbors -> current input -> selected
func (model Model) sortGridPointsByRenderPriority(gridPoints []gridPoint, neighborPointIndices map[int]bool) []gridPoint {
	sortedPoints := make([]gridPoint, len(gridPoints))
	copy(sortedPoints, gridPoints)

	sort.SliceStable(sortedPoints, func(firstIndex, secondIndex int) bool {
		calculateRenderPriority := func(point gridPoint) int {
			if point.isSelected {
				return 3 // Highest priority - render last (on top)
			}
			if point.isCurrent {
				return 2
			}
			if neighborPointIndices[point.pointIndex] {
				return 1
			}
			return 0 // Lowest priority - render first (underneath)
		}
		return calculateRenderPriority(sortedPoints[firstIndex]) < calculateRenderPriority(sortedPoints[secondIndex])
	})

	return sortedPoints
}

// renderGridPointsWithLabels draws markers and labels for each point on the canvas.
func (model Model) renderGridPointsWithLabels(canvasGrid [][]canvasCell, gridPoints []gridPoint, neighborPointIndices map[int]bool, canvasWidth int, styles canvasStyles) {
	// Check if any point is selected
	hasSelection := model.selectedIndex >= 0 && model.selectedIndex < len(model.storedPoints)

	for _, point := range gridPoints {
		// When focus mode is enabled and a point is selected, hide unrelated points to show connector lines clearly
		if model.focusMode && hasSelection && !point.isSelected && !point.isCurrent && !neighborPointIndices[point.pointIndex] {
			continue
		}

		// Determine the marker symbol and styles based on point state
		var markerSymbol string
		var markerStyle lipgloss.Style
		var labelStyle lipgloss.Style

		if point.isSelected {
			markerSymbol = "[*]"
			markerStyle = styles.selectedDotStyle
			labelStyle = styles.selectedLabelStyle
		} else if point.isCurrent {
			markerSymbol = "●"
			markerStyle = styles.currentStyle
			labelStyle = styles.currentStyle
		} else if neighborPointIndices[point.pointIndex] {
			markerSymbol = "◆"
			markerStyle = styles.neighborDotStyle
			labelStyle = styles.neighborLabelStyle
		} else {
			markerSymbol = "○"
			markerStyle = styles.normalStyle
			labelStyle = styles.normalStyle
		}

		// Calculate marker starting position (selected marker is wider)
		markerRunes := []rune(markerSymbol)
		markerStartColumn := point.columnIndex
		if point.isSelected {
			markerStartColumn = point.columnIndex - 1
			if markerStartColumn < 0 {
				markerStartColumn = 0
			}
		}

		// Draw the marker character(s)
		for runeOffset, markerRune := range markerRunes {
			if markerStartColumn+runeOffset < canvasWidth {
				canvasGrid[point.rowIndex][markerStartColumn+runeOffset] = canvasCell{
					char:       markerRune,
					style:      markerStyle,
					isSelected: point.isSelected,
					isCurrent:  point.isCurrent,
				}
			}
		}

		// Truncate and render the label next to the marker
		labelText := point.label
		if len(labelText) > 12 {
			labelText = labelText[:12]
		}
		labelStartColumn := point.columnIndex + len(markerRunes) + 1
		if point.isSelected {
			labelStartColumn = point.columnIndex + 3
		}
		for characterOffset, labelCharacter := range labelText {
			if labelStartColumn+characterOffset < canvasWidth {
				canvasGrid[point.rowIndex][labelStartColumn+characterOffset] = canvasCell{char: labelCharacter, style: labelStyle}
			}
		}
	}
}

// canvasGridToString converts the 2D canvas grid into a renderable string.
func (model Model) canvasGridToString(canvasGrid [][]canvasCell) string {
	var outputBuilder strings.Builder

	for rowIndex, gridRow := range canvasGrid {
		for _, cell := range gridRow {
			outputBuilder.WriteString(cell.style.Render(string(cell.char)))
		}
		// Add newline between rows, but not after the last row
		if rowIndex < len(canvasGrid)-1 {
			outputBuilder.WriteString("\n")
		}
	}

	return outputBuilder.String()
}

// drawLineOnCanvas uses Bresenham's line algorithm to draw a line between two points.
// Bresenham's algorithm efficiently determines which cells to fill when drawing a line
// by tracking accumulated error and adjusting the position accordingly.
func drawLineOnCanvas(canvasGrid [][]canvasCell, startX, startY, endX, endY int, lineStyle lipgloss.Style) {
	// Calculate the absolute distance to travel in each dimension
	deltaX := absoluteValue(endX - startX)
	deltaY := absoluteValue(endY - startY)

	// Determine the direction of travel for each axis (1 = positive, -1 = negative)
	stepDirectionX := 1
	if startX > endX {
		stepDirectionX = -1
	}
	stepDirectionY := 1
	if startY > endY {
		stepDirectionY = -1
	}

	// Initialize error term - this tracks when we need to step in the steeper direction
	// The error term represents the difference between ideal line position and current grid position
	errorTerm := deltaX - deltaY

	// Current position starts at the line's start point
	currentX := startX
	currentY := startY

	// Main loop: continue until we reach the end point
	for {
		// Draw a dot at the current position if within bounds and cell is empty
		if currentY >= 0 && currentY < len(canvasGrid) && currentX >= 0 && currentX < len(canvasGrid[0]) {
			if canvasGrid[currentY][currentX].char == ' ' {
				canvasGrid[currentY][currentX] = canvasCell{char: '·', style: lineStyle, isLine: true}
			}
		}

		// Check if we've reached the destination
		if currentX == endX && currentY == endY {
			break
		}

		// Calculate doubled error to avoid floating point arithmetic
		// This is the key insight of Bresenham's algorithm
		doubledError := 2 * errorTerm

		// If error indicates we should step in X direction
		if doubledError > -deltaY {
			errorTerm -= deltaY
			currentX += stepDirectionX
		}

		// If error indicates we should step in Y direction
		if doubledError < deltaX {
			errorTerm += deltaX
			currentY += stepDirectionY
		}
	}
}

// absoluteValue returns the absolute value of an integer.
func absoluteValue(number int) int {
	if number < 0 {
		return -number
	}
	return number
}
