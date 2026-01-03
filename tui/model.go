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
}

type embeddingResult struct {
	vector []float32
	err    error
}

type pointsUpdated struct {
	points       []projection.Point2D
	storedPoints []qdrant.Point
}

func NewModel(ollamaClient *ollama.Client, qdrantClient *qdrant.Client) Model {
	return Model{
		ollama:        ollamaClient,
		qdrant:        qdrantClient,
		width:         80,
		height:        24,
		selectedIndex: -1,
		showMetadata:  true,
	}
}

func (m Model) Init() tea.Cmd {
	return m.loadPoints()
}

func (m *Model) loadPoints() tea.Cmd {
	client := m.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		stored, err := client.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		var vectors [][]float32
		var texts []string
		for _, p := range stored {
			vectors = append(vectors, p.Vector)
			texts = append(texts, p.Text)
		}

		points := projection.ProjectTo2D(vectors, texts)
		return pointsUpdated{points: points, storedPoints: stored}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "esc":
			return m, tea.Quit
		case "enter":
			if m.input != "" && m.currentVec != nil {
				cmd := m.saveCurrentEmbedding()
				m.input = ""
				m.cursorPos = 0
				m.currentVec = nil
				return m, cmd
			}
		case "backspace":
			if m.cursorPos > 0 {
				m.input = m.input[:m.cursorPos-1] + m.input[m.cursorPos:]
				m.cursorPos--
				return m, m.debounceEmbed()
			}
		case "left":
			if m.cursorPos > 0 {
				m.cursorPos--
			}
		case "right":
			if m.cursorPos < len(m.input) {
				m.cursorPos++
			}
		case "tab":
			if len(m.storedPoints) > 0 {
				m.selectedIndex = (m.selectedIndex + 1) % len(m.storedPoints)
			}
		case "shift+tab":
			if len(m.storedPoints) > 0 {
				m.selectedIndex--
				if m.selectedIndex < 0 {
					m.selectedIndex = len(m.storedPoints) - 1
				}
			}
		case "up":
			if len(m.storedPoints) > 0 {
				m.selectedIndex--
				if m.selectedIndex < 0 {
					m.selectedIndex = len(m.storedPoints) - 1
				}
			}
		case "down":
			if len(m.storedPoints) > 0 {
				m.selectedIndex = (m.selectedIndex + 1) % len(m.storedPoints)
			}
		case "/":
			m.showMetadata = !m.showMetadata
		case "D":
			if m.selectedIndex >= 0 && m.selectedIndex < len(m.storedPoints) {
				return m, m.deleteSelected()
			}
		default:
			if len(msg.String()) == 1 {
				m.input = m.input[:m.cursorPos] + msg.String() + m.input[m.cursorPos:]
				m.cursorPos++
				return m, m.debounceEmbed()
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case embeddingResult:
		m.embedding = false
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		m.currentVec = msg.vector
		m.err = nil
		if m.currentVec != nil {
			return m, m.updateVisualization()
		}

	case pointsUpdated:
		m.points = msg.points
		if msg.storedPoints != nil {
			m.storedPoints = msg.storedPoints
		}
		var texts []string
		for _, p := range m.points {
			texts = append(texts, p.Text)
		}
		m.savedTexts = texts
		if m.selectedIndex >= len(m.storedPoints) {
			m.selectedIndex = len(m.storedPoints) - 1
		}
	}

	return m, nil
}

func (m *Model) debounceEmbed() tea.Cmd {
	input := m.input
	client := m.ollama
	return func() tea.Msg {
		time.Sleep(150 * time.Millisecond)
		if input == "" {
			return embeddingResult{}
		}
		vec, err := client.Embed(input)
		return embeddingResult{vector: vec, err: err}
	}
}

func (m *Model) saveCurrentEmbedding() tea.Cmd {
	text := m.input
	vec := m.currentVec
	client := m.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		id := uuid.New().String()
		if err := client.Upsert(ctx, id, text, vec); err != nil {
			return embeddingResult{err: err}
		}

		stored, err := client.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		var vectors [][]float32
		var texts []string
		for _, p := range stored {
			vectors = append(vectors, p.Vector)
			texts = append(texts, p.Text)
		}

		points := projection.ProjectTo2D(vectors, texts)
		return pointsUpdated{points: points, storedPoints: stored}
	}
}

func (m *Model) deleteSelected() tea.Cmd {
	if m.selectedIndex < 0 || m.selectedIndex >= len(m.storedPoints) {
		return nil
	}
	id := m.storedPoints[m.selectedIndex].ID
	client := m.qdrant
	return func() tea.Msg {
		ctx := context.Background()
		if err := client.Delete(ctx, id); err != nil {
			return embeddingResult{err: err}
		}

		stored, err := client.GetAll(ctx)
		if err != nil {
			return embeddingResult{err: err}
		}

		var vectors [][]float32
		var texts []string
		for _, p := range stored {
			vectors = append(vectors, p.Vector)
			texts = append(texts, p.Text)
		}

		points := projection.ProjectTo2D(vectors, texts)
		return pointsUpdated{points: points, storedPoints: stored}
	}
}

func (m *Model) updateVisualization() tea.Cmd {
	client := m.qdrant
	currentVec := m.currentVec
	input := m.input
	return func() tea.Msg {
		ctx := context.Background()
		stored, err := client.GetAll(ctx)
		if err != nil {
			return pointsUpdated{}
		}

		var vectors [][]float32
		var texts []string
		for _, p := range stored {
			vectors = append(vectors, p.Vector)
			texts = append(texts, p.Text)
		}

		if currentVec != nil {
			vectors = append(vectors, currentVec)
			texts = append(texts, input+" ●")
		}

		points := projection.ProjectTo2D(vectors, texts)
		return pointsUpdated{points: points}
	}
}

func (m Model) View() string {
	margin := 1
	gap := 1
	totalWidth := m.width - margin*2

	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	helpStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	inputBorder := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63")).Padding(0, 1)
	canvasBorder := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("240"))
	metaBorder := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63")).Padding(0, 1)

	var b strings.Builder
	b.WriteString(titleStyle.Render("latent"))
	b.WriteString("\n")

	inputDisplay := m.input
	if m.embedding {
		inputDisplay += " (embedding...)"
	}

	canvasHeight := m.height - 9
	if canvasHeight < 10 {
		canvasHeight = 10
	}

	showMeta := m.showMetadata && m.selectedIndex >= 0 && m.selectedIndex < len(m.storedPoints)
	if showMeta {
		metaOuter := 26
		metaInner := metaOuter - 4
		canvasInner := totalWidth - metaOuter - gap - 2

		b.WriteString(inputBorder.Width(totalWidth - 4).Render(inputDisplay))
		b.WriteString("\n")

		canvas := m.renderCanvas(canvasInner, canvasHeight)
		metadata := m.renderMetadata(metaInner, canvasHeight-1)

		left := canvasBorder.Width(canvasInner).Render(canvas)
		right := metaBorder.Width(metaInner).Height(canvasHeight - 1).Render(metadata)

		b.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, left, strings.Repeat(" ", gap), right))
	} else {
		innerWidth := totalWidth - 4

		b.WriteString(inputBorder.Width(innerWidth).Render(inputDisplay))
		b.WriteString("\n")

		canvas := m.renderCanvas(innerWidth, canvasHeight)
		b.WriteString(canvasBorder.Width(innerWidth).Render(canvas))
	}
	b.WriteString("\n")

	if m.err != nil {
		errStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
		b.WriteString(errStyle.Render("Error: "+m.err.Error()) + "\n")
	}

	b.WriteString(helpStyle.Render("Up/Down: select | /: metadata | D: delete | Enter: save | Esc: quit"))

	return lipgloss.NewStyle().Padding(1, margin).Render(b.String())
}

func (m Model) renderMetadata(width, height int) string {
	if m.selectedIndex < 0 || m.selectedIndex >= len(m.storedPoints) {
		return ""
	}

	selected := m.storedPoints[m.selectedIndex]
	var lines []string

	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	valueStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("255"))

	lines = append(lines, headerStyle.Render("Selected"))
	lines = append(lines, valueStyle.Render(truncate(selected.Text, width)))
	lines = append(lines, "")

	if len(selected.Vector) > 0 {
		lines = append(lines, labelStyle.Render("Dim: ")+valueStyle.Render(fmt.Sprintf("%d", len(selected.Vector))))

		min, max, mean := vectorStats(selected.Vector)
		lines = append(lines, labelStyle.Render("Min/Max: ")+valueStyle.Render(fmt.Sprintf("%.3f / %.3f", min, max)))
		lines = append(lines, labelStyle.Render("Mean: ")+valueStyle.Render(fmt.Sprintf("%.4f", mean)))

		norm := vectorNorm(selected.Vector)
		lines = append(lines, labelStyle.Render("L2 norm: ")+valueStyle.Render(fmt.Sprintf("%.4f", norm)))
		lines = append(lines, "")
	}

	neighbors := m.findNearestNeighbors(m.selectedIndex, 5)
	if len(neighbors) > 0 {
		lines = append(lines, headerStyle.Render("Nearest"))
		for _, n := range neighbors {
			line := fmt.Sprintf("%.3f %s", n.similarity, truncate(n.text, width-7))
			lines = append(lines, line)
		}
	}

	for len(lines) < height {
		lines = append(lines, "")
	}
	if len(lines) > height {
		lines = lines[:height]
	}

	return strings.Join(lines, "\n")
}

type neighbor struct {
	text       string
	similarity float64
}

func (m Model) findNearestNeighbors(selectedIdx int, k int) []neighbor {
	if selectedIdx < 0 || selectedIdx >= len(m.storedPoints) {
		return nil
	}

	selected := m.storedPoints[selectedIdx]
	var neighbors []neighbor

	for i, p := range m.storedPoints {
		if i == selectedIdx {
			continue
		}
		sim := cosineSimilarity(selected.Vector, p.Vector)
		neighbors = append(neighbors, neighbor{text: p.Text, similarity: sim})
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].similarity > neighbors[j].similarity
	})

	if len(neighbors) > k {
		neighbors = neighbors[:k]
	}
	return neighbors
}

func (m Model) findNearestNeighborIndices(selectedIdx int, k int) []int {
	if selectedIdx < 0 || selectedIdx >= len(m.storedPoints) {
		return nil
	}

	selected := m.storedPoints[selectedIdx]
	type indexedNeighbor struct {
		idx        int
		similarity float64
	}
	var neighbors []indexedNeighbor

	for i, p := range m.storedPoints {
		if i == selectedIdx {
			continue
		}
		sim := cosineSimilarity(selected.Vector, p.Vector)
		neighbors = append(neighbors, indexedNeighbor{idx: i, similarity: sim})
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].similarity > neighbors[j].similarity
	})

	if len(neighbors) > k {
		neighbors = neighbors[:k]
	}

	var indices []int
	for _, n := range neighbors {
		indices = append(indices, n.idx)
	}
	return indices
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func vectorStats(v []float32) (min, max, mean float64) {
	if len(v) == 0 {
		return
	}
	min = float64(v[0])
	max = float64(v[0])
	var sum float64
	for _, val := range v {
		f := float64(val)
		if f < min {
			min = f
		}
		if f > max {
			max = f
		}
		sum += f
	}
	mean = sum / float64(len(v))
	return
}

func vectorNorm(v []float32) float64 {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}
	return math.Sqrt(sum)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen < 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

type canvasCell struct {
	char       rune
	style      lipgloss.Style
	isSelected bool
	isCurrent  bool
	isLine     bool
}

func (m Model) renderCanvas(width, height int) string {
	grid := make([][]canvasCell, height)
	for i := range grid {
		grid[i] = make([]canvasCell, width)
		for j := range grid[i] {
			grid[i][j] = canvasCell{char: ' ', style: lipgloss.NewStyle()}
		}
	}

	selectedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("46")).Bold(true)
	currentStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("213"))
	normalStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	lineStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
	neighborStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("34"))

	if len(m.points) == 0 {
		centerY := height / 2
		msg := "No embeddings yet - start typing!"
		startX := (width - len(msg)) / 2
		if startX < 0 {
			startX = 0
		}
		for i, c := range msg {
			if startX+i < width {
				grid[centerY][startX+i] = canvasCell{char: c, style: lipgloss.NewStyle()}
			}
		}
	} else {
		minX, maxX := m.points[0].X, m.points[0].X
		minY, maxY := m.points[0].Y, m.points[0].Y
		for _, p := range m.points {
			if p.X < minX {
				minX = p.X
			}
			if p.X > maxX {
				maxX = p.X
			}
			if p.Y < minY {
				minY = p.Y
			}
			if p.Y > maxY {
				maxY = p.Y
			}
		}

		rangeX := maxX - minX
		rangeY := maxY - minY
		if rangeX == 0 {
			rangeX = 1
		}
		if rangeY == 0 {
			rangeY = 1
		}

		padding := 2
		plotWidth := width - 2*padding
		plotHeight := height - 2*padding

		type gridPoint struct {
			y, x       int
			idx        int
			label      string
			isCurrent  bool
			isSelected bool
		}
		var gridPoints []gridPoint

		for i, p := range m.points {
			x := padding + int((p.X-minX)/rangeX*float64(plotWidth-1))
			y := padding + int((p.Y-minY)/rangeY*float64(plotHeight-1))

			if x < 0 {
				x = 0
			}
			if x >= width {
				x = width - 1
			}
			if y < 0 {
				y = 0
			}
			if y >= height {
				y = height - 1
			}

			isCurrent := i == len(m.points)-1 && strings.HasSuffix(p.Text, " ●")
			isSelected := i == m.selectedIndex

			gridPoints = append(gridPoints, gridPoint{y: y, x: x, idx: i, label: p.Text, isCurrent: isCurrent, isSelected: isSelected})
		}

		neighborIndices := make(map[int]bool)
		var selectedGP *gridPoint
		for i := range gridPoints {
			if gridPoints[i].isSelected {
				selectedGP = &gridPoints[i]
				neighbors := m.findNearestNeighborIndices(m.selectedIndex, 5)
				for _, ni := range neighbors {
					neighborIndices[ni] = true
				}
				break
			}
		}

		if selectedGP != nil {
			for _, gp := range gridPoints {
				if neighborIndices[gp.idx] {
					drawLine(grid, selectedGP.x, selectedGP.y, gp.x, gp.y, lineStyle)
				}
			}
		}

		for _, gp := range gridPoints {
			var marker string
			var style lipgloss.Style

			if gp.isSelected {
				marker = "[*]"
				style = selectedStyle
			} else if gp.isCurrent {
				marker = "●"
				style = currentStyle
			} else if neighborIndices[gp.idx] {
				marker = "◆"
				style = neighborStyle
			} else {
				marker = "○"
				style = normalStyle
			}

			markerRunes := []rune(marker)
			startX := gp.x
			if gp.isSelected {
				startX = gp.x - 1
				if startX < 0 {
					startX = 0
				}
			}

			for j, c := range markerRunes {
				if startX+j < width {
					grid[gp.y][startX+j] = canvasCell{char: c, style: style, isSelected: gp.isSelected, isCurrent: gp.isCurrent}
				}
			}

			label := gp.label
			if len(label) > 12 {
				label = label[:12]
			}
			labelStart := gp.x + len(markerRunes)
			if gp.isSelected {
				labelStart = gp.x + 2
			}
			labelStyle := normalStyle
			if gp.isSelected {
				labelStyle = selectedStyle
			} else if neighborIndices[gp.idx] {
				labelStyle = neighborStyle
			}
			for j, c := range label {
				if labelStart+j < width {
					grid[gp.y][labelStart+j] = canvasCell{char: c, style: labelStyle}
				}
			}
		}
	}

	var b strings.Builder
	for i, row := range grid {
		for _, c := range row {
			b.WriteString(c.style.Render(string(c.char)))
		}
		if i < len(grid)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

func drawLine(grid [][]canvasCell, x0, y0, x1, y1 int, style lipgloss.Style) {
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy

	for {
		if y0 >= 0 && y0 < len(grid) && x0 >= 0 && x0 < len(grid[0]) {
			if grid[y0][x0].char == ' ' {
				lineChar := '·'
				if dx > dy*2 {
					lineChar = '─'
				} else if dy > dx*2 {
					lineChar = '│'
				} else if (sx > 0 && sy > 0) || (sx < 0 && sy < 0) {
					lineChar = '╲'
				} else {
					lineChar = '╱'
				}
				grid[y0][x0] = canvasCell{char: lineChar, style: style, isLine: true}
			}
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
