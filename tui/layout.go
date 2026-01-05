package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/reflow/truncate"
)

const (
	overlayPanelWidth  = 44
	overlayPanelHeight = 20
	inputOverlayWidth  = 60
	inputOverlayHeight = 3
	minCanvasWidth     = 40
	minCanvasHeight    = 10
	tabBarHeight       = 1
	statusBarHeight    = 1
	borderSize         = 2
)

type viewTab int

const (
	tabProjection viewTab = iota
	tabList
	tabStats
)

type inputMode int

const (
	modeNormal inputMode = iota
	modeInput
)

type layoutDimensions struct {
	totalWidth    int
	totalHeight   int
	canvasWidth   int
	canvasHeight  int
}

func (m Model) calculateLayout() layoutDimensions {
	marginX := 2
	marginY := 2

	totalWidth := m.width - marginX
	totalHeight := m.height - marginY

	canvasHeight := totalHeight - tabBarHeight - statusBarHeight
	if canvasHeight < minCanvasHeight {
		canvasHeight = minCanvasHeight
	}

	canvasWidth := totalWidth - borderSize
	if canvasWidth < minCanvasWidth {
		canvasWidth = minCanvasWidth
	}

	return layoutDimensions{
		totalWidth:   totalWidth,
		totalHeight:  totalHeight,
		canvasWidth:  canvasWidth,
		canvasHeight: canvasHeight,
	}
}

type styles struct {
	title         lipgloss.Style
	canvas        lipgloss.Style
	overlay       lipgloss.Style
	input         lipgloss.Style
	tabActive     lipgloss.Style
	tabInactive   lipgloss.Style
	tabBar        lipgloss.Style
	statusBar     lipgloss.Style
	errorText     lipgloss.Style
}

func newStyles() styles {
	accentColor := lipgloss.Color("#FF87D7")
	borderColor := lipgloss.Color("#5F5FAF")
	canvasBorderColor := lipgloss.Color("#FF8700")
	dimColor := lipgloss.Color("#6C6C6C")
	bgColor := lipgloss.Color("#303030")

	return styles{
		title: lipgloss.NewStyle().
			Bold(true).
			Foreground(accentColor),

		canvas: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(canvasBorderColor),

		overlay: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderColor).
			Background(bgColor).
			Padding(0, 1),

		input: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(accentColor).
			Background(bgColor).
			Padding(0, 1),

		tabActive: lipgloss.NewStyle().
			Bold(true).
			Foreground(accentColor).
			Padding(0, 1),

		tabInactive: lipgloss.NewStyle().
			Foreground(dimColor).
			Padding(0, 1),

		tabBar: lipgloss.NewStyle().
			Foreground(dimColor),

		statusBar: lipgloss.NewStyle().
			Foreground(dimColor),

		errorText: lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF0000")),
	}
}

func (m Model) renderTabBar(s styles, width int) string {
	tabs := []struct {
		name string
		tab  viewTab
	}{
		{"projection", tabProjection},
		{"list", tabList},
		{"stats", tabStats},
	}

	var parts []string
	for _, t := range tabs {
		style := s.tabInactive
		if t.tab == m.activeTab {
			style = s.tabActive
		}
		parts = append(parts, style.Render(t.name))
	}

	tabRow := strings.Join(parts, s.tabBar.Render(" │ "))
	title := s.title.Render("latent")
	
	tabWidth := lipgloss.Width(tabRow)
	titleWidth := lipgloss.Width(title)
	gap := width - tabWidth - titleWidth
	if gap < 1 {
		gap = 1
	}

	return tabRow + strings.Repeat(" ", gap) + title
}

func (m Model) renderContentArea(s styles, layout layoutDimensions) string {
	canvasInnerWidth := layout.canvasWidth - borderSize
	canvasInnerHeight := layout.canvasHeight - borderSize

	var content string
	switch m.activeTab {
	case tabProjection:
		content = m.renderProjectionTab(s, layout, canvasInnerWidth, canvasInnerHeight)
	case tabList:
		content = m.renderListTab(s, canvasInnerWidth, canvasInnerHeight)
	case tabStats:
		content = m.renderStatsTab(s, canvasInnerWidth, canvasInnerHeight)
	default:
		content = m.renderProjectionTab(s, layout, canvasInnerWidth, canvasInnerHeight)
	}

	return content
}

func (m Model) renderProjectionTab(s styles, layout layoutDimensions, width, height int) string {
	canvasContent := m.renderCanvas(width, height)
	canvasBox := s.canvas.
		Width(width).
		Height(height).
		Render(canvasContent)

	showPanel := m.showMetadata && m.selectedIndex >= 0 && m.selectedIndex < len(m.storedPoints)
	if showPanel {
		canvasBox = m.overlayMetadataPanel(canvasBox, s, layout)
	}

	if m.inputMode == modeInput {
		canvasBox = m.overlayInputBox(canvasBox, s, layout)
	}

	return canvasBox
}

func (m Model) renderListTab(s styles, width, height int) string {
	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#FF87D7"))
	dimStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#6C6C6C"))
	textStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#EEEEEE"))
	selectedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#FF8700")).Bold(true)

	var lines []string
	lines = append(lines, headerStyle.Render("stored embeddings"))
	lines = append(lines, "")

	if len(m.storedPoints) == 0 {
		lines = append(lines, dimStyle.Render("no embeddings stored yet"))
		lines = append(lines, dimStyle.Render("press I to add one"))
	} else {
		maxVisible := height - 4
		startIdx := 0
		if m.selectedIndex > maxVisible-1 {
			startIdx = m.selectedIndex - maxVisible + 1
		}

		for i := startIdx; i < len(m.storedPoints) && i-startIdx < maxVisible; i++ {
			point := m.storedPoints[i]
			text := point.Text
			if len(text) > width-6 {
				text = text[:width-9] + "..."
			}

			prefix := "  "
			lineStyle := textStyle
			if i == m.selectedIndex {
				prefix = "> "
				lineStyle = selectedStyle
			}

			lines = append(lines, lineStyle.Render(prefix+text))
		}

		if len(m.storedPoints) > maxVisible {
			lines = append(lines, "")
			lines = append(lines, dimStyle.Render("use arrow keys to scroll"))
		}
	}

	content := strings.Join(lines, "\n")
	return s.canvas.
		Width(width).
		Height(height).
		Render(content)
}

func (m Model) renderStatsTab(s styles, width, height int) string {
	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#FF87D7"))
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#6C6C6C"))
	valueStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#EEEEEE"))

	var lines []string
	lines = append(lines, headerStyle.Render("collection stats"))
	lines = append(lines, "")

	lines = append(lines, labelStyle.Render("total embeddings: ")+valueStyle.Render(fmt.Sprintf("%d", len(m.storedPoints))))

	if len(m.storedPoints) > 0 {
		var totalLength int
		var minLen, maxLen int = len(m.storedPoints[0].Text), len(m.storedPoints[0].Text)
		for _, p := range m.storedPoints {
			textLen := len(p.Text)
			totalLength += textLen
			if textLen < minLen {
				minLen = textLen
			}
			if textLen > maxLen {
				maxLen = textLen
			}
		}
		avgLen := totalLength / len(m.storedPoints)

		lines = append(lines, "")
		lines = append(lines, headerStyle.Render("text lengths"))
		lines = append(lines, labelStyle.Render("min: ")+valueStyle.Render(fmt.Sprintf("%d chars", minLen)))
		lines = append(lines, labelStyle.Render("max: ")+valueStyle.Render(fmt.Sprintf("%d chars", maxLen)))
		lines = append(lines, labelStyle.Render("avg: ")+valueStyle.Render(fmt.Sprintf("%d chars", avgLen)))

		if len(m.storedPoints[0].Vector) > 0 {
			lines = append(lines, "")
			lines = append(lines, headerStyle.Render("vector info"))
			lines = append(lines, labelStyle.Render("dimensions: ")+valueStyle.Render(fmt.Sprintf("%d", len(m.storedPoints[0].Vector))))
		}

		if m.showClusters && len(m.clusterLabels) > 0 {
			clusterCounts := make(map[int]int)
			noiseCount := 0
			for _, label := range m.clusterLabels {
				if label < 0 {
					noiseCount++
				} else {
					clusterCounts[label]++
				}
			}
			numClusters := len(clusterCounts)

			lines = append(lines, "")
			lines = append(lines, headerStyle.Render("clustering"))
			lines = append(lines, labelStyle.Render("clusters found: ")+valueStyle.Render(fmt.Sprintf("%d", numClusters)))
			lines = append(lines, labelStyle.Render("noise points: ")+valueStyle.Render(fmt.Sprintf("%d", noiseCount)))
		}

		projMethod := "PCA"
		if m.useUMAP {
			projMethod = "UMAP"
		}
		lines = append(lines, "")
		lines = append(lines, headerStyle.Render("projection"))
		lines = append(lines, labelStyle.Render("method: ")+valueStyle.Render(projMethod))
	}

	content := strings.Join(lines, "\n")
	return s.canvas.
		Width(width).
		Height(height).
		Render(content)
}

func (m Model) overlayMetadataPanel(base string, s styles, layout layoutDimensions) string {
	panelInnerWidth := overlayPanelWidth - 4
	panelInnerHeight := overlayPanelHeight

	if panelInnerHeight > layout.canvasHeight-4 {
		panelInnerHeight = layout.canvasHeight - 4
	}

	metadataContent := m.renderMetadata(panelInnerWidth, panelInnerHeight)
	panel := s.overlay.
		Width(panelInnerWidth).
		Height(panelInnerHeight).
		Render(metadataContent)

	return overlayAt(base, panel, layout.canvasWidth-overlayPanelWidth-1, 1)
}

func (m Model) overlayInputBox(base string, s styles, layout layoutDimensions) string {
	inputWidth := inputOverlayWidth - 4
	inputText := m.input
	if m.embedding {
		inputText += " ..."
	}
	if inputText == "" {
		inputText = "Type to embed, Enter to save, Esc to cancel"
	}

	inputBox := s.input.
		Width(inputWidth).
		Render(inputText)

	x := (layout.canvasWidth - inputOverlayWidth) / 2
	y := layout.canvasHeight / 2

	return overlayAt(base, inputBox, x, y)
}

func overlayAt(base, overlay string, x, y int) string {
	bgLines, bgWidth := getLines(base)
	fgLines, fgWidth := getLines(overlay)
	bgHeight := len(bgLines)
	fgHeight := len(fgLines)

	if fgWidth >= bgWidth && fgHeight >= bgHeight {
		return overlay
	}

	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x > bgWidth-fgWidth {
		x = bgWidth - fgWidth
	}
	if y > bgHeight-fgHeight {
		y = bgHeight - fgHeight
	}

	var b strings.Builder
	for i, bgLine := range bgLines {
		if i > 0 {
			b.WriteByte('\n')
		}
		if i < y || i >= y+fgHeight {
			b.WriteString(bgLine)
			continue
		}

		pos := 0
		if x > 0 {
			left := truncate.String(bgLine, uint(x))
			pos = ansi.StringWidth(left)
			b.WriteString(left)
			if pos < x {
				b.WriteString(strings.Repeat(" ", x-pos))
				pos = x
			}
		}

		fgLine := fgLines[i-y]
		b.WriteString(fgLine)
		pos += ansi.StringWidth(fgLine)

		right := ansi.TruncateLeft(bgLine, pos, "")
		lineWidth := ansi.StringWidth(bgLine)
		rightWidth := ansi.StringWidth(right)
		if rightWidth <= lineWidth-pos {
			b.WriteString(strings.Repeat(" ", lineWidth-rightWidth-pos))
		}
		b.WriteString(right)
	}

	return b.String()
}

func getLines(s string) ([]string, int) {
	lines := strings.Split(s, "\n")
	widest := 0
	for _, l := range lines {
		w := ansi.StringWidth(l)
		if widest < w {
			widest = w
		}
	}
	return lines, widest
}

func (m Model) renderStatusBar(s styles, width int) string {
	var help string

	if m.inputMode == modeInput {
		help = "Enter: save │ Esc: cancel"
	} else {
		projectionMethod := "PCA"
		if m.useUMAP {
			projectionMethod = "UMAP"
		}
		clusterStatus := "off"
		if m.showClusters {
			clusterStatus = "on"
		}

		help = "↑↓: select │ /: info │ I: input │ L: labels │ F: focus │ P: " + projectionMethod + " │ C: clusters " + clusterStatus + " │ D: delete │ 1-3: tabs │ Esc: quit"
	}

	version := m.version
	padding := width - lipgloss.Width(help) - lipgloss.Width(version)
	if padding < 1 {
		padding = 1
	}

	return s.statusBar.Render(help + strings.Repeat(" ", padding) + version)
}

func (m Model) renderError(s styles) string {
	if m.err == nil {
		return ""
	}
	return s.errorText.Render("Error: " + m.err.Error())
}
