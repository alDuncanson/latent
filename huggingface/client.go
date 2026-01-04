// Package huggingface provides a client for the Hugging Face Dataset Viewer API.
// It fetches dataset rows via REST and extracts text for embedding.
package huggingface

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

const baseURL = "https://datasets-server.huggingface.co"

// Client interacts with the Hugging Face Dataset Viewer API.
type Client struct {
	httpClient *http.Client
}

// NewClient creates a new Hugging Face API client.
func NewClient() *Client {
	return &Client{httpClient: &http.Client{}}
}

// SplitsResponse represents the response from the /splits endpoint.
type SplitsResponse struct {
	Splits []Split `json:"splits"`
}

// Split represents a dataset split.
type Split struct {
	Dataset string `json:"dataset"`
	Config  string `json:"config"`
	Split   string `json:"split"`
}

// RowsResponse represents the response from the /rows endpoint.
type RowsResponse struct {
	Rows []RowWrapper `json:"rows"`
}

// RowWrapper wraps an individual row from the dataset.
type RowWrapper struct {
	RowIdx int                    `json:"row_idx"`
	Row    map[string]interface{} `json:"row"`
}

// GetSplits fetches available splits for a dataset.
func (c *Client) GetSplits(dataset string) (*SplitsResponse, error) {
	reqURL := fmt.Sprintf("%s/splits?dataset=%s", baseURL, url.QueryEscape(dataset))

	resp, err := c.httpClient.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var result SplitsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// GetRows fetches rows from a dataset split.
func (c *Client) GetRows(dataset, config, split string, offset, length int) (*RowsResponse, error) {
	reqURL := fmt.Sprintf("%s/rows?dataset=%s&config=%s&split=%s&offset=%s&length=%s",
		baseURL,
		url.QueryEscape(dataset),
		url.QueryEscape(config),
		url.QueryEscape(split),
		strconv.Itoa(offset),
		strconv.Itoa(length),
	)

	resp, err := c.httpClient.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var result RowsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// FetchTexts fetches all text values from a dataset column.
// It paginates through the dataset in chunks of 100 rows (API max).
func (c *Client) FetchTexts(dataset, config, split, column string, maxRows int) ([]string, error) {
	var texts []string
	offset := 0
	pageSize := 100

	for {
		if maxRows > 0 && offset >= maxRows {
			break
		}

		remaining := pageSize
		if maxRows > 0 && offset+pageSize > maxRows {
			remaining = maxRows - offset
		}

		rows, err := c.GetRows(dataset, config, split, offset, remaining)
		if err != nil {
			return nil, err
		}

		if len(rows.Rows) == 0 {
			break
		}

		for _, wrapper := range rows.Rows {
			if val, ok := wrapper.Row[column]; ok {
				if text, ok := val.(string); ok && text != "" {
					texts = append(texts, text)
				}
			}
		}

		offset += len(rows.Rows)

		if len(rows.Rows) < remaining {
			break
		}
	}

	return texts, nil
}
