// Package qdrant provides a gRPC client for interacting with a Qdrant vector database.
// It handles collection management and CRUD operations for vector embeddings with
// associated text payloads, used for storing and retrieving text embeddings.
package qdrant

import (
	"context"
	"fmt"

	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client wraps gRPC connections to a Qdrant vector database instance.
// It provides methods for upserting, retrieving, and deleting vector points.
type Client struct {
	connection        *grpc.ClientConn
	pointsClient      pb.PointsClient
	collectionsClient pb.CollectionsClient
	collectionName    string
	vectorSize        uint64
}

// Point represents a single vector embedding with its associated metadata.
// Each point has a unique ID, the original text that was embedded, and the embedding vector.
type Point struct {
	ID     string
	Text   string
	Vector []float32
}

// NewClient creates a new Qdrant client connected to the specified address.
// It initializes the gRPC connection and ensures the target collection exists,
// creating it with cosine distance if necessary.
func NewClient(address, collectionName string, vectorSize uint64) (*Client, error) {
	connection, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("connect to qdrant: %w", err)
	}

	client := &Client{
		connection:        connection,
		pointsClient:      pb.NewPointsClient(connection),
		collectionsClient: pb.NewCollectionsClient(connection),
		collectionName:    collectionName,
		vectorSize:        vectorSize,
	}

	if err := client.ensureCollectionExists(context.Background()); err != nil {
		connection.Close()
		return nil, err
	}

	return client, nil
}

// ensureCollectionExists checks if the target collection exists in Qdrant.
// If it doesn't exist, it creates a new collection configured for cosine similarity.
func (client *Client) ensureCollectionExists(ctx context.Context) error {
	_, err := client.collectionsClient.Get(ctx, &pb.GetCollectionInfoRequest{
		CollectionName: client.collectionName,
	})
	if err == nil {
		return nil
	}

	_, err = client.collectionsClient.Create(ctx, &pb.CreateCollection{
		CollectionName: client.collectionName,
		VectorsConfig: &pb.VectorsConfig{
			Config: &pb.VectorsConfig_Params{
				Params: &pb.VectorParams{
					Size:     client.vectorSize,
					Distance: pb.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("create collection: %w", err)
	}

	return nil
}

// Upsert inserts or updates a vector point in the collection.
// The point is identified by a UUID, stores the original text as payload,
// and contains the embedding vector for similarity searches.
func (client *Client) Upsert(ctx context.Context, pointID string, text string, vector []float32) error {
	pointToUpsert := &pb.PointStruct{
		Id: &pb.PointId{
			PointIdOptions: &pb.PointId_Uuid{Uuid: pointID},
		},
		Vectors: &pb.Vectors{
			VectorsOptions: &pb.Vectors_Vector{
				Vector: &pb.Vector{Data: vector},
			},
		},
		Payload: map[string]*pb.Value{
			"text": {Kind: &pb.Value_StringValue{StringValue: text}},
		},
	}

	_, err := client.pointsClient.Upsert(ctx, &pb.UpsertPoints{
		CollectionName: client.collectionName,
		Points:         []*pb.PointStruct{pointToUpsert},
	})
	return err
}

// GetAll retrieves all vector points from the collection.
// It scrolls through the collection and returns up to 1000 points,
// each containing the ID, original text, and embedding vector.
func (client *Client) GetAll(ctx context.Context) ([]Point, error) {
	scrollResponse, err := client.pointsClient.Scroll(ctx, &pb.ScrollPoints{
		CollectionName: client.collectionName,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
		WithVectors:    &pb.WithVectorsSelector{SelectorOptions: &pb.WithVectorsSelector_Enable{Enable: true}},
		Limit:          pb.PtrOf(uint32(1000)),
	})
	if err != nil {
		return nil, fmt.Errorf("scroll points: %w", err)
	}

	var points []Point
	for _, retrievedPoint := range scrollResponse.Result {
		var pointID string
		if uuid := retrievedPoint.Id.GetUuid(); uuid != "" {
			pointID = uuid
		}

		var textContent string
		if textPayload, exists := retrievedPoint.Payload["text"]; exists {
			textContent = textPayload.GetStringValue()
		}

		var embeddingVector []float32
		if vectorData := retrievedPoint.Vectors.GetVector(); vectorData != nil {
			embeddingVector = vectorData.Data
		}

		points = append(points, Point{
			ID:     pointID,
			Text:   textContent,
			Vector: embeddingVector,
		})
	}

	return points, nil
}

// Delete removes a vector point from the collection by its UUID.
func (client *Client) Delete(ctx context.Context, pointID string) error {
	pointSelector := &pb.PointsSelector{
		PointsSelectorOneOf: &pb.PointsSelector_Points{
			Points: &pb.PointsIdsList{
				Ids: []*pb.PointId{
					{PointIdOptions: &pb.PointId_Uuid{Uuid: pointID}},
				},
			},
		},
	}

	_, err := client.pointsClient.Delete(ctx, &pb.DeletePoints{
		CollectionName: client.collectionName,
		Points:         pointSelector,
	})
	return err
}

// Close terminates the gRPC connection to the Qdrant server.
func (client *Client) Close() error {
	return client.connection.Close()
}
