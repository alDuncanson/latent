package qdrant

import (
	"context"
	"fmt"

	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	conn           *grpc.ClientConn
	pointsClient   pb.PointsClient
	collectClient  pb.CollectionsClient
	collectionName string
	vectorSize     uint64
}

type Point struct {
	ID     string
	Text   string
	Vector []float32
}

func NewClient(addr, collectionName string, vectorSize uint64) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("connect to qdrant: %w", err)
	}

	c := &Client{
		conn:           conn,
		pointsClient:   pb.NewPointsClient(conn),
		collectClient:  pb.NewCollectionsClient(conn),
		collectionName: collectionName,
		vectorSize:     vectorSize,
	}

	if err := c.ensureCollection(context.Background()); err != nil {
		conn.Close()
		return nil, err
	}

	return c, nil
}

func (c *Client) ensureCollection(ctx context.Context) error {
	_, err := c.collectClient.Get(ctx, &pb.GetCollectionInfoRequest{
		CollectionName: c.collectionName,
	})
	if err == nil {
		return nil
	}

	_, err = c.collectClient.Create(ctx, &pb.CreateCollection{
		CollectionName: c.collectionName,
		VectorsConfig: &pb.VectorsConfig{
			Config: &pb.VectorsConfig_Params{
				Params: &pb.VectorParams{
					Size:     c.vectorSize,
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

func (c *Client) Upsert(ctx context.Context, id string, text string, vector []float32) error {
	_, err := c.pointsClient.Upsert(ctx, &pb.UpsertPoints{
		CollectionName: c.collectionName,
		Points: []*pb.PointStruct{
			{
				Id: &pb.PointId{
					PointIdOptions: &pb.PointId_Uuid{Uuid: id},
				},
				Vectors: &pb.Vectors{
					VectorsOptions: &pb.Vectors_Vector{
						Vector: &pb.Vector{Data: vector},
					},
				},
				Payload: map[string]*pb.Value{
					"text": {Kind: &pb.Value_StringValue{StringValue: text}},
				},
			},
		},
	})
	return err
}

func (c *Client) GetAll(ctx context.Context) ([]Point, error) {
	resp, err := c.pointsClient.Scroll(ctx, &pb.ScrollPoints{
		CollectionName: c.collectionName,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
		WithVectors:    &pb.WithVectorsSelector{SelectorOptions: &pb.WithVectorsSelector_Enable{Enable: true}},
		Limit:          pb.PtrOf(uint32(1000)),
	})
	if err != nil {
		return nil, fmt.Errorf("scroll points: %w", err)
	}

	var points []Point
	for _, p := range resp.Result {
		var id string
		if uuid := p.Id.GetUuid(); uuid != "" {
			id = uuid
		}

		var text string
		if t, ok := p.Payload["text"]; ok {
			text = t.GetStringValue()
		}

		var vector []float32
		if v := p.Vectors.GetVector(); v != nil {
			vector = v.Data
		}

		points = append(points, Point{
			ID:     id,
			Text:   text,
			Vector: vector,
		})
	}

	return points, nil
}

func (c *Client) Delete(ctx context.Context, id string) error {
	_, err := c.pointsClient.Delete(ctx, &pb.DeletePoints{
		CollectionName: c.collectionName,
		Points: &pb.PointsSelector{
			PointsSelectorOneOf: &pb.PointsSelector_Points{
				Points: &pb.PointsIdsList{
					Ids: []*pb.PointId{
						{PointIdOptions: &pb.PointId_Uuid{Uuid: id}},
					},
				},
			},
		},
	})
	return err
}

func (c *Client) Close() error {
	return c.conn.Close()
}
