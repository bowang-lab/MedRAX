package queue

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/models"
)

func TestPool_EnqueueAndExecute(t *testing.T) {
	processor := func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error) {
		return "Answer for: " + job.Request.Query, nil, nil
	}

	p := NewPool(2, 10, processor)
	defer p.Shutdown(1 * time.Second)

	req := models.ReasonRequest{
		Query: "Check for pneumothorax",
	}

	job, err := p.Enqueue(req)
	if err != nil {
		t.Fatalf("Failed to enqueue: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	completed, err := p.WaitForJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("WaitForJob failed: %v", err)
	}

	if completed.Status != models.StatusCompleted {
		t.Errorf("Expected status completed, got %v", completed.Status)
	}
	if completed.Answer != "Answer for: Check for pneumothorax" {
		t.Errorf("Unexpected answer: %s", completed.Answer)
	}
}

func TestPool_QueueSaturation(t *testing.T) {
	// Block processor to fill queue
	blockCh := make(chan struct{})
	processor := func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error) {
		<-blockCh
		return "done", nil, nil
	}

	// 1 worker, 1 queue capacity
	p := NewPool(1, 1, processor)
	defer func() {
		close(blockCh)
		p.Shutdown(1 * time.Second)
	}()

	// First job takes the worker
	_, err := p.Enqueue(models.ReasonRequest{Query: "job 1"})
	if err != nil {
		t.Fatalf("job 1 failed: %v", err)
	}

	// Wait slightly for worker to pick it up
	time.Sleep(20 * time.Millisecond)

	// Second job fills queue buffer
	_, err = p.Enqueue(models.ReasonRequest{Query: "job 2"})
	if err != nil {
		t.Fatalf("job 2 failed: %v", err)
	}

	// Third job should exceed capacity and return ErrQueueFull
	_, err = p.Enqueue(models.ReasonRequest{Query: "job 3"})
	if !errors.Is(err, ErrQueueFull) {
		t.Errorf("Expected ErrQueueFull, got %v", err)
	}
}

func TestPool_Stats(t *testing.T) {
	processor := func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error) {
		return "ok", nil, nil
	}
	p := NewPool(2, 10, processor)
	defer p.Shutdown(1 * time.Second)

	_, err := p.Enqueue(models.ReasonRequest{Query: "stats test"})
	if err != nil {
		t.Fatalf("enqueue failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	list := p.ListJobs(10)
	if len(list) == 0 {
		t.Fatalf("expected jobs list > 0")
	}

	_, _ = p.WaitForJob(ctx, list[0].ID)
	_, _, completed, _ := p.Stats()
	if completed != 1 {
		t.Errorf("expected 1 completed job, got %d", completed)
	}
}

