package queue

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/models"
)

var (
	ErrQueueFull     = errors.New("job queue is full; retry with backpressure")
	ErrJobNotFound   = errors.New("job not found")
	ErrPoolClosed    = errors.New("worker pool is stopped")
	ErrJobTimeout    = errors.New("job timed out")
)

// TaskProcessor defines the function signature for executing a clinical reasoning job
type TaskProcessor func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error)

// Pool coordinates concurrent execution of asynchronous medical jobs
type Pool struct {
	mu           sync.RWMutex
	jobs         map[string]*models.Job
	jobQueue     chan *models.Job
	workers      int
	wg           sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc
	closed       bool
	processor    TaskProcessor
	completedCnt int64
	failedCnt    int64
	activeCnt    int64
}

// NewPool initializes and starts the worker pool
func NewPool(workers int, queueCap int, processor TaskProcessor) *Pool {
	if workers <= 0 {
		workers = 4
	}
	if queueCap <= 0 {
		queueCap = 100
	}

	ctx, cancel := context.WithCancel(context.Background())
	p := &Pool{
		jobs:      make(map[string]*models.Job),
		jobQueue:  make(chan *models.Job, queueCap),
		workers:   workers,
		ctx:       ctx,
		cancel:    cancel,
		processor: processor,
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.workerLoop()
	}

	return p
}

func generateID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// Enqueue registers and schedules a new medical job
func (p *Pool) Enqueue(req models.ReasonRequest) (*models.Job, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrPoolClosed
	}

	job := &models.Job{
		ID:        generateID(),
		Status:    models.StatusQueued,
		Request:   req,
		CreatedAt: time.Now(),
	}
	p.jobs[job.ID] = job
	p.mu.Unlock()

	select {
	case p.jobQueue <- job:
		return job, nil
	default:
		p.mu.Lock()
		job.Status = models.StatusFailed
		job.Error = ErrQueueFull.Error()
		p.mu.Unlock()
		atomic.AddInt64(&p.failedCnt, 1)
		return job, ErrQueueFull
	}
}

// GetJob returns a job by ID
func (p *Pool) GetJob(id string) (*models.Job, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	job, exists := p.jobs[id]
	if !exists {
		return nil, false
	}
	// Return a copy to ensure caller safety
	cpy := *job
	return &cpy, true
}

// ListJobs returns the most recent jobs up to the specified limit
func (p *Pool) ListJobs(limit int) []*models.Job {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if limit <= 0 || limit > 100 {
		limit = 50
	}

	list := make([]*models.Job, 0, len(p.jobs))
	for _, j := range p.jobs {
		cpy := *j
		list = append(list, &cpy)
	}

	if len(list) > limit {
		list = list[len(list)-limit:]
	}
	return list
}

// Stats returns operational counts
func (p *Pool) Stats() (queued int, active int, completed int64, failed int64) {
	p.mu.RLock()
	queued = len(p.jobQueue)
	p.mu.RUnlock()
	active = int(atomic.LoadInt64(&p.activeCnt))
	completed = atomic.LoadInt64(&p.completedCnt)
	failed = atomic.LoadInt64(&p.failedCnt)
	return
}

func (p *Pool) workerLoop() {
	defer p.wg.Done()

	for {
		select {
		case <-p.ctx.Done():
			return
		case job, ok := <-p.jobQueue:
			if !ok {
				return
			}
			p.processOne(job)
		}
	}
}

func (p *Pool) processOne(job *models.Job) {
	atomic.AddInt64(&p.activeCnt, 1)
	defer atomic.AddInt64(&p.activeCnt, -1)

	now := time.Now()
	p.mu.Lock()
	job.Status = models.StatusRunning
	job.StartedAt = &now
	p.mu.Unlock()

	var answer string
	var traces []map[string]interface{}
	var err error

	if p.processor != nil {
		answer, traces, err = p.processor(p.ctx, job)
	} else {
		// Default simulated processor
		time.Sleep(10 * time.Millisecond)
		answer = "Analysis complete: no active acute cardiopulmonary process."
	}

	finished := time.Now()
	duration := finished.Sub(now).Seconds()

	p.mu.Lock()
	job.FinishedAt = &finished
	job.DurationSec = duration
	if err != nil {
		job.Status = models.StatusFailed
		job.Error = err.Error()
		atomic.AddInt64(&p.failedCnt, 1)
	} else {
		job.Status = models.StatusCompleted
		job.Answer = answer
		job.ToolTraces = traces
		atomic.AddInt64(&p.completedCnt, 1)
	}
	p.mu.Unlock()
}

// WaitForJob blocks until the specified job completes or context is cancelled
func (p *Pool) WaitForJob(ctx context.Context, id string) (*models.Job, error) {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			if job, ok := p.GetJob(id); ok {
				if job.Status == models.StatusCompleted || job.Status == models.StatusFailed {
					return job, nil
				}
			} else {
				return nil, ErrJobNotFound
			}
		}
	}
}

// Shutdown gracefully stops the worker pool
func (p *Pool) Shutdown(timeout time.Duration) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	close(p.jobQueue)
	p.mu.Unlock()

	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(timeout):
		p.cancel()
	}
}

