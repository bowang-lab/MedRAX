package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/models"
	"github.com/bowang-lab/medrax-server/pkg/queue"
)

// ReasonHandler accepts new clinical reasoning requests
func ReasonHandler(pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodPost {
			http.Error(w, `{"error":"method not allowed; use POST"}`, http.StatusMethodNotAllowed)
			return
		}

		var req models.ReasonRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid JSON payload"})
			return
		}

		if strings.TrimSpace(req.Query) == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "query cannot be empty"})
			return
		}

		job, err := pool.Enqueue(req)
		if err != nil {
			if err == queue.ErrQueueFull {
				w.WriteHeader(http.StatusTooManyRequests)
				_ = json.NewEncoder(w).Encode(map[string]string{
					"error": "server is busy; queue capacity reached",
				})
				return
			}
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		// If synchronous processing requested, wait with timeout
		if req.Sync {
			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()

			completedJob, err := pool.WaitForJob(ctx, job.ID)
			if err != nil {
				w.WriteHeader(http.StatusGatewayTimeout)
				_ = json.NewEncoder(w).Encode(models.ReasonResponse{
					JobID:  job.ID,
					Status: models.StatusFailed,
					Error:  err.Error(),
				})
				return
			}

			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(models.ReasonResponse{
				JobID:       completedJob.ID,
				Status:      completedJob.Status,
				Answer:      completedJob.Answer,
				ToolTraces:  completedJob.ToolTraces,
				DurationSec: completedJob.DurationSec,
				CreatedAt:   completedJob.CreatedAt,
				FinishedAt:  completedJob.FinishedAt,
			})
			return
		}

		// Asynchronous: return 202 Accepted
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(models.ReasonResponse{
			JobID:     job.ID,
			Status:    models.StatusQueued,
			CreatedAt: job.CreatedAt,
		})
	}
}

// JobStatusHandler retrieves the status of a specific job
func JobStatusHandler(pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"method not allowed; use GET"}`, http.StatusMethodNotAllowed)
			return
		}

		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 4 {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "job ID required in path /api/v1/jobs/{id}"})
			return
		}
		jobID := parts[3]

		job, found := pool.GetJob(jobID)
		if !found {
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "job not found"})
			return
		}

		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(models.ReasonResponse{
			JobID:       job.ID,
			Status:      job.Status,
			Answer:      job.Answer,
			ToolTraces:  job.ToolTraces,
			DurationSec: job.DurationSec,
			Error:       job.Error,
			CreatedAt:   job.CreatedAt,
			FinishedAt:  job.FinishedAt,
		})
	}
}

// JobsListHandler retrieves a list of recent jobs
func JobsListHandler(pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"method not allowed; use GET"}`, http.StatusMethodNotAllowed)
			return
		}

		limit := 50
		if qLimit := r.URL.Query().Get("limit"); qLimit != "" {
			if parsed, err := strconv.Atoi(qLimit); err == nil && parsed > 0 {
				limit = parsed
			}
		}

		jobs := pool.ListJobs(limit)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"count": len(jobs),
			"jobs":  jobs,
		})
	}
}

// ReasonStreamHandler streams real-time agent updates and tool traces using Server-Sent Events (SSE)
func ReasonStreamHandler(pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, `{"error":"streaming unsupported"}`, http.StatusInternalServerError)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, `{"error":"method not allowed; use POST"}`, http.StatusMethodNotAllowed)
			return
		}

		var req models.ReasonRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid JSON payload"}`, http.StatusBadRequest)
			return
		}

		if strings.TrimSpace(req.Query) == "" {
			http.Error(w, `{"error":"query cannot be empty"}`, http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		job, err := pool.Enqueue(req)
		if err != nil {
			fmt.Fprintf(w, "event: error\ndata: {\"error\":%q}\n\n", err.Error())
			flusher.Flush()
			return
		}

		fmt.Fprintf(w, "event: status\ndata: {\"job_id\":%q,\"status\":\"queued\"}\n\n", job.ID)
		flusher.Flush()

		ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
		defer cancel()

		ticker := time.NewTicker(25 * time.Millisecond)
		defer ticker.Stop()

		lastStatus := models.StatusQueued
		sentTraces := 0

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(w, "event: error\ndata: {\"error\":\"request context cancelled or timed out\"}\n\n")
				flusher.Flush()
				return
			case <-ticker.C:
				current, found := pool.GetJob(job.ID)
				if !found {
					fmt.Fprintf(w, "event: error\ndata: {\"error\":\"job not found\"}\n\n")
					flusher.Flush()
					return
				}

				if current.Status != lastStatus {
					fmt.Fprintf(w, "event: status\ndata: {\"job_id\":%q,\"status\":%q}\n\n", current.ID, current.Status)
					flusher.Flush()
					lastStatus = current.Status
				}

				for sentTraces < len(current.ToolTraces) {
					traceJSON, _ := json.Marshal(current.ToolTraces[sentTraces])
					fmt.Fprintf(w, "event: trace\ndata: %s\n\n", string(traceJSON))
					flusher.Flush()
					sentTraces++
				}

				if current.Status == models.StatusCompleted || current.Status == models.StatusFailed {
					resJSON, _ := json.Marshal(models.ReasonResponse{
						JobID:       current.ID,
						Status:      current.Status,
						Answer:      current.Answer,
						ToolTraces:  current.ToolTraces,
						DurationSec: current.DurationSec,
						Error:       current.Error,
						CreatedAt:   current.CreatedAt,
						FinishedAt:  current.FinishedAt,
					})
					fmt.Fprintf(w, "event: complete\ndata: %s\n\n", string(resJSON))
					flusher.Flush()
					return
				}
			}
		}
	}
}

