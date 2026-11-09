package handlers

import (
	"encoding/json"
	"net/http"
	"runtime"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/models"
	"github.com/bowang-lab/medrax-server/pkg/queue"
)

// HealthHandler returns system health and worker telemetry
func HealthHandler(startTime time.Time, version string, pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		queued, active, completed, failed := pool.Stats()
		uptime := time.Since(startTime).Seconds()

		res := models.HealthResponse{
			Status:         "healthy",
			Version:        version,
			UptimeSeconds:  uptime,
			ActiveJobs:     active,
			QueuedJobs:     queued,
			CompletedJobs:  completed,
			FailedJobs:     failed,
			AllocatedMB:    float64(m.Alloc) / 1024.0 / 1024.0,
			NumGoroutines:  runtime.NumGoroutine(),
			Timestamp:      time.Now(),
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(res)
	}
}

