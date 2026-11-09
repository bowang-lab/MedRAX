package handlers

import (
	"fmt"
	"net/http"
	"runtime"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/queue"
)

// MetricsHandler outputs Prometheus-formatted metrics
func MetricsHandler(startTime time.Time, pool *queue.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		queued, active, completed, failed := pool.Stats()
		uptime := time.Since(startTime).Seconds()

		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

		fmt.Fprintf(w, "# HELP medrax_uptime_seconds Total service uptime in seconds\n")
		fmt.Fprintf(w, "# TYPE medrax_uptime_seconds gauge\n")
		fmt.Fprintf(w, "medrax_uptime_seconds %.2f\n\n", uptime)

		fmt.Fprintf(w, "# HELP medrax_goroutines Number of active goroutines\n")
		fmt.Fprintf(w, "# TYPE medrax_goroutines gauge\n")
		fmt.Fprintf(w, "medrax_goroutines %d\n\n", runtime.NumGoroutine())

		fmt.Fprintf(w, "# HELP medrax_memory_allocated_bytes Memory allocated by Go runtime\n")
		fmt.Fprintf(w, "# TYPE medrax_memory_allocated_bytes gauge\n")
		fmt.Fprintf(w, "medrax_memory_allocated_bytes %d\n\n", m.Alloc)

		fmt.Fprintf(w, "# HELP medrax_jobs_active Active jobs being processed\n")
		fmt.Fprintf(w, "# TYPE medrax_jobs_active gauge\n")
		fmt.Fprintf(w, "medrax_jobs_active %d\n\n", active)

		fmt.Fprintf(w, "# HELP medrax_jobs_queued Jobs waiting in queue\n")
		fmt.Fprintf(w, "# TYPE medrax_jobs_queued gauge\n")
		fmt.Fprintf(w, "medrax_jobs_queued %d\n\n", queued)

		fmt.Fprintf(w, "# HELP medrax_jobs_completed_total Total completed jobs\n")
		fmt.Fprintf(w, "# TYPE medrax_jobs_completed_total counter\n")
		fmt.Fprintf(w, "medrax_jobs_completed_total %d\n\n", completed)

		fmt.Fprintf(w, "# HELP medrax_jobs_failed_total Total failed jobs\n")
		fmt.Fprintf(w, "# TYPE medrax_jobs_failed_total counter\n")
		fmt.Fprintf(w, "medrax_jobs_failed_total %d\n", failed)
	}
}

