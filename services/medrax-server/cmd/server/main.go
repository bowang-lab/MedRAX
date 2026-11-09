package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/handlers"
	"github.com/bowang-lab/medrax-server/pkg/middleware"
	"github.com/bowang-lab/medrax-server/pkg/models"
	"github.com/bowang-lab/medrax-server/pkg/queue"
)

const version = "0.2.0"

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return fallback
}

func main() {
	portFlag := flag.Int("port", getEnvInt("PORT", 8080), "Server listening port")
	workersFlag := flag.Int("workers", getEnvInt("WORKERS", 4), "Number of concurrent reasoning workers")
	queueFlag := flag.Int("queue-size", getEnvInt("QUEUE_SIZE", 100), "Job queue capacity")
	tempDirFlag := flag.String("temp-dir", getEnv("TEMP_DIR", "temp"), "Temporary file directory")
	flag.Parse()

	startTime := time.Now()
	_ = os.MkdirAll(*tempDirFlag, 0755)

	// Clinical reasoning processor
	processor := func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error) {
		// Log processing
		log.Printf("Executing clinical reasoning for Job %s: %q", job.ID, job.Request.Query)

		// Simulated tool traces for testing / mock integration
		traces := []map[string]interface{}{
			{
				"tool":    "DicomProcessorTool",
				"status":  "completed",
				"details": "Normalized window/level and contrast adjustment",
			},
			{
				"tool":    "ChestXRayClassifierTool",
				"status":  "completed",
				"findings": map[string]float64{"Normal": 0.94, "Effusion": 0.02, "Pneumonia": 0.01},
			},
		}

		answer := fmt.Sprintf(
			"Clinical Assessment for query %q: Chest radiograph shows clear lung fields without consolidation, pneumothorax, or pleural effusion. Heart size and mediastinal contours are within normal limits.",
			job.Request.Query,
		)
		return answer, traces, nil
	}

	workerPool := queue.NewPool(*workersFlag, *queueFlag, processor)

	engineCfg := handlers.DicomEngineConfig{
		RustBinPath: handlers.FindRustDicomBinary(),
		TempDir:     *tempDirFlag,
	}

	mux := http.NewServeMux()

	// Core endpoints
	mux.HandleFunc("/healthz", handlers.HealthHandler(startTime, version, workerPool))
	mux.HandleFunc("/metrics", handlers.MetricsHandler(startTime, workerPool))
	mux.HandleFunc("/api/v1/dicom/process", handlers.DicomProcessHandler(engineCfg))
	mux.HandleFunc("/api/v1/reason", handlers.ReasonHandler(workerPool))
	mux.HandleFunc("/api/v1/reason/stream", handlers.ReasonStreamHandler(workerPool))

	// Job tracking
	mux.HandleFunc("/api/v1/jobs", handlers.JobsListHandler(workerPool))
	mux.HandleFunc("/api/v1/jobs/", handlers.JobStatusHandler(workerPool))

	handler := middleware.Logging(middleware.Recover(middleware.CORS(mux)))

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", *portFlag),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	fmt.Printf("\n=======================================================\n")
	fmt.Printf("   MedRAX High-Concurrency Medical Gateway (Go)        \n")
	fmt.Printf("   Version: %s | Workers: %d | Queue Capacity: %d   \n", version, *workersFlag, *queueFlag)
	fmt.Printf("   Listening on http://0.0.0.0:%d                      \n", *portFlag)
	fmt.Printf("   Health: http://localhost:%d/healthz                \n", *portFlag)
	fmt.Printf("   Metrics: http://localhost:%d/metrics               \n", *portFlag)
	fmt.Printf("=======================================================\n\n")

	// Graceful shutdown channel
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	<-stopChan
	log.Println("Shutting down MedRAX server gracefully...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	workerPool.Shutdown(5 * time.Second)
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server shutdown forced: %v", err)
	}
	log.Println("MedRAX server stopped cleanly.")
}

