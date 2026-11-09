package models

import "time"

// JobStatus represents the execution state of an asynchronous medical task
type JobStatus string

const (
	StatusQueued    JobStatus = "queued"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
)

// HealthResponse represents system health and runtime telemetry
type HealthResponse struct {
	Status         string    `json:"status"`
	Version        string    `json:"version"`
	UptimeSeconds  float64   `json:"uptime_seconds"`
	ActiveJobs     int       `json:"active_jobs"`
	QueuedJobs     int       `json:"queued_jobs"`
	CompletedJobs  int64     `json:"completed_jobs"`
	FailedJobs     int64     `json:"failed_jobs"`
	AllocatedMB    float64   `json:"allocated_memory_mb"`
	NumGoroutines  int       `json:"num_goroutines"`
	Timestamp      time.Time `json:"timestamp"`
}

// DicomProcessRequest specifies parameters for DICOM conversion and windowing
type DicomProcessRequest struct {
	FilePath     string   `json:"file_path,omitempty"`
	WindowCenter *float64 `json:"window_center,omitempty"`
	WindowWidth  *float64 `json:"window_width,omitempty"`
}

// DicomProcessResponse contains the resulting viewable image path and medical tags
type DicomProcessResponse struct {
	ImagePath string                 `json:"image_path"`
	Metadata  map[string]interface{} `json:"metadata"`
	Status    string                 `json:"status"`
	Error     string                 `json:"error,omitempty"`
}

// ReasonRequest defines a query submitted to the MedRAX clinical agent
type ReasonRequest struct {
	CaseID      string   `json:"case_id,omitempty"`
	Query       string   `json:"query"`
	ImagePaths  []string `json:"image_paths,omitempty"`
	DicomPaths  []string `json:"dicom_paths,omitempty"`
	Sync        bool     `json:"sync,omitempty"` // If true, wait for completion before returning
}

// ReasonResponse represents the outcome of an agent reasoning task
type ReasonResponse struct {
	JobID       string                 `json:"job_id"`
	Status      JobStatus              `json:"status"`
	Answer      string                 `json:"answer,omitempty"`
	ToolTraces  []map[string]interface{} `json:"tool_traces,omitempty"`
	DurationSec float64                `json:"duration_seconds,omitempty"`
	Error       string                 `json:"error,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	FinishedAt  *time.Time             `json:"finished_at,omitempty"`
}

// Job represents an asynchronous reasoning or processing job
type Job struct {
	ID          string                 `json:"id"`
	Status      JobStatus              `json:"status"`
	Request     ReasonRequest          `json:"request"`
	Answer      string                 `json:"answer,omitempty"`
	ToolTraces  []map[string]interface{} `json:"tool_traces,omitempty"`
	Error       string                 `json:"error,omitempty"`
	DurationSec float64                `json:"duration_seconds,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	StartedAt   *time.Time             `json:"started_at,omitempty"`
	FinishedAt  *time.Time             `json:"finished_at,omitempty"`
}

