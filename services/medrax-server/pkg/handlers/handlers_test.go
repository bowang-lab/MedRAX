package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bowang-lab/medrax-server/pkg/models"
	"github.com/bowang-lab/medrax-server/pkg/queue"
)

func createTestPool() *queue.Pool {
	proc := func(ctx context.Context, job *models.Job) (string, []map[string]interface{}, error) {
		return "Normal radiograph: " + job.Request.Query, nil, nil
	}
	return queue.NewPool(2, 10, proc)
}

func TestHealthHandler(t *testing.T) {
	pool := createTestPool()
	defer pool.Shutdown(500 * time.Millisecond)

	startTime := time.Now().Add(-10 * time.Second)
	h := HealthHandler(startTime, "0.2.0", pool)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	h(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d", w.Code)
	}

	var res models.HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&res); err != nil {
		t.Fatalf("JSON decode error: %v", err)
	}

	if res.Status != "healthy" {
		t.Errorf("Expected status healthy, got %s", res.Status)
	}
	if res.Version != "0.2.0" {
		t.Errorf("Expected version 0.2.0, got %s", res.Version)
	}
	if res.UptimeSeconds < 9.0 {
		t.Errorf("Expected uptime >= 9, got %f", res.UptimeSeconds)
	}
}

func TestMetricsHandler(t *testing.T) {
	pool := createTestPool()
	defer pool.Shutdown(500 * time.Millisecond)

	startTime := time.Now()
	h := MetricsHandler(startTime, pool)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()

	h(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "medrax_uptime_seconds") {
		t.Errorf("Metrics missing uptime: %s", body)
	}
	if !strings.Contains(body, "medrax_goroutines") {
		t.Errorf("Metrics missing goroutines: %s", body)
	}
}

func TestReasonHandler_AsyncAndSync(t *testing.T) {
	pool := createTestPool()
	defer pool.Shutdown(1 * time.Second)

	reasonH := ReasonHandler(pool)

	// 1. Asynchronous request
	asyncReq := models.ReasonRequest{
		Query: "Evaluate cardiac silhouette",
		Sync:  false,
	}
	body, _ := json.Marshal(asyncReq)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/reason", bytes.NewReader(body))
	w := httptest.NewRecorder()

	reasonH(w, req)

	if w.Code != http.StatusAccepted {
		t.Fatalf("Expected 202 Accepted, got %d", w.Code)
	}

	var asyncRes models.ReasonResponse
	_ = json.NewDecoder(w.Body).Decode(&asyncRes)
	if asyncRes.JobID == "" || asyncRes.Status != models.StatusQueued {
		t.Errorf("Invalid async response: %+v", asyncRes)
	}

	// 2. Synchronous request
	syncReq := models.ReasonRequest{
		Query: "Check bilateral hilar regions",
		Sync:  true,
	}
	syncBody, _ := json.Marshal(syncReq)
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/reason", bytes.NewReader(syncBody))
	w2 := httptest.NewRecorder()

	reasonH(w2, req2)

	if w2.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK for sync, got %d", w2.Code)
	}

	var syncRes models.ReasonResponse
	_ = json.NewDecoder(w2.Body).Decode(&syncRes)
	if syncRes.Status != models.StatusCompleted {
		t.Errorf("Expected completed status, got %s", syncRes.Status)
	}
	if !strings.Contains(syncRes.Answer, "bilateral hilar regions") {
		t.Errorf("Unexpected answer: %s", syncRes.Answer)
	}
}

func TestDicomProcessHandler_File(t *testing.T) {
	// Find demo DICOM
	dicomFile := filepath.Join("..", "..", "..", "..", "demo", "chest", "LIDC.dcm")
	if _, err := os.Stat(dicomFile); err != nil {
		dicomFile = filepath.Join("..", "..", "demo", "chest", "LIDC.dcm")
	}
	if _, err := os.Stat(dicomFile); err != nil {
		dicomFile = filepath.Join("demo", "chest", "LIDC.dcm")
	}
	if _, err := os.Stat(dicomFile); err != nil {
		t.Skip("demo/chest/LIDC.dcm not found, skipping handler test")
	}

	cfg := DicomEngineConfig{
		RustBinPath: FindRustDicomBinary(),
		TempDir:     t.TempDir(),
	}

	h := DicomProcessHandler(cfg)

	reqPayload := models.DicomProcessRequest{
		FilePath: dicomFile,
	}
	body, _ := json.Marshal(reqPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/dicom/process", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	var res models.DicomProcessResponse
	if err := json.NewDecoder(w.Body).Decode(&res); err != nil {
		t.Fatalf("JSON decode error: %v", err)
	}

	if res.Status != "completed" {
		t.Errorf("Expected completed status, got %s", res.Status)
	}
	if res.ImagePath == "" {
		t.Errorf("Expected valid image path, got empty")
	}
}

func TestReasonStreamHandler(t *testing.T) {
	pool := createTestPool()
	defer pool.Shutdown(1 * time.Second)

	streamH := ReasonStreamHandler(pool)

	reqPayload := models.ReasonRequest{
		Query: "Stream clinical reasoning test",
	}
	body, _ := json.Marshal(reqPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/reason/stream", bytes.NewReader(body))
	w := httptest.NewRecorder()

	streamH(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "event: status") {
		t.Errorf("Expected SSE status event, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "event: complete") {
		t.Errorf("Expected SSE complete event, got: %s", bodyStr)
	}
}

