package handlers

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/bowang-lab/medrax-server/pkg/models"
)

// DicomEngineConfig defines paths to native acceleration binaries
type DicomEngineConfig struct {
	RustBinPath string
	TempDir     string
}

// FindRustDicomBinary locates the medrax-dicom binary if compiled
func FindRustDicomBinary() string {
	if env := os.Getenv("MEDRAX_DICOM_BIN"); env != "" {
		if _, err := os.Stat(env); err == nil {
			return env
		}
	}

	// Check relative to working directory or git repo root
	candidates := []string{
		"crates/medrax-core/target/release/medrax-dicom.exe",
		"crates/medrax-core/target/release/medrax-dicom",
		"../../crates/medrax-core/target/release/medrax-dicom.exe",
		"../../crates/medrax-core/target/release/medrax-dicom",
		"../../../crates/medrax-core/target/release/medrax-dicom.exe",
		"../../../crates/medrax-core/target/release/medrax-dicom",
		"../../../../crates/medrax-core/target/release/medrax-dicom.exe",
		"../../../../crates/medrax-core/target/release/medrax-dicom",
		"crates/medrax-core/target/debug/medrax-dicom.exe",
		"crates/medrax-core/target/debug/medrax-dicom",
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			abs, err := filepath.Abs(c)
			if err == nil {
				return abs
			}
			return c
		}
	}

	if path, err := exec.LookPath("medrax-dicom"); err == nil {
		return path
	}
	return ""
}

// DicomProcessHandler handles DICOM conversion and contrast windowing
func DicomProcessHandler(cfg DicomEngineConfig) http.HandlerFunc {
	tempDir := cfg.TempDir
	if tempDir == "" {
		tempDir = "temp"
	}
	_ = os.MkdirAll(tempDir, 0755)

	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
				Status: "failed",
				Error:  "method not allowed; use POST",
			})
			return
		}

		var dicomPath string
		var windowCenter *float64
		var windowWidth *float64

		contentType := r.Header.Get("Content-Type")
		if strings.HasPrefix(contentType, "multipart/form-data") {
			// Limit upload to 100MB
			if err := r.ParseMultipartForm(100 << 20); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
					Status: "failed",
					Error:  fmt.Sprintf("failed to parse multipart upload: %v", err),
				})
				return
			}

			file, header, err := r.FormFile("file")
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
					Status: "failed",
					Error:  "missing 'file' in multipart form",
				})
				return
			}
			defer file.Close()

			// Compute SHA256 for deduplication
			hasher := sha256.New()
			tmpUploadPath := filepath.Join(tempDir, fmt.Sprintf("upload_%s", header.Filename))
			outTemp, err := os.Create(tmpUploadPath)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
					Status: "failed",
					Error:  fmt.Sprintf("failed to create temp file: %v", err),
				})
				return
			}

			multi := io.MultiWriter(outTemp, hasher)
			if _, err := io.Copy(multi, file); err != nil {
				outTemp.Close()
				w.WriteHeader(http.StatusInternalServerError)
				_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
					Status: "failed",
					Error:  fmt.Sprintf("failed to write uploaded file: %v", err),
				})
				return
			}
			outTemp.Close()

			hashStr := hex.EncodeToString(hasher.Sum(nil))[:12]
			finalUploadPath := filepath.Join(tempDir, fmt.Sprintf("upload_%s.dcm", hashStr))
			_ = os.Rename(tmpUploadPath, finalUploadPath)
			dicomPath = finalUploadPath

		} else {
			// JSON body
			var req models.DicomProcessRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
					Status: "failed",
					Error:  fmt.Sprintf("invalid JSON payload: %v", err),
				})
				return
			}
			dicomPath = req.FilePath
			windowCenter = req.WindowCenter
			windowWidth = req.WindowWidth
		}

		if dicomPath == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
				Status: "failed",
				Error:  "file_path or uploaded file is required",
			})
			return
		}

		if _, err := os.Stat(dicomPath); err != nil {
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
				Status: "failed",
				Error:  fmt.Sprintf("DICOM file not found: %s", dicomPath),
			})
			return
		}

		// Process via Rust engine if available
		rustBin := cfg.RustBinPath
		if rustBin == "" {
			rustBin = FindRustDicomBinary()
		}

		outPngName := fmt.Sprintf("processed_%d.png", os.Getpid())
		outPngPath := filepath.Join(tempDir, outPngName)

		if rustBin != "" {
			args := []string{
				"process-file",
				"--input", dicomPath,
				"--output", outPngPath,
			}
			if windowCenter != nil {
				args = append(args, "--window-center", fmt.Sprintf("%f", *windowCenter))
			}
			if windowWidth != nil {
				args = append(args, "--window-width", fmt.Sprintf("%f", *windowWidth))
			}

			cmd := exec.Command(rustBin, args...)
			output, err := cmd.Output()
			if err == nil {
				var rustRes struct {
					ImagePath string                 `json:"image_path"`
					Metadata  map[string]interface{} `json:"metadata"`
				}
				if err := json.Unmarshal(output, &rustRes); err == nil {
					rustRes.Metadata["engine"] = "rust-medrax-core"
					w.WriteHeader(http.StatusOK)
					_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
						ImagePath: outPngPath,
						Metadata:  rustRes.Metadata,
						Status:    "completed",
					})
					return
				}
			}
		}

		// Fallback Python processing
		pyScript := fmt.Sprintf(`
import json, sys
from medrax.tools.dicom import DicomProcessorTool
tool = DicomProcessorTool(temp_dir=%q)
res, meta = tool._run(%q)
print(json.dumps({'image_path': res.get('image_path', ''), 'metadata': meta}))
`, tempDir, dicomPath)

		cmd := exec.Command("python", "-c", pyScript)
		out, err := cmd.Output()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
				Status: "failed",
				Error:  fmt.Sprintf("processing error: %v", err),
			})
			return
		}

		var pyRes struct {
			ImagePath string                 `json:"image_path"`
			Metadata  map[string]interface{} `json:"metadata"`
		}
		if err := json.Unmarshal(out, &pyRes); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
				Status: "failed",
				Error:  fmt.Sprintf("failed to parse result: %v", err),
			})
			return
		}

		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(models.DicomProcessResponse{
			ImagePath: pyRes.ImagePath,
			Metadata:  pyRes.Metadata,
			Status:    "completed",
		})
	}
}

