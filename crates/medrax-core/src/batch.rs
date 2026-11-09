use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::dicom::DicomImage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFileResult {
    pub input_file: String,
    pub output_file: Option<String>,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total_found: usize,
    pub total_succeeded: usize,
    pub total_failed: usize,
    pub elapsed_seconds: f64,
    pub throughput_files_per_sec: f64,
    pub results: Vec<BatchFileResult>,
}

/// Find all DICOM files in an input directory
pub fn find_dicom_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.as_ref().exists() {
        return Ok(files);
    }

    for entry in fs::read_dir(dir.as_ref())? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            if ext == "dcm" || ext == "dicom" {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Run parallel batch conversion of DICOM files using Rayon
pub fn process_batch<P: AsRef<Path>, Q: AsRef<Path>>(
    input_dir: P,
    output_dir: Q,
    num_threads: Option<usize>,
) -> Result<BatchSummary> {
    let out_dir = output_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let dicom_files = find_dicom_files(input_dir)?;
    let total_found = dicom_files.len();

    if let Some(threads) = num_threads {
        if threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
        }
    }

    let start = Instant::now();

    let results: Vec<BatchFileResult> = dicom_files
        .par_iter()
        .map(|path| {
            let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let out_png = out_dir.join(format!("{}.png", file_stem));

            match DicomImage::convert_to_png(path, &out_png, None, None) {
                Ok(_) => BatchFileResult {
                    input_file: path.to_string_lossy().to_string(),
                    output_file: Some(out_png.to_string_lossy().to_string()),
                    status: "success".to_string(),
                    error: None,
                },
                Err(e) => BatchFileResult {
                    input_file: path.to_string_lossy().to_string(),
                    output_file: None,
                    status: "failed".to_string(),
                    error: Some(e.to_string()),
                },
            }
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let total_succeeded = results.iter().filter(|r| r.status == "success").count();
    let total_failed = results.len() - total_succeeded;
    let throughput = if elapsed > 0.0 {
        total_succeeded as f64 / elapsed
    } else {
        total_succeeded as f64
    };

    Ok(BatchSummary {
        total_found,
        total_succeeded,
        total_failed,
        elapsed_seconds: elapsed,
        throughput_files_per_sec: throughput,
        results,
    })
}
