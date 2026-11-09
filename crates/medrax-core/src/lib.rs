pub mod dicom;
pub mod metrics;
pub mod batch;

pub use dicom::{DicomImage, DicomMetadata, ProcessedDicomResult};
pub use metrics::{
    compute_metrics_from_buffers, compute_metrics_from_files, compute_multi_label_metrics,
    OrganMetrics,
};
pub use batch::{process_batch, BatchSummary};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
