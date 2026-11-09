use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use medrax_core::{
    dicom::DicomImage,
    metrics::compute_metrics_from_files,
    batch::process_batch,
};

#[derive(Parser, Debug)]
#[command(
    name = "medrax-dicom",
    about = "High-performance Medical Imaging and DICOM Processing Engine for MedRAX",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Process a single DICOM file to PNG with contrast adjustment
    ProcessFile {
        #[arg(short, long, help = "Path to input DICOM (.dcm) file")]
        input: PathBuf,

        #[arg(short, long, help = "Path to destination output PNG file")]
        output: PathBuf,

        #[arg(long, help = "Optional custom window center (level)")]
        window_center: Option<f64>,

        #[arg(long, help = "Optional custom window width")]
        window_width: Option<f64>,
    },

    /// Run parallel batch processing of DICOM files in a directory using Rayon
    BatchProcess {
        #[arg(short, long, help = "Input directory containing DICOM files")]
        input_dir: PathBuf,

        #[arg(short, long, help = "Output directory for converted PNG images")]
        output_dir: PathBuf,

        #[arg(short, long, help = "Number of worker threads (default: CPU core count)")]
        threads: Option<usize>,
    },

    /// Compute anatomical organ metrics from segmentation mask and original image
    ComputeMetrics {
        #[arg(short, long, help = "Path to segmentation mask image")]
        mask: PathBuf,

        #[arg(short, long, help = "Path to original radiology image")]
        image: PathBuf,

        #[arg(long, default_value = "0.2", help = "Pixel spacing in mm")]
        pixel_spacing: f64,

        #[arg(long, default_value = "1.0", help = "Model confidence score")]
        confidence: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ProcessFile {
            input,
            output,
            window_center,
            window_width,
        } => {
            let res = DicomImage::convert_to_png(input, output, window_center, window_width)?;
            let json = serde_json::to_string_pretty(&res)?;
            println!("{}", json);
        }

        Commands::BatchProcess {
            input_dir,
            output_dir,
            threads,
        } => {
            let summary = process_batch(input_dir, output_dir, threads)?;
            let json = serde_json::to_string_pretty(&summary)?;
            println!("{}", json);
        }

        Commands::ComputeMetrics {
            mask,
            image,
            pixel_spacing,
            confidence,
        } => {
            let metrics = compute_metrics_from_files(mask, image, pixel_spacing, confidence)?;
            let json = serde_json::to_string_pretty(&metrics)?;
            println!("{}", json);
        }
    }

    Ok(())
}
