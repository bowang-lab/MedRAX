use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{bail, Context, Result};
use image::{GrayImage, Luma};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DicomMetadata {
    pub patient_id: Option<String>,
    pub study_date: Option<String>,
    pub modality: Option<String>,
    pub photometric_interpretation: Option<String>,
    pub pixel_spacing: Option<Vec<f64>>,
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
    pub rescale_slope: Option<f64>,
    pub rescale_intercept: Option<f64>,
    pub rows: u32,
    pub columns: u32,
    pub bits_stored: u16,
    pub analysis_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDicomResult {
    pub image_path: String,
    pub metadata: DicomMetadata,
}

/// A parsed raw DICOM structure
pub struct DicomImage {
    pub rows: u32,
    pub cols: u32,
    pub bits_allocated: u16,
    pub bits_stored: u16,
    pub is_signed: bool,
    pub photometric: String,
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
    pub rescale_slope: Option<f64>,
    pub rescale_intercept: Option<f64>,
    pub pixel_spacing: Option<Vec<f64>>,
    pub patient_id: Option<String>,
    pub study_date: Option<String>,
    pub modality: Option<String>,
    pub pixels: Vec<f64>,
}

impl DicomImage {
    /// Read and parse a DICOM file from disk
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open DICOM file: {:?}", path.as_ref()))?;
        let mut reader = BufReader::new(file);

        // Standard DICOM has 128-byte preamble followed by "DICM" magic
        let mut preamble = [0u8; 128];
        reader.read_exact(&mut preamble).context("Could not read DICOM preamble")?;

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).context("Could not read DICOM magic bytes")?;
        if &magic != b"DICM" {
            // Some non-preamble legacy DICOMs start immediately at byte 0
            reader.seek(SeekFrom::Start(0))?;
        }

        let mut rows: Option<u32> = None;
        let mut cols: Option<u32> = None;
        let mut bits_allocated: u16 = 16;
        let mut bits_stored: u16 = 16;
        let mut is_signed = false;
        let mut photometric = "MONOCHROME2".to_string();
        let mut window_center: Option<f64> = None;
        let mut window_width: Option<f64> = None;
        let mut rescale_slope: Option<f64> = None;
        let mut rescale_intercept: Option<f64> = None;
        let mut pixel_spacing: Option<Vec<f64>> = None;
        let mut patient_id: Option<String> = None;
        let mut study_date: Option<String> = None;
        let mut modality: Option<String> = None;
        let mut pixel_data_offset: Option<u64> = None;

        // Parse data elements
        let mut tag_buf = [0u8; 4];
        while reader.read_exact(&mut tag_buf).is_ok() {
            let group = u16::from_le_bytes([tag_buf[0], tag_buf[1]]);
            let element = u16::from_le_bytes([tag_buf[2], tag_buf[3]]);

            // Read VR or length
            let mut vr_buf = [0u8; 2];
            if reader.read_exact(&mut vr_buf).is_err() {
                break;
            }

            let is_explicit = matches!(
                &vr_buf,
                b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN" | b"AE" | b"AS" | b"AT" | b"CS"
                    | b"DA" | b"DS" | b"DT" | b"FD" | b"FL" | b"IS" | b"LO" | b"LT" | b"PN"
                    | b"SH" | b"SL" | b"SS" | b"ST" | b"TM" | b"UI" | b"UL" | b"US"
            );

            let length = if is_explicit {
                let vr = std::str::from_utf8(&vr_buf).unwrap_or("UN");
                if matches!(vr, "OB" | "OW" | "OF" | "SQ" | "UT" | "UN") {
                    let mut reserved = [0u8; 2];
                    reader.read_exact(&mut reserved)?;
                    let mut len_buf = [0u8; 4];
                    reader.read_exact(&mut len_buf)?;
                    u32::from_le_bytes(len_buf) as usize
                } else {
                    let mut len_buf = [0u8; 2];
                    reader.read_exact(&mut len_buf)?;
                    u16::from_le_bytes(len_buf) as usize
                }
            } else {
                // Implicit VR: 4-byte length directly following tag
                let mut len_buf = [0u8; 2];
                reader.read_exact(&mut len_buf)?;
                let total_len = u32::from_le_bytes([vr_buf[0], vr_buf[1], len_buf[0], len_buf[1]]);
                total_len as usize
            };

            // Check for Pixel Data tag (7FE0, 0010)
            if group == 0x7FE0 && element == 0x0010 {
                let current_pos = reader.stream_position()?;
                pixel_data_offset = Some(current_pos);
                break;
            }

            if length == 0xFFFFFFFF {
                // Undefined length sequence: scan forward until Sequence Delimitation Item (FFFE, E0DD)
                let mut delim_buf = [0u8; 4];
                let mut found_delim = false;
                while reader.read_exact(&mut delim_buf).is_ok() {
                    let d_group = u16::from_le_bytes([delim_buf[0], delim_buf[1]]);
                    let d_element = u16::from_le_bytes([delim_buf[2], delim_buf[3]]);
                    if d_group == 0xFFFE && d_element == 0xE0DD {
                        let mut delim_len = [0u8; 4];
                        let _ = reader.read_exact(&mut delim_len);
                        found_delim = true;
                        break;
                    }
                    reader.seek(SeekFrom::Current(-3))?;
                }
                if !found_delim {
                    break;
                }
                continue;
            }

            // Read known tags
            let mut val_bytes = vec![0u8; length];
            if reader.read_exact(&mut val_bytes).is_err() {
                break;
            }

            let text_val = String::from_utf8_lossy(&val_bytes).trim().trim_matches('\0').to_string();

            match (group, element) {
                (0x0010, 0x0020) => patient_id = Some(text_val),
                (0x0008, 0x0020) => study_date = Some(text_val),
                (0x0008, 0x0060) => modality = Some(text_val),
                (0x0028, 0x0004) => photometric = text_val.to_uppercase(),
                (0x0028, 0x0010) => {
                    if val_bytes.len() >= 2 {
                        rows = Some(u16::from_le_bytes([val_bytes[0], val_bytes[1]]) as u32);
                    }
                }
                (0x0028, 0x0011) => {
                    if val_bytes.len() >= 2 {
                        cols = Some(u16::from_le_bytes([val_bytes[0], val_bytes[1]]) as u32);
                    }
                }
                (0x0028, 0x0100) => {
                    if val_bytes.len() >= 2 {
                        bits_allocated = u16::from_le_bytes([val_bytes[0], val_bytes[1]]);
                    }
                }
                (0x0028, 0x0101) => {
                    if val_bytes.len() >= 2 {
                        bits_stored = u16::from_le_bytes([val_bytes[0], val_bytes[1]]);
                    }
                }
                (0x0028, 0x0103) => {
                    if val_bytes.len() >= 2 {
                        is_signed = u16::from_le_bytes([val_bytes[0], val_bytes[1]]) != 0;
                    }
                }
                (0x0028, 0x0030) => {
                    // Pixel Spacing e.g. "0.14\0.14"
                    let parts: Vec<f64> = text_val
                        .split('\\')
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect();
                    if !parts.is_empty() {
                        pixel_spacing = Some(parts);
                    }
                }
                (0x0028, 0x1050) => {
                    let first = text_val.split('\\').next().unwrap_or("");
                    window_center = first.trim().parse::<f64>().ok();
                }
                (0x0028, 0x1051) => {
                    let first = text_val.split('\\').next().unwrap_or("");
                    window_width = first.trim().parse::<f64>().ok();
                }
                (0x0028, 0x1052) => rescale_intercept = text_val.parse::<f64>().ok(),
                (0x0028, 0x1053) => rescale_slope = text_val.parse::<f64>().ok(),
                _ => {}
            }
        }

        let num_rows = rows.unwrap_or(0);
        let num_cols = cols.unwrap_or(0);
        let total_pixels = (num_rows * num_cols) as usize;

        let offset = match pixel_data_offset {
            Some(off) => off,
            None => {
                // Search file for Pixel Data Tag (0x7FE0, 0x0010)
                reader.seek(SeekFrom::Start(0))?;
                let mut all_bytes = Vec::new();
                reader.read_to_end(&mut all_bytes)?;
                let needle = [0xE0u8, 0x7F, 0x10, 0x00];
                let pos = all_bytes
                    .windows(4)
                    .position(|w| w == needle)
                    .context("Pixel data tag not found in DICOM file")?;

                if pos + 6 <= all_bytes.len()
                    && (&all_bytes[pos + 4..pos + 6] == b"OW"
                        || &all_bytes[pos + 4..pos + 6] == b"OB")
                {
                    (pos + 12) as u64
                } else {
                    (pos + 8) as u64
                }
            }
        };
        reader.seek(SeekFrom::Start(offset))?;

        let mut pixels = Vec::with_capacity(total_pixels);
        if bits_allocated > 8 {
            let bytes_needed = total_pixels * 2;
            let mut raw_bytes = vec![0u8; bytes_needed];
            let read_len = reader.read(&mut raw_bytes)?;
            let count = read_len / 2;

            for i in 0..count {
                let b0 = raw_bytes[2 * i];
                let b1 = raw_bytes[2 * i + 1];
                let val = if is_signed {
                    i16::from_le_bytes([b0, b1]) as f64
                } else {
                    u16::from_le_bytes([b0, b1]) as f64
                };
                pixels.push(val);
            }
        } else {
            let mut raw_bytes = vec![0u8; total_pixels];
            let read_len = reader.read(&mut raw_bytes)?;
            for i in 0..read_len {
                pixels.push(raw_bytes[i] as f64);
            }
        }

        if pixels.is_empty() {
            bail!("No pixel data could be decoded from DICOM file");
        }

        Ok(Self {
            rows: num_rows,
            cols: num_cols,
            bits_allocated,
            bits_stored,
            is_signed,
            photometric,
            window_center,
            window_width,
            rescale_slope,
            rescale_intercept,
            pixel_spacing,
            patient_id,
            study_date,
            modality,
            pixels,
        })
    }

    /// Process the pixel array with rescaling, MONOCHROME1 inversion, and window/level contrast
    pub fn process_to_gray_image(
        &mut self,
        custom_center: Option<f64>,
        custom_width: Option<f64>,
    ) -> (GrayImage, DicomMetadata) {
        let slope = self.rescale_slope.unwrap_or(1.0);
        let intercept = self.rescale_intercept.unwrap_or(0.0);

        // Apply RescaleSlope and RescaleIntercept
        if slope != 1.0 || intercept != 0.0 {
            for p in self.pixels.iter_mut() {
                *p = *p * slope + intercept;
            }
        }

        // Apply MONOCHROME1 inversion (radiologic standard: low value = bone = white)
        let is_monochrome1 = self.photometric.contains("MONOCHROME1");
        if is_monochrome1 {
            let max_val = self.pixels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for p in self.pixels.iter_mut() {
                *p = max_val - *p;
            }
        }

        // Determine window center and width
        let win_center = custom_center.or(self.window_center);
        let win_width = custom_width.or(self.window_width);

        let (effective_center, effective_width) = match (win_center, win_width) {
            (Some(c), Some(w)) if w > 0.0 => (c, w),
            _ => {
                let min_val = self.pixels.iter().copied().fold(f64::INFINITY, f64::min);
                let max_val = self.pixels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let diff = if max_val > min_val { max_val - min_val } else { 1.0 };
                (min_val + diff / 2.0, diff)
            }
        };

        let img_min = effective_center - effective_width / 2.0;
        let img_max = effective_center + effective_width / 2.0;

        // Apply windowing and scale to [0, 255]
        let width = self.cols.max(1);
        let height = self.rows.max(1);
        let mut gray_img = GrayImage::new(width, height);

        for (idx, p) in self.pixels.iter().enumerate() {
            let clamped = p.clamp(img_min, img_max);
            let normalized = ((clamped - img_min) / effective_width * 255.0).clamp(0.0, 255.0);
            let byte_val = normalized.round() as u8;

            let x = (idx as u32) % width;
            let y = (idx as u32) / width;
            if x < width && y < height {
                gray_img.put_pixel(x, y, Luma([byte_val]));
            }
        }

        let metadata = DicomMetadata {
            patient_id: self.patient_id.clone(),
            study_date: self.study_date.clone(),
            modality: self.modality.clone(),
            photometric_interpretation: Some(self.photometric.clone()),
            pixel_spacing: self.pixel_spacing.clone(),
            window_center: Some(effective_center),
            window_width: Some(effective_width),
            rescale_slope: self.rescale_slope,
            rescale_intercept: self.rescale_intercept,
            rows: self.rows,
            columns: self.cols,
            bits_stored: self.bits_stored,
            analysis_status: "completed".to_string(),
        };

        (gray_img, metadata)
    }

    /// Process DICOM and save to destination PNG path
    pub fn convert_to_png<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        custom_center: Option<f64>,
        custom_width: Option<f64>,
    ) -> Result<ProcessedDicomResult> {
        let mut dicom = Self::read_from_file(input_path)?;
        let (image, metadata) = dicom.process_to_gray_image(custom_center, custom_width);

        if let Some(parent) = output_path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        image
            .save(output_path.as_ref())
            .with_context(|| format!("Failed to save PNG to {:?}", output_path.as_ref()))?;

        Ok(ProcessedDicomResult {
            image_path: output_path.as_ref().to_string_lossy().to_string(),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_pixels_and_windowing() {
        let mut dcm = DicomImage {
            rows: 2,
            cols: 2,
            bits_allocated: 16,
            bits_stored: 16,
            is_signed: false,
            photometric: "MONOCHROME2".to_string(),
            window_center: Some(50.0),
            window_width: Some(100.0),
            rescale_slope: None,
            rescale_intercept: None,
            pixel_spacing: Some(vec![0.2, 0.2]),
            patient_id: Some("TEST_PATIENT".to_string()),
            study_date: Some("20261109".to_string()),
            modality: Some("CR".to_string()),
            pixels: vec![0.0, 50.0, 100.0, 150.0],
        };

        let (img, meta) = dcm.process_to_gray_image(None, None);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(meta.patient_id.unwrap(), "TEST_PATIENT");
        assert_eq!(meta.modality.unwrap(), "CR");

        // Values at 0, 50, 100, 150 with center=50, width=100 (range [0, 100])
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
        assert_eq!(img.get_pixel(1, 0).0[0], 128);
        assert_eq!(img.get_pixel(0, 1).0[0], 255);
        assert_eq!(img.get_pixel(1, 1).0[0], 255);
    }

    #[test]
    fn test_monochrome1_inversion() {
        let mut dcm = DicomImage {
            rows: 1,
            cols: 2,
            bits_allocated: 16,
            bits_stored: 16,
            is_signed: false,
            photometric: "MONOCHROME1".to_string(),
            window_center: None,
            window_width: None,
            rescale_slope: None,
            rescale_intercept: None,
            pixel_spacing: None,
            patient_id: None,
            study_date: None,
            modality: None,
            pixels: vec![100.0, 0.0],
        };

        let (img, meta) = dcm.process_to_gray_image(None, None);
        assert_eq!(meta.photometric_interpretation.unwrap(), "MONOCHROME1");
        // Original 100 becomes 0, original 0 becomes 100 -> inverted
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
        assert_eq!(img.get_pixel(1, 0).0[0], 255);
    }
}
