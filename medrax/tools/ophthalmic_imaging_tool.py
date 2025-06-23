# Placeholder for OphthalmicImagingTool
# This tool will handle Eye FFA (Fundus Fluorescein Angiography) and OCT (Optical Coherence Tomography) images/series.

import os
import zipfile
import pydicom
from pydicom.errors import InvalidDicomError
from pathlib import Path
import uuid
import shutil
from PIL import Image
import numpy as np
import cv2 # For video processing for FFA

# Attempt to reuse or adapt logic from MedicalImagingSeriesTool if possible,
# especially for DICOM series handling.

class OphthalmicImagingTool:
    def __init__(self, temp_dir="temp/"):
        self.name = "Ophthalmic Imaging Analyzer"
        self.description = (
            "Processes ophthalmic images like FFA (Fundus Fluorescein Angiography) and OCT "
            "(Optical Coherence Tomography). Handles DICOM series, single images (PNG, JPG), and basic video processing for FFA."
        )
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Windowing parameters might differ for ophthalmic images or might not be needed for standard formats
        # We can adapt the windowing logic if processing raw DICOM pixel data.

    def _apply_windowing_simplified(self, img: np.ndarray) -> np.ndarray:
        """Simplified normalization for display, common for processed ophthalmic images."""
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = ((img - min_val) / (max_val - min_val) * 255)
        else: # Handle flat image
            img = np.zeros_like(img) if min_val == 0 else np.full_like(img, 255)
        return img.astype(np.uint8)

    def _process_dicom_ophthalmic_slice(self, dicom_file_path: str) -> tuple[Optional[np.ndarray], dict]:
        """Processes a single DICOM ophthalmic slice to an image array and extracts metadata."""
        try:
            dcm = pydicom.dcmread(dicom_file_path)
            img_array = dcm.pixel_array.astype(float)

            # Ophthalmic DICOMs might have specific windowing or be post-processed.
            # A general normalization is often sufficient for display if specific tags aren't used.
            # Using RescaleSlope/Intercept if available
            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                img_array = img_array * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

            img_array_processed = self._apply_windowing_simplified(img_array)

            metadata = {
                "PatientID": getattr(dcm, "PatientID", "N/A"),
                "StudyDate": getattr(dcm, "StudyDate", "N/A"),
                "Modality": getattr(dcm, "Modality", "N/A"), # Should be 'OP' for ophthalmic photography, 'OCT' for OCT
                "Laterality": getattr(dcm, "Laterality", getattr(dcm.get("ImageLaterality"), "value", "N/A")), # Common for eyes OD/OS/OU
                "ImageType": getattr(dcm, "ImageType", "N/A"),
                "SeriesDescription": getattr(dcm, "SeriesDescription", "N/A"),
                "ImageComments": getattr(dcm, "ImageComments", "N/A"),
                # OCT Specific (examples, actual tags can vary)
                "OCTAcquisitionDomain": getattr(dcm, "OCTAcquisitionDomain", "N/A"), # e.g. BSCAN, VOLUME
                "ReferencedFrameNumber": getattr(dcm, "ReferencedFrameNumber", "N/A"), # For OCT B-scans from a volume
            }
            # Add more OCT specific tags if needed, e.g. from (0022,xxxx) group or (0052,xxxx)
            # Example: dcm.get((0x0022, 0x0041), None) # AcquisitionDate
            return img_array_processed, metadata
        except Exception as e:
            print(f"Error processing ophthalmic DICOM file {dicom_file_path}: {e}")
            return None, {"error": str(e)}

    def _handle_dicom_series(self, processing_dir: Path) -> dict:
        """Handles a directory of DICOM files, presumably an ophthalmic series."""
        dicom_files_data = []
        for root, _, files in os.walk(processing_dir):
            for file in files:
                file_path_obj = Path(root) / file
                if not file.lower().endswith((".dcm", ".dicom")) and Path(file).suffix: # Skip if known non-DICOM extension
                    continue
                try:
                    dcm_peek = pydicom.dcmread(file_path_obj, stop_before_pixels=True)
                    if "SOPInstanceUID" not in dcm_peek: continue

                    instance_num = getattr(dcm_peek, "InstanceNumber", None)
                    # For ophthalmic, AcquisitionDateTime or ContentDateTime might be better for sorting if available
                    acq_datetime_str = getattr(dcm_peek, "AcquisitionDateTime", None)

                    sort_key_primary = float('inf') # Primary sort key (e.g. datetime)
                    if acq_datetime_str:
                        try: # DICOM DT format: YYYYMMDDHHMMSS.FFFFFF
                            sort_key_primary = float(acq_datetime_str.split('.')[0]) # Use main part for sorting
                        except: pass

                    sort_key_secondary = int(instance_num) if instance_num is not None else float('inf')

                    dicom_files_data.append((str(file_path_obj), sort_key_primary, sort_key_secondary))
                except InvalidDicomError: continue
                except Exception: continue

        if not dicom_files_data:
            return {"error": "No valid DICOM files found in the series directory.", "status": "failed"}

        dicom_files_data.sort(key=lambda x: (x[1], x[2]))
        sorted_dicom_paths = [item[0] for item in dicom_files_data]
        num_slices = len(sorted_dicom_paths)
        representative_slice_path = sorted_dicom_paths[num_slices // 2]

        img_array, rep_slice_metadata = self._process_dicom_ophthalmic_slice(representative_slice_path)

        representative_image_display_path = None
        if img_array is not None:
            output_png_path = self.temp_dir / f"ophth_rep_slice_{uuid.uuid4().hex[:8]}.png"
            Image.fromarray(img_array).save(output_png_path)
            representative_image_display_path = str(output_png_path)

        first_slice_dcm = pydicom.dcmread(sorted_dicom_paths[0], stop_before_pixels=True, force=True)
        series_level_metadata = {
            "PatientID": getattr(first_slice_dcm, "PatientID", "N/A"),
            "StudyDate": getattr(first_slice_dcm, "StudyDate", "N/A"),
            "Modality": getattr(first_slice_dcm, "Modality", "N/A"),
            "Laterality": getattr(first_slice_dcm, "Laterality", getattr(first_slice_dcm.get("ImageLaterality"),"value","N/A")),
            "SeriesDescription": getattr(first_slice_dcm, "SeriesDescription", "N/A"),
        }

        return {
            "status": "completed" if representative_image_display_path else "completed_with_errors",
            "image_type": "DICOM Series",
            "representative_image_path": representative_image_display_path,
            "number_of_images_in_series": num_slices,
            "series_metadata": series_level_metadata,
            "representative_slice_details": rep_slice_metadata
        }

    def _handle_single_image(self, image_path: Path) -> dict:
        """Handles a single image file (PNG, JPG, TIF)."""
        try:
            img = Image.open(image_path)
            # Convert to RGB for consistency if it's not, e.g. palette images
            if img.mode not in ['RGB', 'L', 'RGBA']: # L for grayscale
                 img = img.convert('RGB')

            # For display, save a copy or use original if it's already suitable
            display_path = self.temp_dir / f"ophth_single_img_{uuid.uuid4().hex[:8]}{image_path.suffix}"
            img.save(display_path)

            return {
                "status": "completed",
                "image_type": f"Single Image ({image_path.suffix.upper()})",
                "image_path": str(display_path),
                "metadata": {"filename": image_path.name, "dimensions": img.size}
            }
        except Exception as e:
            return {"error": f"Failed to process single image: {str(e)}", "status": "failed"}

    def _handle_video_ffa(self, video_path: Path) -> dict:
        """Handles an FFA video file, extracting key frames."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"error": "Could not open video file.", "status": "failed"}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Extract a few frames (e.g., start, middle, end)
            frame_indices_to_extract = [0]
            if total_frames > 1:
                frame_indices_to_extract.append(total_frames // 2)
                frame_indices_to_extract.append(total_frames - 1)
            frame_indices_to_extract = sorted(list(set(frame_indices_to_extract))) # Unique, sorted

            extracted_frame_paths = []
            for frame_idx in frame_indices_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_path = self.temp_dir / f"ffa_frame_{frame_idx}_{uuid.uuid4().hex[:8]}.png"
                    frame_img.save(frame_path)
                    extracted_frame_paths.append(str(frame_path))

            cap.release()

            if not extracted_frame_paths:
                return {"error": "Could not extract any frames from video.", "status": "failed"}

            return {
                "status": "completed",
                "image_type": "FFA Video",
                "representative_image_path": extracted_frame_paths[0], # Show first extracted frame as main
                "extracted_frames_paths": extracted_frame_paths,
                "metadata": {"filename": video_path.name, "total_frames": total_frames, "duration_seconds": duration, "fps": fps}
            }
        except Exception as e:
            return {"error": f"Failed to process video FFA: {str(e)}", "status": "failed"}


    def run(self, input_path: str, file_type_hint: Optional[str] = None):
        """
        Processes an ophthalmic image/series.
        Args:
            input_path (str): Path to the image, DICOM directory, ZIP, or video file.
            file_type_hint (str, optional): 'zip', 'dir', 'dicom_series', 'image', 'video'. If None, attempts to infer.
        """
        path_obj = Path(input_path)
        if not path_obj.exists():
            return {"error": f"Input path does not exist: {input_path}", "status": "failed"}

        # Infer file type if hint not provided
        inferred_type = ""
        ext = path_obj.suffix.lower()
        if file_type_hint:
            inferred_type = file_type_hint
        elif path_obj.is_dir() or (ext == ".zip" and zipfile.is_zipfile(input_path)): # Check if dir or valid zip
             # If it's a zip, we'll assume it's a DICOM series for now
            inferred_type = "dicom_series_archive" if ext == ".zip" else "dicom_series_dir"
        elif ext in ['.dcm', '.dicom']: # Single DICOM file (could be part of series, or standalone)
            inferred_type = "single_dicom" # Or treat as 1-file series.
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            inferred_type = "single_image"
        elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
            inferred_type = "video_ffa"
        else: # Default to trying as DICOM series dir if no obvious type
            if path_obj.is_file(): # If it's a file with unknown extension, could be proprietary
                 return {"error": f"Unsupported or unknown single file type: {ext}", "status": "failed"}
            inferred_type = "dicom_series_dir" # Fallback for directories

        temp_extraction_dir = None
        processing_path = path_obj

        if inferred_type == "dicom_series_archive": # Handle ZIP
            temp_extraction_dir = self.temp_dir / f"extracted_ophth_series_{uuid.uuid4().hex[:8]}"
            try:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extraction_dir)
                processing_path = temp_extraction_dir
                print(f"Extracted ophthalmic ZIP archive to {processing_path}")
            except Exception as e:
                return {"error": f"Failed to extract ZIP file: {str(e)}", "status": "failed"}
            # Now processing_path is the directory
            results = self._handle_dicom_series(processing_path)

        elif inferred_type == "dicom_series_dir":
            results = self._handle_dicom_series(processing_path)

        elif inferred_type == "single_dicom": # Handle as a series of one for now, or adapt
            # Create a temp dir and copy the file into it to use _handle_dicom_series
            temp_single_dicom_dir = self.temp_dir / f"single_dcm_temp_{uuid.uuid4().hex[:8]}"
            temp_single_dicom_dir.mkdir()
            shutil.copy(input_path, temp_single_dicom_dir / path_obj.name)
            results = self._handle_dicom_series(temp_single_dicom_dir) # Process as a "series"
            shutil.rmtree(temp_single_dicom_dir) # Clean up temp dir for single DICOM

        elif inferred_type == "single_image":
            results = self._handle_single_image(processing_path)

        elif inferred_type == "video_ffa":
            results = self._handle_video_ffa(processing_path)

        else:
            results = {"error": f"Could not determine how to handle input type: {inferred_type}", "status": "failed"}

        if temp_extraction_dir and temp_extraction_dir.exists():
            shutil.rmtree(temp_extraction_dir)

        return results


if __name__ == '__main__':
    tool_temp_dir = Path("temp_ophth_tool/")
    tool_temp_dir.mkdir(parents=True, exist_ok=True)
    ophth_tool = OphthalmicImagingTool(temp_dir=str(tool_temp_dir))

    # --- Test with a single dummy image ---
    dummy_img_path = tool_temp_dir / "dummy_eye_image.png"
    try:
        Image.new('RGB', (60, 30), color = 'red').save(dummy_img_path)
        print(f"\n--- Testing single image: {dummy_img_path} ---")
        img_results = ophth_tool.run(str(dummy_img_path))
        print("Single Image Results:")
        import json
        print(json.dumps(img_results, indent=2, default=str))
    except Exception as e:
        print(f"Error in single image test setup: {e}")

    # --- Test with dummy DICOM series (similar to MedicalImagingSeriesTool test) ---
    dummy_ophth_series_path = tool_temp_dir / "dummy_ophth_dcm_series"
    dummy_ophth_series_path.mkdir(exist_ok=True)
    # Create a few dummy DICOM files
    for i in range(2): # Shorter series for this test
        meta = pydicom.Dataset()
        meta.InstanceNumber = str(i + 1)
        meta.AcquisitionDateTime = f"20230101100{i}00" # YYYYMMDDHHMMSS
        meta.SOPInstanceUID = pydicom.uid.generate_uid()
        meta.PatientID = "EyePatient"
        meta.Modality = "OCT" # Example modality
        meta.Laterality = "OD" # Right eye
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.5.4' # Ophthalmic Tomography Image Storage
        file_meta.MediaStorageSOPInstanceUID = meta.SOPInstanceUID
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds = pydicom.FileDataset(dummy_ophth_series_path / f"ophth_slice{i+1}.dcm", {}, file_meta=file_meta, preamble=b"\0"*128)
        ds.update(meta)
        ds.PixelData = b'\x00\x00\x00\x00' # Dummy pixels
        ds.Rows, ds.Columns, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 2,2,8,8,7,0,1
        ds.PhotometricInterpretation = "MONOCHROME2"
        try:
            ds.save_as(dummy_ophth_series_path / f"ophth_slice{i+1}.dcm")
        except Exception as e_save:
            print(f"Error saving dummy DICOM {dummy_ophth_series_path / f'ophth_slice{i+1}.dcm'}: {e_save}")

    if list(dummy_ophth_series_path.glob("*.dcm")):
        print(f"\n--- Testing with Ophthalmic DICOM directory: {dummy_ophth_series_path} ---")
        dir_results = ophth_tool.run(str(dummy_ophth_series_path))
        print("Ophthalmic DICOM Series Results:")
        print(json.dumps(dir_results, indent=2, default=str))
    else:
        print(f"\nSkipping Ophthalmic DICOM directory test as {dummy_ophth_series_path} is empty or files failed to save.")

    # --- Test with dummy video (requires OpenCV - cv2) ---
    dummy_video_path = tool_temp_dir / "dummy_ffa.mp4"
    # Create a short, tiny dummy MP4 video using cv2
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
        out = cv2.VideoWriter(str(dummy_video_path), fourcc, 1.0, (10, 10)) # 1fps, 10x10
        if not out.isOpened():
            raise Exception("Failed to open VideoWriter")
        for _ in range(3): # 3 frames
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        print(f"Created dummy video: {dummy_video_path}")

        print(f"\n--- Testing with dummy FFA video: {dummy_video_path} ---")
        video_results = ophth_tool.run(str(dummy_video_path))
        print("FFA Video Results:")
        print(json.dumps(video_results, indent=2, default=str))

    except Exception as e_video:
        print(f"Skipping video test due to error (OpenCV/cv2 might not be fully functional or installed): {e_video}")

    # print(f"\nManual cleanup of {tool_temp_dir} might be needed.")
