# Placeholder for MedicalImagingSeriesTool
# This tool will handle DICOM series (e.g., from MR, CT)
# - Accept a path to a directory of DICOM files or a ZIP file.
# - Extract metadata from the series.
# - Select a representative slice and convert it to PNG for display (possibly using DicomProcessorTool's logic).
# - Output: path to representative PNG, series metadata.

import os
import zipfile
import pydicom
from pydicom.errors import InvalidDicomError
from pathlib import Path
import uuid
import shutil
from PIL import Image # For saving the representative slice
import numpy as np

# Attempt to import DicomProcessorTool to reuse its processing logic, or parts of it.
# If direct import/use is too complex, relevant methods can be adapted.
try:
    from .dicom import DicomProcessorTool, DicomProcessorInput
except ImportError:
    # Fallback if direct import is problematic in some contexts (e.g. running this file standalone for test)
    DicomProcessorTool = None
    print("Warning: DicomProcessorTool not imported directly. Some functionality might be limited or duplicated.")


class MedicalImagingSeriesTool:
    def __init__(self, temp_dir="temp/", dicom_processor_tool_instance=None):
        self.name = "Medical Imaging Series Analyzer"
        self.description = (
            "Processes a series of DICOM files (from a directory or ZIP archive), "
            "extracts metadata, generates a representative image, and provides series information. "
            "Useful for modalities like CT or MR."
        )
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # If an instance of DicomProcessorTool is passed, we can use it.
        # Otherwise, we might need to replicate some of its logic for applying windowing / creating PNGs.
        self.dicom_processor = dicom_processor_tool_instance
        if not self.dicom_processor and DicomProcessorTool:
            # If not passed, try to create one (assuming it's in the same 'tools' package)
            # This might require DicomProcessorTool to be adaptable or its path configured.
            # For simplicity, we'll assume it can be instantiated or its methods adapted.
            print("MedicalImagingSeriesTool: DicomProcessorTool instance not provided, will use adapted logic for image processing.")


    def _apply_windowing(self, img: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply window/level adjustment to the image. Copied/adapted from DicomProcessorTool."""
        img_min = center - width // 2
        img_max = center + width // 2
        img = np.clip(img, img_min, img_max)
        # Handle potential division by zero if width is 0
        if width == 0:
            return (np.zeros_like(img) if img_min == 0 else np.full_like(img, 255)).astype(np.uint8)
        return ((img - img_min) / width * 255).astype(np.uint8)

    def _process_single_dicom_to_image(
        self,
        dicom_file_path: str,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Processes a single DICOM file to an image array and extracts metadata.
        Adapted from DicomProcessorTool._process_dicom.
        """
        try:
            dcm = pydicom.dcmread(dicom_file_path)
            img = dcm.pixel_array.astype(float)

            if window_center is None and hasattr(dcm, "WindowCenter"):
                wc = dcm.WindowCenter
                window_center = wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc
            if window_width is None and hasattr(dcm, "WindowWidth"):
                ww = dcm.WindowWidth
                window_width = ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww

            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

            if window_center is not None and window_width is not None:
                img = self._apply_windowing(img, float(window_center), float(window_width))
            else: # Basic normalization if no windowing info
                min_val, max_val = img.min(), img.max()
                if max_val > min_val:
                    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else: # Handle flat image
                    img = np.zeros_like(img, dtype=np.uint8) if min_val == 0 else np.full_like(img, 255, dtype=np.uint8)


            metadata = {
                "PatientID": getattr(dcm, "PatientID", "N/A"),
                "StudyDate": getattr(dcm, "StudyDate", "N/A"),
                "Modality": getattr(dcm, "Modality", "N/A"),
                "SeriesDescription": getattr(dcm, "SeriesDescription", "N/A"),
                "BodyPartExamined": getattr(dcm, "BodyPartExamined", "N/A"),
                "InstanceNumber": getattr(dcm, "InstanceNumber", None),
                "ImagePositionPatient": getattr(dcm, "ImagePositionPatient", None),
                "SliceThickness": getattr(dcm, "SliceThickness", None),
            }
            return img.astype(np.uint8), metadata
        except Exception as e:
            print(f"Error processing DICOM file {dicom_file_path}: {e}")
            return None, {"error": str(e)}


    def run(self, input_path: str, file_type_hint: Optional[str] = None):
        """
        Processes a DICOM series from a directory or ZIP file.

        Args:
            input_path (str): Path to the DICOM directory or ZIP file.
            file_type_hint (str, optional): 'zip' or 'dir' if known. If None, attempts to infer.

        Returns:
            dict: Results including path to representative image, series metadata, etc.
                  Example: {"representative_image_path": "...", "series_metadata": {...}, "status": "..."}
        """
        series_files = []
        processing_dir = Path(input_path)
        is_zip = (file_type_hint == 'zip') or (input_path.lower().endswith(".zip"))

        temp_extraction_dir = None

        if is_zip:
            temp_extraction_dir = self.temp_dir / f"extracted_series_{uuid.uuid4().hex[:8]}"
            try:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extraction_dir)
                processing_dir = temp_extraction_dir
                print(f"Extracted ZIP archive to {processing_dir}")
            except zipfile.BadZipFile:
                return {"error": "Invalid or corrupted ZIP file.", "status": "failed"}
            except Exception as e:
                return {"error": f"Failed to extract ZIP file: {str(e)}", "status": "failed"}

        if not processing_dir.is_dir():
            return {"error": f"Input path {processing_dir} is not a directory.", "status": "failed"}

        dicom_files_data = [] # List of tuples (filepath, instancenumber, imageposition_z)
        for root, _, files in os.walk(processing_dir):
            for file in files:
                if file.lower().endswith((".dcm", ".dicom")) or not Path(file).suffix : # check common DICOM extensions or no extension
                    file_path_obj = Path(root) / file
                    try:
                        dcm_peek = pydicom.dcmread(file_path_obj, stop_before_pixels=True)
                        if "SOPInstanceUID" not in dcm_peek: # Minimal check for DICOM
                            continue

                        instance_num = getattr(dcm_peek, "InstanceNumber", None)
                        img_pos_pat = getattr(dcm_peek, "ImagePositionPatient", None)

                        # Convert to float for sorting, handle potential errors
                        try:
                            instance_num = int(instance_num) if instance_num is not None else float('-inf')
                        except ValueError:
                            instance_num = float('-inf') # Or some other default for unreadable numbers

                        # Use Z component of ImagePositionPatient for sorting if available
                        # This is often more reliable for slice order than InstanceNumber
                        pos_z = None
                        if img_pos_pat and len(img_pos_pat) == 3:
                            try:
                                pos_z = float(img_pos_pat[2])
                            except (ValueError, TypeError):
                                pos_z = float('-inf')
                        else: # Fallback if ImagePositionPatient is not suitable
                            pos_z = float('-inf')

                        dicom_files_data.append((str(file_path_obj), instance_num, pos_z))
                    except InvalidDicomError:
                        print(f"Skipping non-DICOM or invalid file: {file_path_obj}")
                    except Exception as e:
                        print(f"Error reading attributes from {file_path_obj}: {e}")

        if not dicom_files_data:
            if temp_extraction_dir: shutil.rmtree(temp_extraction_dir)
            return {"error": "No valid DICOM files found in the provided path.", "status": "failed"}

        # Sort files: primarily by Z-position, secondarily by instance number
        dicom_files_data.sort(key=lambda x: (x[2], x[1]))

        sorted_dicom_paths = [item[0] for item in dicom_files_data]
        num_slices = len(sorted_dicom_paths)

        # Select representative slice (e.g., middle one)
        representative_slice_path = sorted_dicom_paths[num_slices // 2]

        # Process the representative slice to get an image and its metadata
        # Using the adapted internal method for now.
        img_array, rep_slice_metadata = self._process_single_dicom_to_image(representative_slice_path)

        representative_image_display_path = None
        if img_array is not None:
            output_png_path = self.temp_dir / f"rep_slice_{uuid.uuid4().hex[:8]}.png"
            try:
                Image.fromarray(img_array).save(output_png_path)
                representative_image_display_path = str(output_png_path)
                print(f"Saved representative slice PNG to {representative_image_display_path}")
            except Exception as e:
                print(f"Error saving representative slice PNG: {e}")
                rep_slice_metadata["png_error"] = str(e)

        # Gather overall series metadata (e.g., from the first slice, assuming it's consistent for series-level tags)
        first_slice_dcm = None
        series_level_metadata = {}
        try:
            first_slice_dcm = pydicom.dcmread(sorted_dicom_paths[0], stop_before_pixels=True)
            series_level_metadata = {
                "PatientID": getattr(first_slice_dcm, "PatientID", "N/A"),
                "StudyDate": getattr(first_slice_dcm, "StudyDate", "N/A"),
                "StudyDescription": getattr(first_slice_dcm, "StudyDescription", "N/A"),
                "Modality": getattr(first_slice_dcm, "Modality", "N/A"),
                "SeriesDescription": getattr(first_slice_dcm, "SeriesDescription", "N/A"),
                "BodyPartExamined": getattr(first_slice_dcm, "BodyPartExamined", "N/A"),
                "SeriesNumber": getattr(first_slice_dcm, "SeriesNumber", "N/A"),
                "Laterality": getattr(first_slice_dcm, "Laterality", "N/A"),
            }
        except Exception as e:
            print(f"Could not read series-level metadata from first slice: {e}")
            series_level_metadata["error"] = "Could not read series-level metadata from first slice."


        # Clean up extracted files if from ZIP
        if temp_extraction_dir:
            try:
                shutil.rmtree(temp_extraction_dir)
                print(f"Cleaned up temporary extraction directory {temp_extraction_dir}")
            except Exception as e:
                print(f"Error cleaning up temp directory {temp_extraction_dir}: {e}")

        return {
            "status": "completed" if representative_image_display_path else "completed_with_errors",
            "representative_image_path": representative_image_display_path,
            "number_of_slices": num_slices,
            "series_metadata": series_level_metadata,
            "representative_slice_details": rep_slice_metadata,
            "sorted_slice_paths_sample": sorted_dicom_paths[:5] # Sample of paths
        }


if __name__ == '__main__':
    # Example Usage (requires sample DICOM series)
    # Create a dummy temp directory for the tool's own temp files
    tool_temp_dir = Path("temp_imaging_series_tool/")
    tool_temp_dir.mkdir(parents=True, exist_ok=True)

    series_tool = MedicalImagingSeriesTool(temp_dir=str(tool_temp_dir))

    # --- Test with a directory of DICOM files ---
    # You would need to create a directory 'test_dicom_series/' with some .dcm files
    # For example:
    # test_series_dir = Path("test_dicom_series/")
    # test_series_dir.mkdir(exist_ok=True)
    # # Populate with a few dummy DICOM files (e.g., copied from online samples or created with pydicom)
    # # Ensure they have InstanceNumber and optionally ImagePositionPatient for sorting.
    # # For a quick test, one might copy a few files from an existing test dataset.

    # Example: to create minimal dummy DICOMs for testing structure (not renderable as valid images)
    dummy_series_path = tool_temp_dir / "dummy_dcm_series"
    dummy_series_path.mkdir(exist_ok=True)
    for i in range(3):
        meta = pydicom.Dataset()
        meta.InstanceNumber = str(i + 1)
        meta.ImagePositionPatient = [0, 0, i * 1.5] # Simulate slice progression
        meta.SOPInstanceUID = pydicom.uid.generate_uid()
        meta.PatientID = "TestPatient"
        meta.Modality = "CT"
        # Add minimal required FileMetaInformation
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = meta.SOPInstanceUID
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = pydicom.FileDataset(
            dummy_series_path / f"slice{i+1}.dcm",
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )
        ds.update(meta)
        # Add dummy pixel data for the file to be considered "valid" by some checks
        # These pixels won't form a meaningful image without more DICOM tags.
        ds.PixelData = b'\x00\x00\x00\x00'
        ds.Rows = 2
        ds.Columns = 2
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        try:
            ds.save_as(dummy_series_path / f"slice{i+1}.dcm")
            print(f"Created dummy DICOM: {dummy_series_path / f'slice{i+1}.dcm'}")
        except Exception as e_save:
            print(f"Error saving dummy DICOM {dummy_series_path / f'slice{i+1}.dcm'}: {e_save}")


    if list(dummy_series_path.glob("*.dcm")): # Check if dummy files were created
        print(f"\n--- Testing with DICOM directory: {dummy_series_path} ---")
        dir_results = series_tool.run(str(dummy_series_path))
        print("Directory Results:")
        import json
        print(json.dumps(dir_results, indent=2, default=str)) # default=str for Path objects etc.
    else:
        print(f"\nSkipping DICOM directory test as {dummy_series_path} is empty or files failed to save.")

    # --- Test with a ZIP file ---
    # You would need to create a ZIP file 'test_dicom_series.zip' containing DICOM files
    # For example, zip the 'test_dicom_series/' directory created above.
    zip_file_path = tool_temp_dir / "test_dicom_series.zip"
    if list(dummy_series_path.glob("*.dcm")):
        try:
            with zipfile.ZipFile(zip_file_path, 'w') as zf:
                for dcm_file in dummy_series_path.glob("*.dcm"):
                    zf.write(dcm_file, arcname=dcm_file.name) # arcname to store files at zip root
            print(f"Created dummy ZIP: {zip_file_path}")

            print(f"\n--- Testing with ZIP file: {zip_file_path} ---")
            zip_results = series_tool.run(str(zip_file_path))
            print("ZIP Results:")
            print(json.dumps(zip_results, indent=2, default=str))
        except Exception as e_zip_create:
            print(f"Could not create or test ZIP file: {e_zip_create}")
    else:
        print(f"\nSkipping ZIP test as source directory {dummy_series_path} was not populated.")

    # print(f"\nManual cleanup of {tool_temp_dir} might be needed.")
