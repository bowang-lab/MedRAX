import os
import json
import shutil
import subprocess
from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import uuid
import tempfile
import numpy as np
import pydicom
from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


def _find_rust_engine() -> Optional[str]:
    """Find the medrax-dicom Rust engine executable if available."""
    if env_bin := os.getenv("MEDRAX_DICOM_BIN"):
        if os.path.isfile(env_bin):
            return env_bin

    base_dir = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "crates" / "medrax-core" / "target" / "release" / "medrax-dicom.exe",
        base_dir / "crates" / "medrax-core" / "target" / "release" / "medrax-dicom",
        base_dir / "crates" / "medrax-core" / "target" / "debug" / "medrax-dicom.exe",
        base_dir / "crates" / "medrax-core" / "target" / "debug" / "medrax-dicom",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("medrax-dicom")


class DicomProcessorInput(BaseModel):
    """Input schema for the DICOM Processor Tool."""

    dicom_path: str = Field(..., description="Path to the DICOM file")
    window_center: Optional[float] = Field(
        None, description="Window center for contrast adjustment"
    )
    window_width: Optional[float] = Field(None, description="Window width for contrast adjustment")


class DicomProcessorTool(BaseTool):
    """Tool for processing DICOM files and converting them to PNG images."""

    name: str = "dicom_processor"
    description: str = (
        "Processes DICOM medical image files and converts them to standard image format. "
        "No tool supports dicom natively, so this tool is used to convert dicom to png. "
        "Handles window/level adjustments and proper scaling. "
        "Input: Path to DICOM file and optional window/level parameters. "
        "Output: Path to processed image file and DICOM metadata."
    )
    args_schema: Type[BaseModel] = DicomProcessorInput
    temp_dir: Path = None

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize the DICOM processor tool."""
        super().__init__()
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _apply_windowing(self, img: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply window/level adjustment to the image."""
        if width <= 0:
            width = 1.0
        img_min = center - width / 2.0
        img_max = center + width / 2.0
        img = np.clip(img, img_min, img_max)
        img = ((img - img_min) / width * 255.0).astype(np.uint8)
        return img

    def _process_dicom(
        self,
        dicom_path: str,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Process DICOM file and extract metadata."""
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(float)

        photometric = str(getattr(dcm, "PhotometricInterpretation", "")).strip().upper()
        # Invert MONOCHROME1 so 0 is dark and maximum is bright (radiologic standard display)
        if photometric == "MONOCHROME1":
            img = img.max() - img

        # Apply manufacturer's recommended windowing if available and not overridden
        if window_center is None and hasattr(dcm, "WindowCenter"):
            window_center = dcm.WindowCenter
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]

        if window_width is None and hasattr(dcm, "WindowWidth"):
            window_width = dcm.WindowWidth
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]

        # Apply rescale slope/intercept if available
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept

        # Apply windowing if parameters are available
        if window_center is not None and window_width is not None and window_width > 0:
            img = self._apply_windowing(img, window_center, window_width)
        else:
            diff = img.max() - img.min()
            if diff > 0:
                img = ((img - img.min()) / diff * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        metadata = {
            "PatientID": getattr(dcm, "PatientID", None),
            "StudyDate": getattr(dcm, "StudyDate", None),
            "Modality": getattr(dcm, "Modality", None),
            "PhotometricInterpretation": photometric,
            "PixelSpacing": getattr(dcm, "PixelSpacing", None),
            "WindowCenter": window_center,
            "WindowWidth": window_width,
            "ImageOrientation": getattr(dcm, "ImageOrientationPatient", None),
            "ImagePosition": getattr(dcm, "ImagePositionPatient", None),
            "BitsStored": getattr(dcm, "BitsStored", None),
        }

        return img, metadata

    def _run(
        self,
        dicom_path: str,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Process DICOM file and save as viewable image.

        Args:
            dicom_path: Path to input DICOM file
            window_center: Optional center value for windowing
            window_width: Optional width value for windowing
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary with processed image path and metadata dictionary
        """
        # Attempt accelerated Rust processing first
        rust_engine = _find_rust_engine()
        if rust_engine and Path(dicom_path).is_file():
            try:
                output_path = self.temp_dir / f"processed_dicom_{uuid.uuid4().hex[:8]}.png"
                cmd = [
                    rust_engine,
                    "process-file",
                    "--input",
                    str(dicom_path),
                    "--output",
                    str(output_path),
                ]
                if window_center is not None:
                    cmd.extend(["--window-center", str(window_center)])
                if window_width is not None:
                    cmd.extend(["--window-width", str(window_width)])

                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(res.stdout)
                output = {"image_path": str(output_path)}
                raw_meta = data.get("metadata", {})
                metadata = {
                    "PatientID": raw_meta.get("patient_id"),
                    "patient_id": raw_meta.get("patient_id"),
                    "StudyDate": raw_meta.get("study_date"),
                    "study_date": raw_meta.get("study_date"),
                    "Modality": raw_meta.get("modality"),
                    "modality": raw_meta.get("modality"),
                    "PhotometricInterpretation": raw_meta.get("photometric_interpretation"),
                    "photometric_interpretation": raw_meta.get("photometric_interpretation"),
                    "PixelSpacing": raw_meta.get("pixel_spacing"),
                    "pixel_spacing": raw_meta.get("pixel_spacing"),
                    "WindowCenter": raw_meta.get("window_center"),
                    "window_center": raw_meta.get("window_center"),
                    "WindowWidth": raw_meta.get("window_width"),
                    "window_width": raw_meta.get("window_width"),
                    "RescaleSlope": raw_meta.get("rescale_slope"),
                    "rescale_slope": raw_meta.get("rescale_slope"),
                    "RescaleIntercept": raw_meta.get("rescale_intercept"),
                    "rescale_intercept": raw_meta.get("rescale_intercept"),
                    "BitsStored": raw_meta.get("bits_stored"),
                    "bits_stored": raw_meta.get("bits_stored"),
                    "original_path": dicom_path,
                    "output_path": str(output_path),
                    "engine": "rust-medrax-core",
                    "analysis_status": "completed",
                }
                return output, metadata
            except Exception:
                # If Rust execution encounters an issue, gracefully fall back to Python
                pass

        try:
            # Python fallback: process DICOM and save as PNG
            img_array, metadata = self._process_dicom(dicom_path, window_center, window_width)
            output_path = self.temp_dir / f"processed_dicom_{uuid.uuid4().hex[:8]}.png"
            Image.fromarray(img_array).save(output_path)

            output = {
                "image_path": str(output_path),
            }

            metadata.update(
                {
                    "original_path": dicom_path,
                    "output_path": str(output_path),
                    "engine": "python-fallback",
                    "analysis_status": "completed",
                }
            )

            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {
                    "dicom_path": dicom_path,
                    "analysis_status": "failed",
                    "error_details": str(e),
                },
            )

    async def _arun(
        self,
        dicom_path: str,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Async version of _run."""
        return self._run(dicom_path, window_center, window_width)
