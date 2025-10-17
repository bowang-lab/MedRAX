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


class DicomProcessorInput(BaseModel):
    """Input schema for the DICOM Processor Tool."""

    dicom_path: str = Field(..., description="Path to the DICOM file")
    window_center: Optional[float] = Field(None, description="Window center for contrast adjustment")
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
        img_min = center - width // 2
        img_max = center + width // 2
        img = np.clip(img, img_min, img_max)
        img = ((img - img_min) / (width) * 255).astype(np.uint8)
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

        # Apply manufacturer's recommended windowing if available and not overridden
        if window_center is None and hasattr(dcm, "WindowCenter"):
            window_center = dcm.WindowCenter
            if isinstance(window_center, list):
                window_center = window_center[0]

        if window_width is None and hasattr(dcm, "WindowWidth"):
            window_width = dcm.WindowWidth
            if isinstance(window_width, list):
                window_width = window_width[0]

        # Apply rescale slope/intercept if available
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept

        # Apply windowing if parameters are available
        if window_center is not None and window_width is not None:
            img = self._apply_windowing(img, window_center, window_width)
        else:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        metadata = {
            "PatientID": getattr(dcm, "PatientID", None),
            "StudyDate": getattr(dcm, "StudyDate", None),
            "Modality": getattr(dcm, "Modality", None),
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
        try:
            # Process DICOM and save as PNG
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
