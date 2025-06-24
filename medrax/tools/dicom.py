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
        img_array = dcm.pixel_array

        num_frames = getattr(dcm, "NumberOfFrames", 1)
        selected_frame_info = ""
        if num_frames > 1:
            # For multi-frame images (e.g., cine/video), select the first frame.
            # A more sophisticated approach might select a middle frame or allow user input.
            if img_array.ndim == 3 and img_array.shape[0] == num_frames: # Grayscale multi-frame
                img_array = img_array[0]
                selected_frame_info = f"first frame of {num_frames}"
            elif img_array.ndim == 4 and img_array.shape[0] == num_frames: # Color multi-frame (e.g. RGB)
                 img_array = img_array[0]
                 selected_frame_info = f"first frame of {num_frames} (color)"
            # If ndim doesn't match expected, it might be a complex format not simply handled by frame indexing.
            # For now, we proceed, and pydicom's pixel_array might have already made a choice.

        photometric_interpretation = getattr(dcm, "PhotometricInterpretation", "").upper()

        # Handle color images (e.g., RGB, YBR) separately from grayscale
        if photometric_interpretation in ["RGB", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422", "YBR_PARTIAL_420", "YBR_ICT", "YBR_RCT"]:
            # For color images, pydicom's pixel_array usually returns an array that PIL can handle directly.
            # Ensure it's scaled to uint8 if necessary (though pixel_array often does this for RGB).
            if img_array.dtype != np.uint8:
                # If not uint8, try a simple normalization if it seems like raw pixel data.
                # This is a basic heuristic; color handling can be complex.
                if img_array.max() > 255: # Basic check if normalization might be needed
                     img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5) * 255
                img_array = img_array.astype(np.uint8)
            img = img_array # Already processed or does not need grayscale windowing
            processed_how = "color_direct"
        else: # Assume grayscale or other interpretations needing windowing/rescaling
            img = img_array.astype(float)
            processed_how = "grayscale_windowed"

            # Apply manufacturer's recommended windowing if available and not overridden
            if window_center is None and hasattr(dcm, "WindowCenter"):
                wc = dcm.WindowCenter
                window_center = wc[0] if isinstance(wc, list) else wc
            if window_width is None and hasattr(dcm, "WindowWidth"):
                ww = dcm.WindowWidth
                window_width = ww[0] if isinstance(ww, list) else ww

            # Apply rescale slope/intercept if available
            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

            # Apply windowing if parameters are available
            if window_center is not None and window_width is not None:
                img = self._apply_windowing(img, float(window_center), float(window_width))
            else: # Default normalization if no windowing info
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min: # Avoid division by zero for blank images
                    img = ((img - img_min) / (img_max - img_min)) * 255.0
                else:
                    img = np.zeros_like(img) # Blank image
                img = img.astype(np.uint8)

        metadata = {
            "PatientID": getattr(dcm, "PatientID", None),
            "StudyDate": getattr(dcm, "StudyDate", None),
            "StudyTime": getattr(dcm, "StudyTime", None),
            "Modality": getattr(dcm, "Modality", None),
            "PhotometricInterpretation": photometric_interpretation,
            "Rows": getattr(dcm, "Rows", None),
            "Columns": getattr(dcm, "Columns", None),
            "PixelSpacing": [float(ps) for ps in dcm.PixelSpacing] if hasattr(dcm, "PixelSpacing") and dcm.PixelSpacing else None,
            "WindowCenter": window_center,
            "WindowWidth": window_width,
            "RescaleIntercept": float(dcm.RescaleIntercept) if hasattr(dcm, "RescaleIntercept") else None,
            "RescaleSlope": float(dcm.RescaleSlope) if hasattr(dcm, "RescaleSlope") else None,
            "ImageOrientationPatient": [float(iop) for iop in dcm.ImageOrientationPatient] if hasattr(dcm, "ImageOrientationPatient") and dcm.ImageOrientationPatient else None,
            "ImagePositionPatient": [float(ipp) for ipp in dcm.ImagePositionPatient] if hasattr(dcm, "ImagePositionPatient") and dcm.ImagePositionPatient else None,
            "BitsStored": getattr(dcm, "BitsStored", None),
            "NumberOfFrames": num_frames,
            "SelectedFrameInfo": selected_frame_info if selected_frame_info else "single_frame",
            "ProcessingMethod": processed_how,
            # Ultrasound specific (examples, add more as needed)
            "FrameTime": float(dcm.FrameTime) if hasattr(dcm, "FrameTime") and dcm.FrameTime else None,
            "Manufacturer": getattr(dcm, "Manufacturer", None),
            "ManufacturerModelName": getattr(dcm, "ManufacturerModelName", None),
        }
        # Ensure PIL can handle the image (e.g. if it's single channel, make sure it's 2D)
        if img.ndim == 3 and img.shape[-1] == 1 : # (H, W, 1)
            img = img.squeeze(axis=-1) # Convert to (H,W) for grayscale

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
