from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import torch
import numpy as np

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from PIL import Image

from transformers import (
    BertTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
)


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(..., description="Path to the radiology image file, only supports JPG or PNG images")


class ChestXRayReportGeneratorTool(BaseTool):
    """Tool that generates comprehensive chest X-ray reports with both findings and impressions.

    This tool uses two Vision-Encoder-Decoder models (ViT-BERT) trained on CheXpert
    and MIMIC-CXR datasets to generate structured radiology reports. It automatically
    generates both detailed findings and impression summaries for each chest X-ray,
    following standard radiological reporting format.

    The tool uses:
    - Findings model: Generates detailed observations of all visible structures
    - Impression model: Provides concise clinical interpretation and key diagnoses
    """

    name: str = "chest_xray_report_generator"
    description: str = (
        "A tool that analyzes chest X-ray images and generates comprehensive radiology reports "
        "containing both detailed findings and impression summaries. Input should be the path "
        "to a chest X-ray image file. Output is a structured report with both detailed "
        "observations and key clinical conclusions."
    )
    device: Optional[str] = "cuda"
    args_schema: Type[BaseModel] = ChestXRayInput
    findings_model: VisionEncoderDecoderModel = None
    impression_model: VisionEncoderDecoderModel = None
    findings_tokenizer: BertTokenizer = None
    impression_tokenizer: BertTokenizer = None
    findings_processor: ViTImageProcessor = None
    impression_processor: ViTImageProcessor = None
    generation_args: Dict[str, Any] = None

    def __init__(self, cache_dir: str = "/model-weights", device: Optional[str] = "cuda"):
        """Initialize the ChestXRayReportGeneratorTool with both findings and impression models."""
        super().__init__()
        self.device = torch.device(device) if device else "cuda"

        # Initialize findings model
        self.findings_model = VisionEncoderDecoderModel.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline", cache_dir=cache_dir
        ).eval()
        self.findings_tokenizer = BertTokenizer.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline", cache_dir=cache_dir
        )
        self.findings_processor = ViTImageProcessor.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline", cache_dir=cache_dir
        )

        # Initialize impression model
        self.impression_model = VisionEncoderDecoderModel.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline", cache_dir=cache_dir
        ).eval()
        self.impression_tokenizer = BertTokenizer.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline", cache_dir=cache_dir
        )
        self.impression_processor = ViTImageProcessor.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline", cache_dir=cache_dir
        )

        # Move models to device
        self.findings_model = self.findings_model.to(self.device)
        self.impression_model = self.impression_model.to(self.device)

        # Default generation arguments
        self.generation_args = {
            "num_return_sequences": 1,
            "max_length": 128,
            "use_cache": True,
            "beam_width": 2,
        }

    def _process_image(
        self, image_path: str, processor: ViTImageProcessor, model: VisionEncoderDecoderModel
    ) -> torch.Tensor:
        """Process the input image for a specific model.

        Args:
            image_path (str): Path to the input image.
            processor: Image processor for the specific model.
            model: The model to process the image for.

        Returns:
            torch.Tensor: Processed image tensor ready for model input.
        """
        image = Image.open(image_path)
        
        # Properly handle 16-bit grayscale images (common in medical imaging)
        if image.mode == "I;16":
            # Convert 16-bit to 8-bit by normalizing to 0-255 range
            img_array = np.array(image)
            img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            image = Image.fromarray(img_normalized, mode='L')
        
        image = image.convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        expected_size = model.config.encoder.image_size
        actual_size = pixel_values.shape[-1]

        if expected_size != actual_size:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(expected_size, expected_size),
                mode="bilinear",
                align_corners=False,
            )

        pixel_values = pixel_values.to(self.device)

        return pixel_values

    def _generate_report_section(
        self, pixel_values: torch.Tensor, model: VisionEncoderDecoderModel, tokenizer: BertTokenizer
    ) -> str:
        """Generate a report section using the specified model.

        Args:
            pixel_values: Processed image tensor.
            model: The model to use for generation.
            tokenizer: The tokenizer for the model.

        Returns:
            str: Generated text for the report section.
        """
        generation_config = GenerationConfig(
            **{
                **self.generation_args,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": tokenizer.cls_token_id,
            }
        )

        generated_ids = model.generate(pixel_values, generation_config=generation_config)

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Generate a comprehensive chest X-ray report containing both findings and impression.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager.

        Returns:
            Tuple[Dict[str, Any], Dict]: A tuple containing the output dictionary and metadata dictionary.
        """
        try:
            # Process image for both models
            findings_pixels = self._process_image(image_path, self.findings_processor, self.findings_model)
            impression_pixels = self._process_image(image_path, self.impression_processor, self.impression_model)

            # Generate both sections
            with torch.inference_mode():
                findings_text = self._generate_report_section(
                    findings_pixels, self.findings_model, self.findings_tokenizer
                )
                impression_text = self._generate_report_section(
                    impression_pixels, self.impression_model, self.impression_tokenizer
                )

            # Combine into formatted report
            report = "CHEST X-RAY REPORT\n\n" f"FINDINGS:\n{findings_text}\n\n" f"IMPRESSION:\n{impression_text}"

            output = {
                "report": report,
                "findings": findings_text,
                "impression": impression_text,
            }

            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "sections_generated": ["findings", "impression"],
            }

            return output, metadata

        except Exception as e:
            output = {"error": f"Error generating report: {str(e)}"}
            metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Asynchronously generate a comprehensive chest X-ray report."""
        return self._run(image_path)
