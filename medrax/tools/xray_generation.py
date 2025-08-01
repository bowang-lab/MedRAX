from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import uuid
import tempfile
import torch
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class ChestXRayGeneratorInput(BaseModel):
    """Input schema for the Chest X-Ray Generator Tool."""
    
    prompt: str = Field(
        ..., 
        description="Description of the medical condition to generate (e.g., 'big left-sided pleural effusion')"
    )
    height: int = Field(
        512,
        description="Height of generated image in pixels"
    )
    width: int = Field(
        512,
        description="Width of generated image in pixels"
    )
    num_inference_steps: int = Field(
        75,
        description="Number of denoising steps (higher = better quality but slower)"
    )
    guidance_scale: float = Field(
        4.0,
        description="How closely to follow the prompt (higher = more faithful but less diverse)"
    )


class ChestXRayGeneratorTool(BaseTool):
    """Tool for generating synthetic chest X-ray images using a fine-tuned Stable Diffusion model."""

    name: str = "chest_xray_generator"
    description: str = (
        "Generates synthetic chest X-ray images from text descriptions of medical conditions. "
        "Input: Text description of the medical finding or condition to generate, "
        "along with optional parameters for image size (height, width), "
        "quality (num_inference_steps), and prompt adherence (guidance_scale). "
        "Output: Path to the generated X-ray image and generation metadata."
    )
    args_schema: Type[BaseModel] = ChestXRayGeneratorInput

    model: StableDiffusionPipeline = None
    device: torch.device = None
    temp_dir: Path = None

    def __init__(
        self,
        model_path: str = "/model-weights/roentgen",
        cache_dir: str = "/model-weights",
        temp_dir: Optional[str] = None,
        device: Optional[str] = "cuda",
    ):
        """Initialize the chest X-ray generator tool."""
        super().__init__()
        
        self.device = torch.device(device) if device else "cuda"
        self.model = StableDiffusionPipeline.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = self.model.to(torch.float32).to(self.device)
        
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _run(
        self,
        prompt: str,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        height: int = 512,
        width: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Generate a chest X-ray image from a text description.

        Args:
            prompt: Text description of the medical condition to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            height: Height of generated image in pixels
            width: Width of generated image in pixels
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary with image path and metadata dictionary
        """
        try:
            # Generate image
            generation_output = self.model(
                [prompt],
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale
            )

            # Save generated image
            image_path = self.temp_dir / f"generated_xray_{uuid.uuid4().hex[:8]}.png"
            generation_output.images[0].save(image_path)

            output = {
                "image_path": str(image_path),
            }
            
            metadata = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "device": str(self.device),
                "image_size": (height, width),
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {
                    "prompt": prompt,
                    "analysis_status": "failed",
                    "error_details": str(e),
                }
            )

    async def _arun(
        self,
        prompt: str,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        height: int = 512,
        width: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Async version of _run."""
        return self._run(prompt, num_inference_steps, guidance_scale, height, width)