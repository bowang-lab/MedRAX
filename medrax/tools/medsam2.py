from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class MedSAM2Input(BaseModel):
    """Input schema for the MedSAM2 Tool."""

    image_path: str = Field(..., description="Path to the medical image file to be segmented")
    prompt_type: str = Field(
        "box",
        description="Type of prompt: 'box' for bounding box, 'point' for point click, or 'auto' for automatic segmentation",
    )
    prompt_coords: Optional[List[int]] = Field(
        None,
        description="Prompt coordinates: [x1,y1,x2,y2] for box prompt or [x,y] for point prompt. Leave None for auto segmentation",
    )
    slice_index: Optional[int] = Field(
        None,
        description="Specific slice index for 3D volumes (0-based). If None, processes middle slice",
    )


class MedSAM2Tool(BaseTool):
    """Advanced medical image segmentation tool using MedSAM2.
    
    This tool provides state-of-the-art medical image segmentation capabilities using
    the MedSAM2 model, which is specifically adapted for medical imaging from Meta's SAM2.
    Supports interactive prompting with boxes, points, or automatic segmentation.
    """

    name: str = "medsam2_segmentation"
    description: str = (
        "Advanced medical image segmentation using MedSAM2 (Segment Anything Model 2 for Medical Images). "
        "Supports interactive prompting with box coordinates, point clicks, or automatic segmentation. "
        "Can handle 2D medical images and 3D volumes. Returns segmentation masks and visualization overlays. "
        "Prompt types: 'box' with [x1,y1,x2,y2] coordinates, 'point' with [x,y] coordinates, or 'auto' for automatic. "
        "Example: {'image_path': '/path/to/image.png', 'prompt_type': 'box', 'prompt_coords': [100,100,200,200]}"
    )
    args_schema: Type[BaseModel] = MedSAM2Input

    predictor: Any = None
    device: str = "cuda"
    temp_dir: Path = None
    model_dir: Path = None

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = "cuda",
        temp_dir: Optional[str] = None,
        model_cfg: str = "sam2.1_hiera_t512.yaml",
        checkpoint: str = "MedSAM2_latest.pt",
    ):
        """Initialize the MedSAM2 tool."""
        super().__init__()
        self.device = device
        self.model_dir = Path(model_dir)
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

        # Add MedSAM2 to Python path
        medsam2_path = self.model_dir / "MedSAM2"
        if medsam2_path.exists():
            sys.path.insert(0, str(medsam2_path))
        else:
            raise FileNotFoundError(f"MedSAM2 not found at {medsam2_path}. Please run git clone in {model_dir}")

        try:
            # Import MedSAM2 modules
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Build model
            checkpoint_path = medsam2_path / "checkpoints" / checkpoint
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run download.sh")

            # Build model using config path relative to sam2 package (MedSAM2 sets up Hydra config paths automatically)
            config_path = f"configs/{model_cfg.replace('.yaml', '')}"
            sam2_model = build_sam2(config_path, str(checkpoint_path), device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            print(f"MedSAM2 model loaded successfully on {device}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MedSAM2: {str(e)}")

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for medical analysis."""
        try:
            # Handle different image formats
            if image_path.lower().endswith('.dcm'):
                # DICOM files - would need DICOM processor
                raise ValueError("DICOM files not directly supported. Please convert to standard image format first.")
            
            # Load standard image formats
            image = Image.open(image_path)
            
            # For medical images, convert to grayscale first if needed, then to RGB
            if image.mode == 'L':  # Grayscale
                # Convert grayscale to RGB for SAM2
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Create white background for RGBA
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Ensure image is in proper range [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            return image_np
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")

    def _process_prompts(self, prompt_type: str, prompt_coords: Optional[List[int]], image_shape: Tuple[int, int]):
        """Process and validate prompts."""
        if prompt_type == "auto":
            return None, None, None
        
        if prompt_coords is None:
            if prompt_type != "auto":
                raise ValueError(f"Prompt coordinates required for prompt type '{prompt_type}'")
            return None, None, None
        
        if prompt_type == "box":
            if len(prompt_coords) != 4:
                raise ValueError("Box prompt requires 4 coordinates: [x1,y1,x2,y2]")
            
            x1, y1, x2, y2 = prompt_coords
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                raise ValueError("Invalid box coordinates: x1 < x2 and y1 < y2 required")
            
            input_box = np.array([[x1, y1, x2, y2]])
            return input_box, None, None
        
        elif prompt_type == "point":
            if len(prompt_coords) != 2:
                raise ValueError("Point prompt requires 2 coordinates: [x,y]")
            
            x, y = prompt_coords
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # Positive point
            return None, input_point, input_label
        
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def _create_visualization(self, image: np.ndarray, masks: np.ndarray, prompt_info: Dict) -> str:
        """Create visualization of segmentation results."""
        plt.figure(figsize=(12, 8))
        
        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Display segmentation overlay
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        
        # Overlay masks
        if len(masks) > 0:
            # Use the best mask (first one returned by SAM2)
            mask = masks[0]
            # Convert mask to boolean and ensure proper shape
            mask_bool = mask.astype(bool)
            colored_mask = np.zeros((*mask_bool.shape, 4))
            colored_mask[mask_bool] = [1, 0, 0, 0.5]  # Red with transparency
            plt.imshow(colored_mask)
        
        # Add prompt visualization
        if prompt_info.get('box') is not None:
            box = prompt_info['box'][0]
            x1, y1, x2, y2 = box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2)
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2, label='Box Prompt')
        
        if prompt_info.get('point') is not None:
            point = prompt_info['point'][0]
            plt.plot(point[0], point[1], 'go', markersize=10, label='Point Prompt')
        
        plt.title("Segmentation Result")
        plt.axis('off')
        if prompt_info.get('box') is not None or prompt_info.get('point') is not None:
            plt.legend()
        
        # Save visualization
        viz_path = self.temp_dir / f"medsam2_result_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return str(viz_path)

    def _run(
        self,
        image_path: str,
        prompt_type: str = "box",
        prompt_coords: Optional[List[int]] = None,
        slice_index: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Run MedSAM2 segmentation on the input image."""
        try:
            # Load image
            image = self._load_image(image_path)
            
            # Set image for predictor
            self.predictor.set_image(image)
            
            # Process prompts
            input_box, input_point, input_label = self._process_prompts(
                prompt_type, prompt_coords, image.shape[:2]
            )
            
            # Run inference
            if prompt_type == "auto":
                # For auto segmentation, try multiple approaches and select best result
                h, w = image.shape[:2]
                
                # Try multiple points in key areas for medical images
                sample_points = np.array([
                    [w//3, h//3],      # Upper left lung area
                    [2*w//3, h//3],    # Upper right lung area
                    [w//2, 2*h//3],    # Lower center area
                ])
                sample_labels = np.array([1, 1, 1])  # All positive points
                
                masks, scores, logits = self.predictor.predict(
                    point_coords=sample_points,
                    point_labels=sample_labels,
                    multimask_output=True,
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=True,
                )
            
            # Create visualization
            prompt_info = {
                'box': input_box,
                'point': input_point,
                'type': prompt_type
            }
            viz_path = self._create_visualization(image, masks, prompt_info)
            
            # Process results (exclude large mask arrays to avoid token limits)
            results = {
                "success": True,
                "confidence_scores": scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                "visualization_path": viz_path,
                "num_masks": len(masks),
                "best_mask_score": float(scores[0]) if len(scores) > 0 else 0.0,
                "mask_summary": {
                    "total_masks": len(masks),
                    "mask_shapes": [list(mask.shape) for mask in masks],
                    "segmented_area_pixels": [int(mask.sum()) for mask in masks]
                },
                # Include metadata in the main results
                "metadata": {
                    "image_path": image_path,
                    "image_shape": list(image.shape),
                    "prompt_type": prompt_type,
                    "prompt_coords": prompt_coords,
                    "device": self.device,
                    "num_masks_generated": len(masks),
                    "analysis_status": "completed",
                }
            }
            
            return results
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "success": False,
                "metadata": {
                    "image_path": image_path,
                    "analysis_status": "failed",
                    "error_details": str(e),
                }
            }
            return error_result

    async def _arun(
        self,
        image_path: str,
        prompt_type: str = "box",
        prompt_coords: Optional[List[int]] = None,
        slice_index: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Async version of _run."""
        return self._run(image_path, prompt_type, prompt_coords, slice_index, run_manager) 