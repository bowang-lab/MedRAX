from typing import Dict, Optional, Tuple, Type
from pathlib import Path
from pydantic import BaseModel, Field
import uuid

import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class ChestXRayClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    model: xrv.models.DenseNet = None
    device: Optional[str] = "cuda"
    transform: torchvision.transforms.Compose = None
    temp_dir: Path = Path("temp")
    generate_gradcam: bool = True  # Generate Grad-CAM visualizations

    def __init__(self, model_name: str = "densenet121-res224-all", device: Optional[str] = "cuda", 
                 temp_dir: Optional[Path] = Path("temp"), generate_gradcam: bool = True):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.generate_gradcam = generate_gradcam

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, normalizes it, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = img[None, :, :]
        img = self.transform(img)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(self.device)

        return img

    def _generate_gradcam(self, img: torch.Tensor, original_img_path: str, top_pathologies: list) -> str:
        """Generate Grad-CAM heatmap for top pathologies"""
        try:
            print(f"[ChestXRayClassifier] 🎨 Generating Grad-CAM visualization...")
            
            # Load original image for overlay
            original_img = skimage.io.imread(original_img_path)
            if len(original_img.shape) > 2:
                original_img = original_img[:, :, 0]
            
            # Get the last convolutional layer
            # torchxrayvision DenseNet wraps the model differently
            if hasattr(self.model, 'model'):
                target_layer = self.model.model.features[-1]
            elif hasattr(self.model, 'features'):
                target_layer = self.model.features[-1]
            else:
                # Fallback: try to find features in the model structure
                for name, module in self.model.named_modules():
                    if 'features' in name:
                        target_layer = module[-1] if hasattr(module, '__getitem__') else module
                        break
                else:
                    raise AttributeError("Cannot find features layer in DenseNet model")
            
            # Forward pass with gradient tracking
            img.requires_grad = True
            self.model.zero_grad()
            
            # Get activations and gradients
            activations = []
            def hook_fn(module, input, output):
                activations.append(output)
            
            handle = target_layer.register_forward_hook(hook_fn)
            
            # Forward pass
            output = self.model(img)
            
            # Get top 3 pathology indices
            pathology_names = xrv.datasets.default_pathologies
            top_indices = [pathology_names.index(p[0]) for p in top_pathologies[:3]]
            
            # Create combined heatmap figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title('Original X-Ray', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Generate heatmap for each top pathology
            for idx, (pathology_idx, ax) in enumerate(zip(top_indices, axes[1:])):
                # Backward pass for this pathology
                self.model.zero_grad()
                output[0, pathology_idx].backward(retain_graph=True)
                
                # Get gradients and activations
                gradients = img.grad
                activation = activations[0]
                
                # Global average pooling of gradients
                weights = F.adaptive_avg_pool2d(gradients, 1)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activation, dim=1, keepdim=True)
                cam = F.relu(cam)  # ReLU to keep only positive influence
                
                # Normalize
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Resize to original image size
                cam = F.interpolate(cam, size=original_img.shape, mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().detach().numpy()
                
                # Create overlay
                ax.imshow(original_img, cmap='gray', alpha=0.7)
                ax.imshow(cam, cmap='jet', alpha=0.5)
                
                pathology_name, pathology_prob = top_pathologies[idx]
                ax.set_title(f'{pathology_name}\n(p={pathology_prob:.3f})', 
                           fontsize=10, fontweight='bold')
                ax.axis('off')
            
            # Remove hook
            handle.remove()
            
            # Save figure
            plt.tight_layout()
            save_path = self.temp_dir / f"gradcam_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            
            print(f"[ChestXRayClassifier] ✅ Grad-CAM saved: {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"[ChestXRayClassifier] ⚠️ Grad-CAM generation failed: {str(e)}")
            return None

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image for multiple pathologies.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        import time
        print(f"[ChestXRayClassifier] 🔬 Starting pathology classification...")
        start_time = time.time()
        
        try:
            print(f"[ChestXRayClassifier] 📂 Loading image: {image_path}")
            img = self._process_image(image_path)
            print(f"[ChestXRayClassifier] ✅ Image loaded successfully - Shape: {img.shape}")

            print(f"[ChestXRayClassifier] 🧠 Running DenseNet-121 inference on {self.device}...")
            
            # Run inference
            if self.generate_gradcam:
                # Enable gradients for Grad-CAM
                img_with_grad = img.clone().detach().requires_grad_(True)
                preds = self.model(img_with_grad).cpu()[0]
            else:
                with torch.inference_mode():
                    preds = self.model(img).cpu()[0]
            
            elapsed = time.time() - start_time
            print(f"[ChestXRayClassifier] ⚡ Inference completed in {elapsed:.2f}s")

            output = dict(zip(xrv.datasets.default_pathologies, preds.detach().numpy()))
            
            # Find top 3 predictions
            top_3 = sorted(output.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            print(f"[ChestXRayClassifier] 📊 Top 3 findings:")
            for pathology, prob in top_3:
                print(f"   • {pathology}: {float(prob):.3f}")
            
            # Generate Grad-CAM visualization if enabled
            gradcam_path = None
            if self.generate_gradcam:
                gradcam_path = self._generate_gradcam(img_with_grad, image_path, top_3)
            
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "inference_time_seconds": elapsed,
                "device": str(self.device),
                "model": "DenseNet-121",
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }
            
            if gradcam_path:
                metadata["gradcam_image_path"] = gradcam_path
            
            print(f"[ChestXRayClassifier] ✅ Classification completed successfully")
            return output, metadata
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[ChestXRayClassifier] ❌ Error after {elapsed:.2f}s: {str(e)}")
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_time_seconds": elapsed,
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path)
