import os
from typing import ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pydantic import BaseModel, Field
from timm.models.swin_transformer import SwinTransformer

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class OmniSwinTransformer(SwinTransformer):
    """OmniSwinTransformer with multiple classification heads and optional projector."""

    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None

        self.projector = None
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features),
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(
                nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            )
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj and self.projector:
            x = self.projector(x)
        return x


class ArcPlusInput(BaseModel):
    """Input for ArcPlus chest X-ray analysis tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class ArcPlusClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images using the ArcPlus OmniSwinTransformer model.

    This tool uses a pre-trained OmniSwinTransformer model (ArcPlus) to analyze chest X-ray images
    and predict the likelihood of various pathologies across multiple medical datasets. The model
    employs a Swin Transformer architecture with multiple classification heads, each specialized
    for different medical datasets and conditions.

    The ArcPlus model is trained on 6 different medical datasets:
    - MIMIC-CXR: 14 pathologies including common chest conditions
    - CheXpert: 14 pathologies with standardized labeling
    - NIH ChestX-ray14: 14 pathologies from large-scale dataset
    - RSNA: 3 classes for pneumonia detection
    - VinDr-CXR: 6 categories including tuberculosis and lung tumors
    - Shenzhen: 1 class for tuberculosis detection

    Key Features:
    - Multi-head architecture with 6 specialized classification heads
    - 768x768 input resolution for high-detail analysis
    - Projector layer with 1376 features for enhanced representation
    - Sigmoid activation for multi-label classification
    - Covers 52+ distinct pathology categories across datasets

    The model outputs probabilities (0 to 1) for each condition, with higher values
    indicating higher likelihood of the pathology being present in the image.
    """

    name: str = "arcplus_classifier"
    description: str = (
        "Advanced chest X-ray classification tool using ArcPlus OmniSwinTransformer with multi-dataset training. "
        "Analyzes chest X-ray images and provides probability predictions for 52+ pathologies across 6 medical datasets. "
        "Input: Path to chest X-ray image file (JPG/PNG). "
        "Output: Dictionary mapping pathology names to probabilities (0-1). "
        "Features: Multi-head architecture, 768px resolution, projector layer, specialized for medical imaging. "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, "
        "Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax, Mass, Nodule, "
        "Emphysema, Fibrosis, PE, Lung Tumor, Tuberculosis, and many more across MIMIC, CheXpert, NIH, "
        "RSNA, VinDr, and Shenzhen datasets. Higher probabilities indicate higher likelihood of condition presence."
    )
    args_schema: Type[BaseModel] = ArcPlusInput
    model: OmniSwinTransformer = None
    device: Optional[str] = "cuda"
    normalize: transforms.Normalize = None
    disease_list: List[str] = None
    num_classes_list: List[int] = None

    # Disease mappings from the analysis
    mimic_diseases: ClassVar[List[str]] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
    chexpert_diseases: ClassVar[List[str]] = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    nih14_diseases: ClassVar[List[str]] = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    rsna_diseases: ClassVar[List[str]] = ["No Lung Opacity/Not Normal", "Normal", "Lung Opacity"]
    vindr_diseases: ClassVar[List[str]] = [
        "PE",
        "Lung tumor",
        "Pneumonia",
        "Tuberculosis",
        "Other diseases",
        "No finding",
    ]
    shenzhen_diseases: ClassVar[List[str]] = ["TB"]

    def __init__(self, cache_dir: str = None, device: Optional[str] = "cuda"):
        """Initialize the ArcPlus Classifier Tool.

        Args:
            cache_dir (str, optional): Directory containing the pre-trained ArcPlus model checkpoint.
                The tool will automatically look for 'Ark6_swinLarge768_ep50.pth.tar' in this directory.
                If None, model will be initialized with random weights (not recommended for inference).
                Default: None.
            device (str, optional): Device to run the model on ('cuda' for GPU, 'cpu' for CPU).
                GPU is recommended for better performance. Default: "cuda".

        Model Architecture Details:
            - OmniSwinTransformer with 6 classification heads
            - Input resolution: 768x768 pixels
            - Projector features: 1376 dimensions
            - Multi-head configuration: [14, 14, 14, 3, 6, 1] classes per head
            - Total pathologies: 52+ across 6 medical datasets
            - Preprocessing: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        Raises:
            FileNotFoundError: If cache_dir is provided but model file doesn't exist.
            RuntimeError: If model loading fails or device is unavailable.
        """
        super().__init__()

        # Create combined disease list from all supported datasets
        self.disease_list = (
            self.mimic_diseases
            + self.chexpert_diseases
            + self.nih14_diseases
            + self.rsna_diseases
            + self.vindr_diseases
            + self.shenzhen_diseases
        )

        # Multi-head configuration: [MIMIC, CheXpert, NIH, RSNA, VinDr, Shenzhen]
        self.num_classes_list = [14, 14, 14, 3, 6, 1]

        # Initialize the OmniSwinTransformer model with ArcPlus architecture
        self.model = OmniSwinTransformer(
            num_classes_list=self.num_classes_list,
            projector_features=1376,  # Enhanced feature representation
            use_mlp=False,  # Linear projector (not MLP)
            img_size=768,  # High-resolution input
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),  # Swin-Large configuration
            num_heads=(6, 12, 24, 48),
        )

        # Load pre-trained weights if provided
        if cache_dir:
            model_path = os.path.join(cache_dir, "Ark6_swinLarge768_ep50.pth.tar")
            self._load_checkpoint(model_path)

        self.model.eval()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)

        # ImageNet normalization parameters for optimal performance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_checkpoint(self, model_path: str) -> None:
        """
        Load the ArcPlus model checkpoint.

        Args:
            model_path (str): Path to the model checkpoint file.
        """
        # Load the checkpoint (set weights_only=False for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        state_dict = checkpoint["teacher"]  # Use 'teacher' key

        # Remove "module." prefix if present (improved logic from example)
        if any([True if "module." in k else False for k in state_dict.keys()]):
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if k.startswith("module.")
            }

        # Load the model weights
        msg = self.model.load_state_dict(state_dict, strict=False)

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        try:
            # Load and preprocess image following the example pattern
            image = Image.open(image_path).convert("RGB").resize((768, 768))

            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image) / 255.0

            # Apply ImageNet normalization
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            image_tensor = self.normalize(image_tensor)

            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            return image_tensor

        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image using ArcPlus SwinTransformer.

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
        try:
            # Process the image
            image_tensor = self._process_image(image_path)

            # Run model inference
            with torch.no_grad():
                pre_logits = self.model(image_tensor)

                # Apply sigmoid to each output head (as seen in example)
                preds = [torch.sigmoid(out) for out in pre_logits]

                # Concatenate all predictions into single tensor
                preds = torch.cat(preds, dim=1)

                # Convert to numpy
                predictions = preds.cpu().numpy().flatten()

            # Map predictions to disease names
            if len(predictions) != len(self.disease_list):
                print(
                    f"Warning: Expected {len(self.disease_list)} predictions, got {len(predictions)}"
                )
                # Pad or truncate as needed
                if len(predictions) < len(self.disease_list):
                    predictions = np.pad(
                        predictions, (0, len(self.disease_list) - len(predictions))
                    )
                else:
                    predictions = predictions[: len(self.disease_list)]

            # Create output dictionary mapping disease names to probabilities
            output = dict(zip(self.disease_list, predictions.astype(float)))

            metadata = {
                "image_path": image_path,
                "model": "ArcPlus OmniSwinTransformer",
                "analysis_status": "completed",
                "num_predictions": len(predictions),
                "num_heads": len(self.num_classes_list),
                "projector_features": 1376,
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }

            return output, metadata

        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": str(e),
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image using ArcPlus SwinTransformer.

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
