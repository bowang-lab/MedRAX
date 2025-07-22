from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

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
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.
    Additional pathologies may be listed but are currently STUBBED (will return 0.0 or NaN) pending model update:
    Tuberculosis, Pulmonary Fibrosis, ILD (Interstitial Lung Disease), Scoliosis.

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for multiple pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Core pathologies (18) include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Additional pathologies like Tuberculosis, Pulmonary Fibrosis, ILD, Scoliosis are listed but are currently STUBBED (return 0.0/NaN) and require model update for actual prediction. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    model: xrv.models.DenseNet = None
    extended_pathologies: list = None
    device: Optional[str] = "cuda"
    transform: torchvision.transforms.Compose = None

    def __init__(self, model_name: str = "densenet121-res224-all", device: Optional[str] = "cuda"):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

        # Define the full list of pathologies this tool will report on
        # The core 18 are from xrv.datasets.default_pathologies
        self.core_pathologies = sorted(list(xrv.datasets.default_pathologies)) # Ensure consistent order
        self.new_stubbed_pathologies = sorted([
            "Tuberculosis",
            "Pulmonary Fibrosis", # More specific than generic Fibrosis
            "ILD", # Interstitial Lung Disease
            "Scoliosis",
            # Potentially add others here like:
            # "Pneumoperitoneum",
            # "Subcutaneous Emphysema",
            # "Bronchiectasis",
            # "Hilar Enlargement"
        ])
        # Ensure no overlap with core_pathologies if some were to be promoted
        self.new_stubbed_pathologies = [p for p in self.new_stubbed_pathologies if p not in self.core_pathologies]

        self.extended_pathologies = self.core_pathologies + self.new_stubbed_pathologies


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
        try:
            img = self._process_image(image_path)

            with torch.inference_mode():
                preds = self.model(img).cpu()[0]

            # Initialize output with predictions from the core model
            # Ensure self.model.pathologies aligns with how preds are ordered if not default_pathologies
            # For densenet121-res224-all, self.model.pathologies is xrv.datasets.default_pathologies
            output = dict(zip(self.model.pathologies, preds.numpy()))

            # Add stubbed pathologies with a default value (e.g., 0.0 or np.nan)
            # np.nan might be better to indicate "not available" or "not predicted"
            stub_value = 0.0 # Or np.nan, but ensure consistent float output if specified in return type hint
            for pathology in self.new_stubbed_pathologies:
                if pathology not in output: # Should always be true given current logic
                    output[pathology] = stub_value

            # Ensure the final output dictionary has keys in the order of self.extended_pathologies
            # This is mostly for consistent presentation if someone iterates over output.items()
            # However, standard dicts are unordered in older Python, though ordered by insertion in 3.7+
            # For safety, could reconstruct, but dict(zip()) from model is usually sufficient.
            # final_ordered_output = {pathology: output.get(pathology, stub_value) for pathology in self.extended_pathologies}


            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": (
                    "Probabilities range from 0 to 1. Higher values indicate higher likelihood. "
                    "Pathologies marked as STUBBED (e.g., Tuberculosis, Pulmonary Fibrosis, ILD, Scoliosis) "
                    "currently return a default value (0.0) and require model update for actual prediction."
                ),
                "predicted_pathologies": self.core_pathologies,
                "stubbed_pathologies": self.new_stubbed_pathologies,
            }
            # Return the output dictionary which now includes both predicted and stubbed values
            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
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
