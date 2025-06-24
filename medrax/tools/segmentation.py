from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
import traceback

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class ChestXRaySegmentationInput(BaseModel):
    """Input schema for the Chest X-ray Segmentation Tool."""

    image_path: str = Field(..., description="Path to the chest X-ray image file to be segmented")
    organs: Optional[List[str]] = Field(
        None,
        description="List of organs/structures to segment. If None, all available core organs will be segmented. "
        "Core available organs: Left/Right Clavicle, Left/Right Scapula, Left/Right Lung, "
        "Left/Right Hilus Pulmonis, Heart, Aorta, Facies Diaphragmatica (diaphragm), "
        "Mediastinum, Weasand (esophagus), Spine. "
        "Additional structures like Ribs, Pleural Effusion, Consolidation, Nodule/Mass can be requested "
        "but are currently STUBBED (not segmented by the current model) and will be noted in the output.",
    )


class OrganMetrics(BaseModel):
    """Detailed metrics for a segmented organ."""

    # Basic metrics
    area_pixels: int = Field(..., description="Area in pixels")
    area_cm2: float = Field(..., description="Approximate area in cm²")
    centroid: Tuple[float, float] = Field(..., description="(y, x) coordinates of centroid")
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (min_y, min_x, max_y, max_x)"
    )

    # Size metrics
    width: int = Field(..., description="Width of the organ in pixels")
    height: int = Field(..., description="Height of the organ in pixels")
    aspect_ratio: float = Field(..., description="Height/width ratio")

    # Position metrics
    relative_position: Dict[str, float] = Field(
        ..., description="Position relative to image boundaries (0-1 scale)"
    )

    # Analysis metrics
    mean_intensity: float = Field(..., description="Mean pixel intensity in the organ region")
    std_intensity: float = Field(..., description="Standard deviation of pixel intensity")
    confidence_score: float = Field(..., description="Model confidence score for this organ")


class ChestXRaySegmentationTool(BaseTool):
    """Tool for performing detailed segmentation analysis of chest X-ray images."""

    name: str = "chest_xray_segmentation"
    description: str = (
        "Segments chest X-ray images for specified anatomical structures and some pathological findings. "
        "Core structures supported by the current model: Left/Right Clavicle, Left/Right Scapula, "
        "Left/Right Lung, Left/Right Hilus Pulmonis, Heart, Aorta, Facies Diaphragmatica (diaphragm), "
        "Mediastinum, Weasand (esophagus), and Spine. "
        "Segmentation for additional requested structures like Ribs, Pleural Effusion, Consolidation, or Nodule/Mass "
        "is currently STUBBED (i.e., they will be acknowledged but not segmented, pending model update). "
        "Returns a segmentation visualization (for supported structures) and comprehensive metrics. "
        "Inform the user that area calculations are approximate unless the input was a DICOM with pixel spacing data."
    )
    args_schema: Type[BaseModel] = ChestXRaySegmentationInput

    model: Any = None
    # Define which structures are supported by the current model vs stubbed
    supported_organ_map_indices: Dict[str, int] = None
    stubbed_organ_names: List[str] = None
    device: Optional[str] = "cuda"
    transform: Any = None
    pixel_spacing_mm: float = 0.2
    temp_dir: Path = Path("temp")
    organ_map: Dict[str, int] = None

    def __init__(self, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the segmentation tool with model and temporary directory."""
        super().__init__()
        self.model = xrv.baseline_models.chestx_det.PSPNet()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
        )

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

        # Map friendly names to model target indices for PSPNet model
        self.supported_organ_map_indices = {
            "Left Clavicle": 0, "Right Clavicle": 1,
            "Left Scapula": 2, "Right Scapula": 3,
            "Left Lung": 4, "Right Lung": 5,
            "Left Hilus Pulmonis": 6, "Right Hilus Pulmonis": 7,
            "Heart": 8, "Aorta": 9,
            "Facies Diaphragmatica": 10, # Diaphragm
            "Mediastinum": 11, "Weasand": 12, # Esophagus
            "Spine": 13,
        }

        # Define stubbed organs (requested but not supported by current model)
        # These won't have an index in the current model output.
        self.stubbed_organ_names = sorted([
            "Ribs", # Anatomical, but not in PSPNet's 14
            "Pleural Effusion", # Pathological
            "Consolidation", # Pathological
            "Nodule", # Pathological, often used interchangeably with Nodule/Mass
            "Mass",   # Pathological
            # Could add "Lung Lobes" here too if desired as a stub.
        ])

        # The overall organ_map for validation purposes includes all known, supported or stubbed
        self.organ_map = {**self.supported_organ_map_indices,
                            **{name: -1 for name in self.stubbed_organ_names}} # -1 indicates stub/no model index


    def _align_mask_to_original(
        self, mask: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Align a mask from the transformed (cropped/resized) space back to the full original image.
        Assumes that the transform does a center crop to a square of side = min(original height, width)
        and then resizes to (512,512).
        """
        orig_h, orig_w = original_shape
        crop_size = min(orig_h, orig_w)
        crop_top = (orig_h - crop_size) // 2
        crop_left = (orig_w - crop_size) // 2

        # Resize mask (from 512x512) to the cropped region size
        resized_mask = skimage.transform.resize(
            mask, (crop_size, crop_size), order=0, preserve_range=True, anti_aliasing=False
        )
        full_mask = np.zeros(original_shape)
        full_mask[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size] = resized_mask
        return full_mask

    def _compute_organ_metrics(
        self, mask: np.ndarray, original_img: np.ndarray, confidence: float
    ) -> Optional[OrganMetrics]:
        """Compute comprehensive metrics for a single organ mask."""
        # Align mask to the original image coordinates if needed
        if mask.shape != original_img.shape:
            mask = self._align_mask_to_original(mask, original_img.shape)

        props = skimage.measure.regionprops(mask.astype(int))
        if not props:
            return None

        props = props[0]
        area_cm2 = mask.sum() * (self.pixel_spacing_mm / 10) ** 2

        img_height, img_width = mask.shape
        cy, cx = props.centroid
        relative_pos = {
            "top": cy / img_height,
            "left": cx / img_width,
            "center_dist": np.sqrt(((cy / img_height - 0.5) ** 2 + (cx / img_width - 0.5) ** 2)),
        }

        organ_pixels = original_img[mask > 0]
        mean_intensity = organ_pixels.mean() if len(organ_pixels) > 0 else 0
        std_intensity = organ_pixels.std() if len(organ_pixels) > 0 else 0

        return OrganMetrics(
            area_pixels=int(mask.sum()),
            area_cm2=float(area_cm2),
            centroid=(float(cy), float(cx)),
            bbox=tuple(map(int, props.bbox)),
            width=int(props.bbox[3] - props.bbox[1]),
            height=int(props.bbox[2] - props.bbox[0]),
            aspect_ratio=float(
                (props.bbox[2] - props.bbox[0]) / max(1, props.bbox[3] - props.bbox[1])
            ),
            relative_position=relative_pos,
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            confidence_score=float(confidence),
        )

    def _save_visualization(
        self, original_img: np.ndarray, pred_masks: torch.Tensor, organ_indices: List[int]
    ) -> str:
        """Save visualization of original image with segmentation masks overlaid."""
        plt.figure(figsize=(10, 10))
        plt.imshow(
            original_img, cmap="gray", extent=[0, original_img.shape[1], original_img.shape[0], 0]
        )

        # Generate color palette for organs
        colors = plt.cm.rainbow(np.linspace(0, 1, len(organ_indices)))

        # Process and overlay each organ mask
        for idx, (organ_idx, color) in enumerate(zip(organ_indices, colors)):
            mask = pred_masks[0, organ_idx].cpu().numpy()
            if mask.sum() > 0:
                # Align the mask to the original image coordinates
                if mask.shape != original_img.shape:
                    mask = self._align_mask_to_original(mask, original_img.shape)

                # Create a colored overlay with transparency
                colored_mask = np.zeros((*original_img.shape, 4))
                colored_mask[mask > 0] = (*color[:3], 0.3)
                plt.imshow(
                    colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0]
                )

                # Add legend entry for the organ
                organ_name = list(self.organ_map.keys())[
                    list(self.organ_map.values()).index(organ_idx)
                ]
                plt.plot([], [], color=color, label=organ_name, linewidth=3)

        plt.title("Segmentation Overlay")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.axis("off")

        save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for specified organs."""
        requested_organs_list = organs # Keep original request for metadata
        processed_organs_names = []
        stubbed_organs_requested = []

        try:
            # Validate and categorize requested organs
            target_model_indices = [] # Indices for the PSPNet model
            organs_for_model_processing = [] # Names corresponding to target_model_indices

            if not organs: # If None, segment all supported structures
                organs_for_model_processing = list(self.supported_organ_map_indices.keys())
                target_model_indices = list(self.supported_organ_map_indices.values())
                if not requested_organs_list: # If organs was initially None
                    requested_organs_list = list(self.organ_map.keys()) # For metadata, show all known
            else:
                organs = [o.strip() for o in organs]
                invalid_organs = [o for o in organs if o not in self.organ_map]
                if invalid_organs:
                    raise ValueError(f"Invalid or unknown organs specified: {invalid_organs}. Valid options include {list(self.organ_map.keys())}.")

                for organ_name in organs:
                    if organ_name in self.supported_organ_map_indices:
                        organs_for_model_processing.append(organ_name)
                        target_model_indices.append(self.supported_organ_map_indices[organ_name])
                    elif organ_name in self.stubbed_organ_names:
                        stubbed_organs_requested.append(organ_name)

            if not organs_for_model_processing and not stubbed_organs_requested:
                 # This case should ideally be caught by invalid_organs check if organs list was provided but empty of known items
                raise ValueError("No valid organs specified for segmentation or all specified organs are currently stubbed and none are processable by the model.")

            # Load and process image only if there are model-processable organs
            if organs_for_model_processing:
                original_img = skimage.io.imread(image_path)
                if len(original_img.shape) > 2:
                    original_img = original_img[:, :, 0]

                img = xrv.datasets.normalize(original_img, 255)
                img = img[None, ...]
                img = self.transform(img)
                img = torch.from_numpy(img)
                img = img.to(self.device)

                # Generate predictions
                with torch.no_grad():
                    pred = self.model(img)
                pred_probs = torch.sigmoid(pred) # Sigmoid for multi-label segmentation
                pred_masks = (pred_probs > 0.5).float()

                # Save visualization
                viz_path = self._save_visualization(original_img, pred_masks, target_model_indices)

                # Compute metrics for selected organs that were processed by the model
                results = {}
                for organ_name, model_idx in zip(organs_for_model_processing, target_model_indices):
                    mask = pred_masks[0, model_idx].cpu().numpy()
                    if mask.sum() > 0:
                        metrics = self._compute_organ_metrics(
                            mask, original_img, float(pred_probs[0, model_idx].mean().cpu())
                        )
                        if metrics:
                            results[organ_name] = metrics
                            processed_organs_names.append(organ_name)

                output_segmentation_path = viz_path
                output_metrics = {organ: metrics.dict() for organ, metrics in results.items()}

                # Attempt to calculate Cardiothoracic Ratio (CTR)
                # CTR = Max Heart Width / Max Thoracic Width
                heart_metrics = results.get("Heart")
                left_lung_metrics = results.get("Left Lung")
                right_lung_metrics = results.get("Right Lung")
                ctr = None
                ctr_calculation_note = "CTR calculation requires successful segmentation of Heart, Left Lung, and Right Lung."

                if heart_metrics and left_lung_metrics and right_lung_metrics:
                    heart_width_px = heart_metrics.width # width is bbox[3] - bbox[1] (max_x - min_x)

                    # Approximate thoracic width using outer lung boundaries
                    # Assumes standard patient orientation (Right lung is to the image right of Left lung)
                    # bbox = (min_y, min_x, max_y, max_x)
                    thoracic_width_px = right_lung_metrics.bbox[3] - left_lung_metrics.bbox[1]

                    if thoracic_width_px > 0:
                        ctr = heart_width_px / thoracic_width_px
                        output_metrics["CardiothoracicRatio"] = {
                            "value": round(ctr, 3) if ctr is not None else None,
                            "heart_width_pixels": heart_width_px,
                            "thoracic_width_pixels_approx": thoracic_width_px,
                            "note": "Thoracic width is approximated from outer lung boundaries. For precise CTR, ensure accurate pixel_spacing and consider inner rib cage for thoracic width."
                        }
                        ctr_calculation_note = "CTR calculated."
                        if self.pixel_spacing_mm == 0.2: # Default value
                             output_metrics["CardiothoracicRatio"]["warning"] = "CTR calculated using default pixel spacing. For accuracy, ensure pixel_spacing_mm is set from DICOM."
                    else:
                        ctr_calculation_note = "Could not calculate CTR: thoracic width approximation was zero or invalid."
                output_metrics["CTR_Calculation_Status"] = ctr_calculation_note


            else: # No organs to process with the model, only stubbed ones requested
                output_segmentation_path = None
                output_metrics = {}
                original_img_shape = None # Not loaded
                model_img_shape = None # Not processed
                # Try to get original image shape for metadata if possible without full load
                try:
                    img_info = skimage.io.info(image_path)
                    original_img_shape = img_info.shape
                except:
                    original_img_shape = "Unknown (image not loaded for stubbed request)"


            output = {
                "segmentation_image_path": output_segmentation_path,
                "metrics": output_metrics,
            }
            if stubbed_organs_requested:
                output["note_stubbed_organs"] = (
                    f"The following requested structures are currently stubbed and were not segmented: "
                    f"{', '.join(stubbed_organs_requested)}. Model update is required for these."
                )

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": output_segmentation_path,
                "original_size": original_img.shape if 'original_img' in locals() else original_img_shape,
                "model_size": tuple(img.shape[-2:]) if 'img' in locals() else model_img_shape,
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "requested_organs_input": requested_organs_list if requested_organs_list else "All supported",
                "processed_model_organs": organs_for_model_processing, # organs actually sent to model
                "organs_with_metrics": processed_organs_names, # organs for which metrics were computed
                "stubbed_organs_requested": stubbed_organs_requested,
                "analysis_status": "completed",
            }
            if not organs_for_model_processing and stubbed_organs_requested:
                 metadata["analysis_status"] = "completed_only_stubbed"


            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "requested_organs_input": requested_organs_list if requested_organs_list else "All supported",
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, organs)
            if len(original_img.shape) > 2:
                original_img = original_img[:, :, 0]

            img = xrv.datasets.normalize(original_img, 255)
            img = img[None, ...]
            img = self.transform(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)

            # Generate predictions
            with torch.no_grad():
                pred = self.model(img)
            pred_probs = torch.sigmoid(pred)
            pred_masks = (pred_probs > 0.5).float()

            # Save visualization
            viz_path = self._save_visualization(original_img, pred_masks, organ_indices)

            # Compute metrics for selected organs
            results = {}
            for idx, organ_name in zip(organ_indices, organs):
                mask = pred_masks[0, idx].cpu().numpy()
                if mask.sum() > 0:
                    metrics = self._compute_organ_metrics(
                        mask, original_img, float(pred_probs[0, idx].mean().cpu())
                    )
                    if metrics:
                        results[organ_name] = metrics

            output = {
                "segmentation_image_path": viz_path,
                "metrics": {organ: metrics.dict() for organ, metrics in results.items()},
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "original_size": original_img.shape,
                "model_size": tuple(img.shape[-2:]),
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "requested_organs": organs,
                "processed_organs": list(results.keys()),
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, organs)
