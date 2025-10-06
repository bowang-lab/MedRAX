"""
Advanced Tool Manager for MedRAX Web Platform
Handles dynamic loading, unloading, and management of all MedRAX tools
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add MedRAX to path
medrax_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(medrax_root))

# Import logger
from logger_config import get_logger

logger = get_logger(__name__)


class ToolStatus(Enum):
    """Tool status states"""
    AVAILABLE = "available"  # Can be loaded
    LOADED = "loaded"  # Currently loaded and ready
    UNAVAILABLE = "unavailable"  # Dependencies missing
    ERROR = "error"  # Failed to load


@dataclass
class ToolInfo:
    """Information about a MedRAX tool"""
    name: str
    display_name: str
    description: str
    category: str
    status: ToolStatus
    requires_model: bool
    model_size_gb: float
    mac_compatible: bool
    dependencies: List[str]
    error_message: Optional[str] = None
    instance: Optional[Any] = None
    is_cached: bool = False  # Whether model is already downloaded


class ToolManager:
    """Manages dynamic loading and unloading of MedRAX tools"""
    
    # Tool definitions with metadata
    TOOL_DEFINITIONS = {
        "image_visualizer": {
            "display_name": "Image Visualizer",
            "description": "Display and visualize medical images",
            "category": "utility",
            "requires_model": False,
            "model_size_gb": 0.0,
            "mac_compatible": True,
            "dependencies": ["PIL"],
            "class_name": "ImageVisualizerTool",
        },
        "chest_xray_classifier": {
            "display_name": "Pathology Classifier",
            "description": "Classify 18 chest pathologies using DenseNet-121",
            "category": "analysis",
            "requires_model": True,
            "model_size_gb": 0.5,
            "mac_compatible": True,
            "dependencies": ["torch", "torchxrayvision"],
            "class_name": "ChestXRayClassifierTool",
        },
        "chest_xray_segmentation": {
            "display_name": "Anatomical Segmentation",
            "description": "Segment anatomical structures (lungs, heart, etc.)",
            "category": "analysis",
            "requires_model": True,
            "model_size_gb": 0.3,
            "mac_compatible": True,
            "dependencies": ["torch", "torchxrayvision"],
            "class_name": "ChestXRaySegmentationTool",
        },
        "chest_xray_report": {
            "display_name": "Report Generator",
            "description": "Generate radiology reports using ViT-BERT",
            "category": "analysis",
            "requires_model": True,
            "model_size_gb": 1.2,
            "mac_compatible": True,
            "dependencies": ["torch", "transformers"],
            "class_name": "ChestXRayReportGeneratorTool",
            "model_id": "StanfordAIMI/interpret-cxr-vit-bert-base",
        },
        "chest_xray_expert": {
            "display_name": "CheXagent VQA",
            "description": "Expert visual question answering using CheXagent-2-3b",
            "category": "expert",
            "requires_model": True,
            "model_size_gb": 6.0,
            "mac_compatible": True,
            "dependencies": ["torch", "transformers"],
            "class_name": "XRayVQATool",
            "model_id": "StanfordAIMI/CheXagent-8b",
        },
        "xray_phrase_grounding": {
            "display_name": "Phrase Grounding (Maira-2)",
            "description": "Localize medical findings using bounding boxes (requires HuggingFace login)",
            "category": "expert",
            "requires_model": True,
            "model_size_gb": 10.0,
            "mac_compatible": True,
            "dependencies": ["torch", "transformers"],
            "class_name": "XRayPhraseGroundingTool",
            "model_id": "microsoft/maira-2",
        },
        "dicom_processor": {
            "display_name": "DICOM Processor",
            "description": "Convert and process DICOM medical images",
            "category": "utility",
            "requires_model": False,
            "model_size_gb": 0.0,
            "mac_compatible": False,  # gdcm issues on ARM64 Mac
            "dependencies": ["pydicom", "gdcm"],
            "class_name": "DicomProcessorTool",
        },
        "llava_med": {
            "display_name": "LLaVA-Med",
            "description": "Medical multimodal reasoning",
            "category": "expert",
            "requires_model": True,
            "model_size_gb": 13.0,
            "mac_compatible": True,
            "dependencies": ["torch", "transformers"],
            "class_name": "LlavaMedTool",
            "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        },
        "chest_xray_generator": {
            "display_name": "X-Ray Generator (RoentGen)",
            "description": "Generate synthetic chest X-rays from text",
            "category": "generation",
            "requires_model": True,
            "model_size_gb": 5.0,
            "mac_compatible": True,
            "dependencies": ["torch", "diffusers"],
            "class_name": "ChestXRayGeneratorTool",
            "model_id": "BIRTLAB/roentgen-v2",
        },
        "medical_knowledge_rag": {
            "display_name": "Medical Knowledge RAG",
            "description": "Answer questions using medical knowledge base",
            "category": "knowledge",
            "requires_model": False,
            "model_size_gb": 0.0,
            "mac_compatible": True,
            "dependencies": ["cohere", "langchain_cohere", "langchain_chroma", "langchain_community", "datasets"],
            "class_name": "RAGTool",
        },
    }
    
    def __init__(self, device: str = "cpu", model_dir: str = "/model-weights", temp_dir: str = "temp"):
        """Initialize the tool manager"""
        self.device = device
        self.model_dir = model_dir
        self.temp_dir = temp_dir
        self.tools: Dict[str, ToolInfo] = {}
        self.is_mac = platform.system() == "Darwin"
        self.is_arm64 = platform.machine() == "arm64"
        
        # Initialize tool registry
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the tool registry with availability checks"""
        for tool_id, definition in self.TOOL_DEFINITIONS.items():
            # Check if tool is compatible with current platform
            if not definition["mac_compatible"] and self.is_mac:
                status = ToolStatus.UNAVAILABLE
                error = f"Not compatible with macOS {platform.machine()}"
            else:
                # Check dependencies
                deps_available = self._check_dependencies(definition["dependencies"])
                if deps_available:
                    status = ToolStatus.AVAILABLE
                    error = None
                else:
                    status = ToolStatus.UNAVAILABLE
                    error = f"Missing dependencies: {', '.join(definition['dependencies'])}"
            
            # Check if model is cached
            is_cached = False
            if definition["requires_model"] and "model_id" in definition:
                is_cached = self._is_model_cached(definition["model_id"])
            
            self.tools[tool_id] = ToolInfo(
                name=tool_id,
                display_name=definition["display_name"],
                description=definition["description"],
                category=definition["category"],
                status=status,
                requires_model=definition["requires_model"],
                model_size_gb=definition["model_size_gb"],
                mac_compatible=definition["mac_compatible"],
                dependencies=definition["dependencies"],
                error_message=error,
                instance=None,
                is_cached=is_cached
            )
    
    def _is_model_cached(self, model_id: str) -> bool:
        """Check if a HuggingFace model is already cached"""
        try:
            from pathlib import Path
            import os
            
            # Get HuggingFace cache directory
            cache_home = os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME")
            if cache_home:
                cache_dir = Path(cache_home) / "hub"
            else:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Model snapshot directory format: models--{org}--{model_name}
            model_dir_name = f"models--{model_id.replace('/', '--')}"
            model_path = cache_dir / model_dir_name
            
            # Check if directory exists and has snapshots
            if model_path.exists():
                snapshots_dir = model_path / "snapshots"
                if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                    logger.debug("model_cached", model_id=model_id, path=str(model_path))
                    return True
            
            return False
        except Exception as e:
            logger.warning("cache_check_error", model_id=model_id, error=str(e))
            return False
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are available"""
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tools"""
        return {
            tool_id: {
                "name": tool.name,
                "display_name": tool.display_name,
                "description": tool.description,
                "category": tool.category,
                "status": tool.status.value,
                "requires_model": tool.requires_model,
                "model_size_gb": tool.model_size_gb,
                "mac_compatible": tool.mac_compatible,
                "dependencies": tool.dependencies,
                "error_message": tool.error_message,
                "is_loaded": tool.instance is not None,
                "is_cached": tool.is_cached
            }
            for tool_id, tool in self.tools.items()
        }
    
    def load_tool(self, tool_id: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Load a specific tool
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        if tool_id not in self.tools:
            return False, f"Tool '{tool_id}' not found"
        
        tool_info = self.tools[tool_id]
        
        if tool_info.status == ToolStatus.UNAVAILABLE:
            return False, tool_info.error_message or "Tool unavailable"
        
        if tool_info.instance is not None:
            return True, None  # Already loaded
        
        try:
            # Import the specific tool class (avoid importing all tools)
            tool_class = None
            
            if tool_id == "image_visualizer":
                from medrax.tools.utils import ImageVisualizerTool
                tool_class = ImageVisualizerTool
            elif tool_id == "chest_xray_classifier":
                from medrax.tools.classification import ChestXRayClassifierTool
                tool_class = ChestXRayClassifierTool
            elif tool_id == "chest_xray_segmentation":
                from medrax.tools.segmentation import ChestXRaySegmentationTool
                tool_class = ChestXRaySegmentationTool
            elif tool_id == "chest_xray_report":
                from medrax.tools.report_generation import ChestXRayReportGeneratorTool
                tool_class = ChestXRayReportGeneratorTool
            elif tool_id == "chest_xray_expert":
                from medrax.tools.xray_vqa import XRayVQATool
                tool_class = XRayVQATool
            elif tool_id == "xray_phrase_grounding":
                from medrax.tools.grounding import XRayPhraseGroundingTool
                tool_class = XRayPhraseGroundingTool
            elif tool_id == "dicom_processor":
                from medrax.tools.dicom import DicomProcessorTool
                tool_class = DicomProcessorTool
            elif tool_id == "llava_med":
                from medrax.tools.llava_med import LlavaMedTool
                tool_class = LlavaMedTool
            elif tool_id == "chest_xray_generator":
                from medrax.tools.generation import ChestXRayGeneratorTool
                tool_class = ChestXRayGeneratorTool
            elif tool_id == "medical_knowledge_rag":
                from medrax.tools.rag import RAGTool
                tool_class = RAGTool
            
            if not tool_class:
                return False, f"Tool class not found for '{tool_id}'"
            
            # Initialize tool with appropriate parameters
            if tool_id in ["image_visualizer"]:
                instance = tool_class()
            elif tool_id in ["dicom_processor"]:
                instance = tool_class(temp_dir=self.temp_dir)
            elif tool_id in ["chest_xray_classifier", "chest_xray_segmentation"]:
                instance = tool_class(device=self.device)
            elif tool_id == "chest_xray_report":
                # Report generator needs cache_dir for model downloads
                instance = tool_class(device=self.device, cache_dir=self.model_dir)
            elif tool_id == "chest_xray_expert":
                instance = tool_class(device=self.device, cache_dir=self.model_dir)
            elif tool_id == "xray_phrase_grounding":
                # Maira-2 requires HuggingFace authentication for gated repo
                instance = tool_class(
                    cache_dir=self.model_dir,
                    temp_dir=self.temp_dir,
                    load_in_8bit=False,  # Quantization doesn't work on Mac
                    device=self.device
                )
            elif tool_id == "llava_med":
                # Don't use quantization on Mac - bitsandbytes doesn't work
                instance = tool_class(
                    cache_dir=self.model_dir,
                    device=self.device,
                    load_in_8bit=False  # Disable on Mac
                )
            elif tool_id == "chest_xray_generator":
                # Use HuggingFace repo instead of local path
                instance = tool_class(
                    model_path="BIRTLAB/roentgen-v2",  # HuggingFace repo
                    cache_dir=self.model_dir,
                    temp_dir=self.temp_dir,
                    device=self.device
                )
            elif tool_id == "medical_knowledge_rag":
                # RAG requires special configuration
                return False, "RAG tool requires configuration (Cohere API key, etc.)"
            else:
                instance = tool_class(**kwargs)
            
            tool_info.instance = instance
            tool_info.status = ToolStatus.LOADED
            tool_info.is_cached = True  # Model is now cached after loading
            return True, None
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load tool: {str(e)}\n{traceback.format_exc()}"
            tool_info.status = ToolStatus.ERROR
            tool_info.error_message = error_msg
            return False, error_msg
    
    def unload_tool(self, tool_id: str) -> Tuple[bool, Optional[str]]:
        """
        Unload a specific tool to free memory
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        if tool_id not in self.tools:
            return False, f"Tool '{tool_id}' not found"
        
        tool_info = self.tools[tool_id]
        
        if tool_info.instance is None:
            return True, None  # Already unloaded
        
        try:
            # Clear the instance
            del tool_info.instance
            tool_info.instance = None
            tool_info.status = ToolStatus.AVAILABLE
            
            # Force garbage collection for large models
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            return True, None
            
        except Exception as e:
            return False, f"Failed to unload tool: {str(e)}"
    
    def get_loaded_tools(self) -> List[Any]:
        """Get list of loaded tool instances for agent"""
        return [
            tool_info.instance
            for tool_info in self.tools.values()
            if tool_info.instance is not None
        ]
    
    def get_loaded_tools_dict(self) -> Dict[str, Any]:
        """Get dictionary of loaded tool instances"""
        return {
            tool_info.instance.name: tool_info.instance
            for tool_info in self.tools.values()
            if tool_info.instance is not None
        }
    
    def load_default_tools(self) -> Dict[str, bool]:
        """Load default recommended tools"""
        default_tools = [
            "image_visualizer",
            "chest_xray_classifier",
            "chest_xray_segmentation",
            "chest_xray_report",
            "chest_xray_expert",
        ]
        
        results = {}
        for tool_id in default_tools:
            success, error = self.load_tool(tool_id)
            results[tool_id] = success
            if success:
                logger.info("success", message=f"Loaded {tool_id}")
            else:
                logger.error("error", message=f"Failed to load {tool_id}: {error}")
        
        return results
    
    def get_tool_recommendations(self) -> Dict[str, List[str]]:
        """Get tool recommendations based on system capabilities"""
        recommendations = {
            "essential": ["image_visualizer", "chest_xray_classifier", "chest_xray_report"],
            "recommended": ["chest_xray_segmentation", "chest_xray_expert"],
            "advanced": [],
            "unavailable": []
        }
        
        for tool_id, tool_info in self.tools.items():
            if tool_info.status == ToolStatus.UNAVAILABLE:
                recommendations["unavailable"].append(tool_id)
            elif tool_info.model_size_gb > 5.0:
                recommendations["advanced"].append(tool_id)
        
        return recommendations

