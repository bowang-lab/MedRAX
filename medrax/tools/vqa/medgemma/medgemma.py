import asyncio
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple
import uuid

from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
import torch
import transformers
from transformers import BitsAndBytesConfig, pipeline
import uvicorn

#TODO: delete this
print("ENVIRONMENT CHECK")
print(f"Python Executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Configuration
UPLOAD_DIR = "./medgemma_images"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic Models for API
class VQAInput(BaseModel):
    """Input schema for the MedGemma VQA API endpoint.
    
    Defines the structure for requests to the /analyze-images/ endpoint.
    Used for validating incoming API requests and generating OpenAPI documentation.
    """
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )

class VQAResponse(BaseModel):
    """Response schema for successful MedGemma VQA API requests.
    
    Defines the structure of successful responses from the /analyze-images/ endpoint.
    Used for response validation and OpenAPI documentation.
    """
    response: str = Field(..., description="Generated medical analysis response from MedGemma model")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the analysis request and results")

class ErrorResponse(BaseModel):
    """Error response schema for failed MedGemma VQA API requests.
    
    Defines the structure of error responses from the /analyze-images/ endpoint.
    Used for error response validation and OpenAPI documentation.
    """
    error: str = Field(..., description="Human-readable error message describing what went wrong")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the error and request context")

# MedGemma Model Handling
class MedGemmaModel:
    """Medical visual question answering model using Google's MedGemma 4B model.

    MedGemma is a specialized multimodal AI model trained on medical images and text.
    It provides expert-level analysis for chest X-rays, dermatology images,
    ophthalmology images, and histopathology slides.

    Key capabilities:
    - Medical image classification and analysis across multiple modalities
    - Visual question answering for radiology, dermatology, pathology, ophthalmology
    - Clinical reasoning and medical knowledge integration
    - Multi-modal medical understanding (text + images)
    - Support for up to 128K context length

    Performance:
    - Full precision (bfloat16): ~8GB VRAM, recommended for medical applications
    - 4-bit quantization (default): Available but may affect quality on some systems

    This class implements a singleton pattern to ensure only one model instance
    is loaded in memory, optimizing resource usage for the FastAPI service.
    """
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance of MedGemmaModel.
        
        Ensures only one model instance exists in memory, preventing
        multiple model loads and conserving GPU memory.
        
        Returns:
            MedGemmaModel: The singleton instance
        """
        if not cls._instance:
            cls._instance = super(MedGemmaModel, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedGemmaModel.

        Args:
            model_name: Name of the MedGemma model to use (default: "google/medgemma-4b-it")
            device: Device to run model on - "cuda" or "cpu" (default: "cuda")
            dtype: Data type for model weights - bfloat16 recommended for efficiency (default: torch.bfloat16)
            cache_dir: Directory to cache downloaded models (default: None)
            load_in_4bit: Whether to load model in 4-bit quantization for memory efficiency (default: True)
            **kwargs: Additional arguments passed to the model pipeline

        Raises:
            RuntimeError: If model initialization fails (e.g., insufficient GPU memory)
        """
        # Re-initialization guard
        if hasattr(self, 'pipe') and self.pipe is not None:
            return

        self.device = device if device and torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Setup model configuration
        model_kwargs = {
            "torch_dtype": self.dtype,
        }

        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        # Handle device mapping and quantization
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
            "use_cache": True,
        }

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = {"": self.device}

        try:
            self.pipe = pipeline("image-text-to-text", **pipeline_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MedGemma pipeline: {str(e)}")

    def _prepare_messages(
        self, image_paths: List[str], prompt: str, system_prompt: str
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Prepare chat messages in the format expected by MedGemma.

        Converts image paths to PIL Image objects and formats them into the
        chat message structure that MedGemma expects for multimodal input.

        Args:
            image_paths: List of file paths to medical images
            prompt: User's question or instruction about the images
            system_prompt: System context message to set the model's role

        Returns:
            Tuple containing:
                - List of formatted chat messages for MedGemma
                - List of loaded PIL Image objects

        Raises:
            FileNotFoundError: If any image file cannot be found
        """
        images = []
        for path in image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Image file not found: {path}")

            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        # Create messages in chat format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": img} for img in images],
            },
        ]

        return messages, images

    def _generate_response(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """Generate response using MedGemma pipeline.

        Processes the formatted messages through the MedGemma model to generate
        a medical analysis response.

        Args:
            messages: Formatted chat messages with images and text
            max_new_tokens: Maximum number of tokens to generate in response

        Returns:
            Generated response text from MedGemma model
        """
        # Generate using pipeline
        output = self.pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Extract generated text from pipeline output
        if (
            isinstance(output, list)
            and output
            and isinstance(output[0].get("generated_text"), list)
        ):
            generated_text = output[0]["generated_text"]
            if generated_text:
                return generated_text[-1].get("content", "").strip()

        return "No response generated"

    def _create_error_response(
        self,
        image_paths: List[str],
        prompt: str,
        error_message: str,
        error_type: str,
        error_details: str,
    ) -> Dict[str, Any]:
        """Create standardized error response metadata.

        Generates consistent error metadata structure for logging and debugging
        purposes across different error scenarios.

        Args:
            image_paths: List of image paths that were being processed
            prompt: User prompt that was being processed
            error_message: Human-readable error message
            error_type: Categorization of the error (e.g., "memory_error", "file_not_found")
            error_details: Detailed technical error information

        Returns:
            Dictionary containing standardized error metadata
        """
        return {
            "image_paths": image_paths,
            "prompt": prompt,
            "analysis_status": "failed",
            "error_type": error_type,
            "error_details": error_details,
        }

    async def aget_response(self, image_paths: List[str], prompt: str, system_prompt: str, max_new_tokens: int) -> str:
        """Async method to get response from MedGemma model.

        Main entry point for generating medical analysis responses. Handles
        the complete pipeline from image loading to response generation
        in an asynchronous manner.

        Args:
            image_paths: List of file paths to medical images
            prompt: User's question or instruction about the images
            system_prompt: System context message to set the model's role
            max_new_tokens: Maximum number of tokens to generate in response

        Returns:
            Generated medical analysis response as a string

        Raises:
            FileNotFoundError: If any image file cannot be found
            RuntimeError: If model inference fails
        """
        loop = asyncio.get_event_loop()
        messages, _ = await loop.run_in_executor(None, self._prepare_messages, image_paths, prompt, system_prompt)

        def _generate():
            return self._generate_response(messages, max_new_tokens)

        return await loop.run_in_executor(None, _generate)

# FastAPI Application
app = FastAPI(
    title="MedGemma VQA API",
    description="API for medical visual question answering using Google's MedGemma model."
)

medgemma_model: Optional[MedGemmaModel] = None

@app.on_event("startup")
async def startup_event():
    """Load the MedGemma model at application startup.
    
    This function is called when the FastAPI application starts up.
    It initializes the MedGemma model as a global singleton instance,
    ensuring the model is loaded and ready to handle requests.
    
    The model is loaded with default settings optimized for medical
    image analysis, including 4-bit quantization for memory efficiency.
    
    Raises:
        SystemExit: If model loading fails, the application will exit
                   to prevent serving requests with an unavailable model.
    """
    global medgemma_model
    try:
        medgemma_model = MedGemmaModel()
        print("MedGemma model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading MedGemma model: {e}")
        exit(1)

@app.post("/analyze-images/",
            response_model=VQAResponse,
            responses={
                500: {"model": ErrorResponse, "description": "Internal server error or model inference failure"},
                404: {"model": ErrorResponse, "description": "Image file not found"},
                400: {"description": "Invalid request format or unsupported image type"},
                503: {"description": "Model not available or not loaded"}
            },
            summary="Analyze one or more medical images",
            description="Upload medical images and receive AI-powered analysis using Google's MedGemma model.")
async def analyze_images(
    images: List[UploadFile] = File(..., description="List of medical image files to analyze (JPG or PNG)."),
    prompt: str = Form(..., description="Question or instruction about the medical images."),
    system_prompt: Optional[str] = Form("You are an expert radiologist.", description="System prompt to set the context for the model."),
    max_new_tokens: int = Form(100, description="Maximum number of tokens to generate in the response.")
):
    """Analyze medical images using MedGemma AI model.
    
    This endpoint accepts one or more medical images along with a prompt
    and returns AI-generated medical analysis.
    
    The endpoint handles the complete pipeline:
    1. Validates uploaded image files
    2. Saves images temporarily to disk
    3. Processes images through MedGemma model
    4. Returns structured analysis with metadata
    5. Cleans up temporary files
    
    Args:
        images: List of uploaded image files (JPG/PNG format)
        prompt: Medical question or instruction about the images
        system_prompt: Context setting for the AI model (default: radiologist role)
        max_new_tokens: Maximum response length (default: 100)
    
    Returns:
        VQAResponse: Contains the AI-generated analysis and request metadata
        
    Raises:
        HTTPException 400: Invalid image format or request structure
        HTTPException 404: Image file not found during processing
        HTTPException 500: Model inference error or memory issues
        HTTPException 503: Model not available for processing
    """
    # Check if model is available
    if medgemma_model is None or medgemma_model.pipe is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    # Process uploaded images
    image_paths = []
    for image in images:
        # Validate image format
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {image.content_type}. Only JPG and PNG are supported.")
        
        # Generate unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{image.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        try:
            # Save uploaded image to disk
            with open(file_path, "wb") as buffer:
                buffer.write(await image.read())
            image_paths.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")

    try:
        # Generate AI analysis
        response_text = await medgemma_model.aget_response(image_paths, prompt, system_prompt, max_new_tokens)
        
        # Prepare success response
        metadata = {
            "image_paths": image_paths,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "num_images": len(image_paths),
            "analysis_status": "completed",
        }
        return VQAResponse(response=response_text, metadata=metadata)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Image file not found: {str(e)}")
    except torch.cuda.OutOfMemoryError as e:
        error_message = "GPU memory exhausted. Try reducing image resolution or max_new_tokens."
        metadata = medgemma_model._create_error_response(
            image_paths, prompt, error_message, "memory_error", str(e)
        )
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        traceback.print_exc()
        metadata = medgemma_model._create_error_response(
            image_paths, prompt, f"Analysis failed: {str(e)}", "general_error", str(e)
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary image files
        for path in image_paths:
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == "__main__":
    """Launch the MedGemma VQA API server.
    
    Starts the FastAPI application with uvicorn server, binding to all
    network interfaces on port 8002.
    """
    uvicorn.run(app, host="0.0.0.0", port=8002)