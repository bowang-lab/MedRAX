from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
import asyncio
import uvicorn
import os
import uuid
import traceback
import sys
import transformers

print("--- ENVIRONMENT CHECK ---")
print(f"Python Executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print("-----------------------")

# --- Configuration ---
CACHE_DIR = "./model_cache"
UPLOAD_DIR = "./uploaded_images"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Pydantic Models for API ---
class VQAInput(BaseModel):
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )

class VQAResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    metadata: Dict[str, Any]

# --- MedGemma Model Handling ---
class MedGemmaModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MedGemmaModel, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 model_name: str = "google/medgemma-4b-it",
                 device: Optional[str] = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 load_in_4bit: bool = False):
        if hasattr(self, 'pipe') and self.pipe is not None:
            return

        self.device = device if device and torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.pipe = None

        model_kwargs = {"torch_dtype": self.dtype, "cache_dir": CACHE_DIR}

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = {"": self.device}

        try:
            self.pipe = pipeline("image-text-to-text",
                                 model=model_name,
                                 model_kwargs=model_kwargs,
                                 trust_remote_code=True,
                                 use_cache=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MedGemma pipeline: {str(e)}")

    def _prepare_messages(
        self, image_paths: List[str], prompt: str, system_prompt: str
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        images = []
        for path in image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Image file not found: {path}")

            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": img} for img in images],
            },
        ]

        return messages, images

    async def aget_response(self, image_paths: List[str], prompt: str, system_prompt: str, max_new_tokens: int) -> str:
        loop = asyncio.get_event_loop()
        messages, _ = await loop.run_in_executor(None, self._prepare_messages, image_paths, prompt, system_prompt)

        def _generate():
            return self.pipe(
                text=messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        output = await loop.run_in_executor(None, _generate)

        if (
            isinstance(output, list)
            and output
            and isinstance(output[0].get("generated_text"), list)
        ):
            generated_text = output[0]["generated_text"]
            if generated_text:
                return generated_text[-1].get("content", "").strip()

        return "No response generated"

# --- FastAPI Application ---
app = FastAPI(title="MedGemma VQA API",
              description="API for medical visual question answering using Google's MedGemma model.")

medgemma_model: Optional[MedGemmaModel] = None

@app.on_event("startup")
async def startup_event():
    """Load the MedGemma model at application startup."""
    global medgemma_model
    try:
        medgemma_model = MedGemmaModel()
        print("MedGemma model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading MedGemma model: {e}")
        # Depending on the desired behavior, you might want to exit the application
        # if the model fails to load.
        # exit(1)

@app.post("/analyze-images/",
            response_model=VQAResponse,
            responses={500: {"model": ErrorResponse},
                       404: {"model": ErrorResponse}},
            summary="Analyze one or more medical images")
async def analyze_images(
    images: List[UploadFile] = File(..., description="List of medical image files to analyze (JPG or PNG)."),
    prompt: str = Form(..., description="Question or instruction about the medical images."),
    system_prompt: Optional[str] = Form("You are an expert radiologist.", description="System prompt to set the context for the model."),
    max_new_tokens: int = Form(100, description="Maximum number of tokens to generate in the response.")
):
    """
    Upload one or more medical images and a prompt to get an analysis from the MedGemma model.
    """
    if medgemma_model is None or medgemma_model.pipe is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    image_paths = []
    for image in images:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {image.content_type}. Only JPG and PNG are supported.")
        
        # Generate a unique filename to avoid overwrites
        unique_filename = f"{uuid.uuid4()}_{image.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await image.read())
            image_paths.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")


    try:
        response_text = await medgemma_model.aget_response(image_paths, prompt, system_prompt, max_new_tokens)
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
    except Exception as e:
        print("--- AN EXCEPTION OCCURRED IN THE ENDPOINT ---")
        traceback.print_exc()
        # Catch potential CUDA out-of-memory errors and other exceptions
        error_message = "An unexpected error occurred during analysis."
        if "CUDA out of memory" in str(e):
            error_message = "GPU memory exhausted. Try reducing image resolution or max_new_tokens."
        
        metadata = {
            "image_paths": image_paths,
            "prompt": prompt,
            "analysis_status": "failed",
            "error_details": str(e),
        }
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        # Clean up saved images
        for path in image_paths:
            try:
                os.remove(path)
            except OSError:
                # Log this error if needed, but don't let it crash the request
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)