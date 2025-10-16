import os
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MedGemmaVQAInput(BaseModel):
    """Input schema for the MedGemma VQA Tool. Only supports JPG or PNG images."""
    image_paths: List[str] = Field(
        ...,
        description="List of paths to medical image files to analyze, only supports JPG or PNG images",
    )
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist who is able to analyze radiological images at any resolution.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )

class MedGemmaAPIClientTool(BaseTool):
    """Medical visual question answering tool using Google's MedGemma 4B model via API.

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
    """

    name: str = "medgemma_medical_vqa"
    description: str = (
        "Advanced medical visual question answering tool using Google's MedGemma 4B instruction-tuned model via API. "
        "Specialized for comprehensive medical image analysis across multiple modalities including chest X-rays, "
        "dermatology images, ophthalmology images, and histopathology slides. Provides expert-level medical "
        "reasoning, diagnosis assistance, and detailed image interpretation with radiologist-level expertise. "
        "Input: List of medical image paths and medical question/prompt with optional custom system prompt. "
        "Output: Comprehensive medical analysis and answers based on visual content with detailed reasoning. "
        "Supports multi-image analysis, comparative studies, and complex medical reasoning tasks. "
        "Model handles images up to 896x896 resolution and supports context up to 128K tokens."
    )
    args_schema: Type[BaseModel] = MedGemmaVQAInput
    return_direct: bool = True

    # API configuration
    api_url: str  # The URL of the running FastAPI service

    def __init__(self, api_url: str, **kwargs: Any):
        """Initialize the MedGemmaAPIClientTool.

        Args:
            api_url: The URL of the running MedGemma FastAPI service
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(api_url=api_url, **kwargs)

    def _prepare_request_data(
        self, image_paths: List[str], prompt: str, system_prompt: str, max_new_tokens: int
    ) -> Tuple[List, Dict]:
        """Prepare multipart form data for API request.

        Args:
            image_paths: List of paths to medical images
            prompt: Question or instruction about the images
            system_prompt: System context for the model
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Tuple of files list and data dictionary
        """
        files_to_send = []
        opened_files = []
        
        for path in image_paths:
            # Detect correct MIME type based on file extension
            from pathlib import Path
            ext = Path(path).suffix.lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            
            with open(path, "rb") as f:
                files_to_send.append(("images", (os.path.basename(path), f.read(), mime_type)))

        data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
        }
        
        return files_to_send, data, opened_files

    def _create_error_response(
        self,
        image_paths: List[str],
        prompt: str,
        error_message: str,
        error_type: str,
        error_details: str,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Create standardized error response.

        Args:
            image_paths: List of image paths
            prompt: User prompt
            error_message: Human-readable error message
            error_type: Type of error
            error_details: Detailed error information

        Returns:
            Tuple of error output and metadata
        """
        output = {"error": error_message}
        metadata = {
            "image_paths": image_paths,
            "prompt": prompt,
            "analysis_status": "failed",
            "error_type": error_type,
            "error_details": error_details,
        }
        return output, metadata

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist who is able to analyze radiological images at any resolution.",
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute medical visual question answering via API.

        Args:
            image_paths: List of paths to medical images
            prompt: Question or instruction about the images
            system_prompt: System context for the model
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple of output dictionary and metadata
        """
        # httpx is a modern HTTP client that supports sync and async
        timeout_config = httpx.Timeout(300.0, connect=10.0)
        client = httpx.Client(timeout=timeout_config)
        
        try:
            # Prepare the multipart form data
            files_to_send, data, opened_files = self._prepare_request_data(
                image_paths, prompt, system_prompt, max_new_tokens
            )
            
            response = client.post(
                f"{self.api_url}/analyze-images/",
                data=data,
                files=files_to_send,
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            response_data = response.json()
            output = {"response": response_data["response"]}
            
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "num_images": len(image_paths),
                "analysis_status": "completed",
            }

            return output, metadata

        except httpx.TimeoutException as e:
            return self._create_error_response(
                image_paths,
                prompt,
                f"Error: The request to the MedGemma API timed out after {timeout_config.read} seconds. The server might be overloaded or the model is taking too long to load. Try again later.",
                "timeout_error",
                str(e)
            )
        except httpx.ConnectError as e:
            return self._create_error_response(
                image_paths,
                prompt,
                f"Error: Could not connect to the MedGemma API. Check if the server address '{self.api_url}' is correct and running.",
                "connection_error",
                str(e)
            )
        except httpx.HTTPStatusError as e:
            return self._create_error_response(
                image_paths,
                prompt,
                f"Error: The MedGemma API returned an error (Status {e.response.status_code}): {e.response.text}",
                "http_error",
                f"Status {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            return self._create_error_response(
                image_paths,
                prompt,
                f"An unexpected error occurred in the MedGemma client tool: {str(e)}",
                "general_error",
                str(e)
            )
        finally:
            # Ensure all opened files are closed
            if 'opened_files' in locals():
                for f in opened_files:
                    f.close()

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist who is able to analyze radiological images at any resolution.",
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the tool asynchronously."""
        async with httpx.AsyncClient() as client:
            try:
                # Prepare the multipart form data
                files_to_send, data, opened_files = self._prepare_request_data(
                    image_paths, prompt, system_prompt, max_new_tokens
                )
                
                response = await client.post(
                    f"{self.api_url}/analyze-images/",
                    data=data,
                    files=files_to_send,
                    timeout=120.0
                )
                response.raise_for_status()
                
                response_data = response.json()
                output = {"response": response_data["response"]}
                
                metadata = {
                    "image_paths": image_paths,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": max_new_tokens,
                    "num_images": len(image_paths),
                    "analysis_status": "completed",
                }

                return output, metadata

            except httpx.HTTPStatusError as e:
                return self._create_error_response(
                    image_paths,
                    prompt,
                    f"Error calling MedGemma API: {e.response.status_code} - {e.response.text}",
                    "http_error",
                    f"Status {e.response.status_code}: {e.response.text}"
                )
            except Exception as e:
                return self._create_error_response(
                    image_paths,
                    prompt,
                    f"An unexpected error occurred: {str(e)}",
                    "general_error",
                    str(e)
                )
            finally:
                # Ensure all opened files are closed
                if 'opened_files' in locals():
                    for f in opened_files:
                        f.close()
