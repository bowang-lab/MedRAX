import httpx
from typing import Dict, List, Optional, Type, Any
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
import os

# This input schema should be identical to the one in your original tool
class MedGemmaVQAInput(BaseModel):
    """Input schema for the MedGemma VQA Tool. The agent provides local paths to images."""
    image_paths: List[str] = Field(
        ...,
        description="List of paths to medical image files to analyze. These are local paths accessible to the agent.",
    )
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )

class MedGemmaAPIClientTool(BaseTool):
    """
    A client tool to interact with a remote MedGemma VQA FastAPI service.
    This tool takes local image paths, reads them, and sends them to the API endpoint
    for analysis.
    """
    name: str = "medgemma_medical_vqa_service"
    description: str = (
        "Sends medical images and a prompt to a specialized MedGemma VQA service for analysis. "
        "Use this for expert-level reasoning, diagnosis assistance, and detailed image interpretation "
        "across modalities like chest X-rays, dermatology, etc. Input must be local image paths and a prompt."
    )
    args_schema: Type[BaseModel] = MedGemmaVQAInput
    api_url: str  # The URL of the running FastAPI service

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool synchronously."""
        # httpx is a modern HTTP client that supports sync and async
        timeout_config = httpx.Timeout(300.0, connect=10.0)
        client = httpx.Client(timeout=timeout_config)
        
        # Prepare the multipart form data
        files_to_send = []
        opened_files = []
        try:
            for path in image_paths:
                f = open(path, "rb")
                opened_files.append(f)
                # The key 'images' must match the parameter name in the FastAPI endpoint
                files_to_send.append(("images", (os.path.basename(path), f, "image/jpeg")))

            data = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
            }
            
            response = client.post(
                f"{self.api_url}/analyze-images/",
                data=data,
                files=files_to_send,
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # The agent expects a string response from a tool
            return response.json()["response"]

        # --- KEY FIX 3: More specific exception handling for clearer errors ---
        except httpx.TimeoutException:
            return f"Error: The request to the MedGemma API timed out after {timeout_config.read} seconds. The server might be overloaded or the model is taking too long to load. Try again later."
        except httpx.ConnectError:
            return f"Error: Could not connect to the MedGemma API. Check if the server address '{self.api_url}' is correct and running."
        except httpx.HTTPStatusError as e:
            return f"Error: The MedGemma API returned an error (Status {e.response.status_code}): {e.response.text}"
        except Exception as e:
            return f"An unexpected error occurred in the MedGemma client tool: {str(e)}"
        finally:
            # Important: Ensure all opened files are closed.
            for f in opened_files:
                f.close()

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool asynchronously."""
        async with httpx.AsyncClient() as client:
            files_to_send = []
            opened_files = []
            try:
                # Note: File I/O is blocking, for a truly async app you might use aiofiles
                # But for this use case, this is generally acceptable.
                for path in image_paths:
                    f = open(path, "rb")
                    opened_files.append(f)
                    files_to_send.append(("images", (os.path.basename(path), f, "image/jpeg")))
                
                data = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": max_new_tokens,
                }
                
                response = await client.post(
                    f"{self.api_url}/analyze-images/",
                    data=data,
                    files=files_to_send,
                    timeout=120.0
                )
                response.raise_for_status()
                
                return response.json()["response"]

            except httpx.HTTPStatusError as e:
                return f"Error calling MedGemma API: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"An unexpected error occurred: {str(e)}"
            finally:
                for f in opened_files:
                    f.close()

if __name__ == "__main__":
    client_tool = MedGemmaAPIClientTool(api_url="http://localhost:8002")
    result = client_tool.run({
        "image_paths": ["demo/chest/pneumonia1.jpg"],
        "prompt": "What abnormality do you see?"
    })
    print(result)