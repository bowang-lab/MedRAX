"""Google LLM provider implementation using langchain_google_genai."""

import os
import time
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base import LLMProvider, LLMRequest, LLMResponse


class GoogleProvider(LLMProvider):
    """Google LLM provider for Gemini models using langchain_google_genai."""

    def _setup(self) -> None:
        """Set up Google langchain client."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Create ChatGoogleGenerativeAI instance
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key
        )

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using langchain Google Gemini.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
            
        Returns:
            LLMResponse: The response from Google Gemini
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        
        # For langchain Google Gemini, we need to construct content differently
        if request.images:
            # For multimodal content, use a list format
            content_parts = [request.text]
            
            # Add images if provided
            valid_images = self._validate_image_paths(request.images)
            for image_path in valid_images:
                try:
                    # For langchain Google, pass image data as base64
                    image_b64 = self._encode_image(image_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"
                    })
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
            
            messages.append(HumanMessage(content=content_parts))
        else:
            # Text-only message
            messages.append(HumanMessage(content=request.text))
        
        # Make API call using langchain
        try:
            # Update client parameters for this request
            self.client.temperature = request.temperature
            self.client.max_output_tokens = request.max_tokens
            
            if request.additional_params and "top_p" in request.additional_params:
                self.client.top_p = request.additional_params["top_p"]
            
            response = self.client.invoke(messages)
            
            duration = time.time() - start_time
            
            # Extract response content
            content = response.content if response.content else ""
            
            # Get usage information if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0)
                }
            
            return LLMResponse(
                content=content,
                usage=usage,
                duration=duration,
                raw_response=response
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                duration=time.time() - start_time,
                raw_response=None
            )
