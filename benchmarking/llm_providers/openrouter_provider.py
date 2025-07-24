"""xAI LLM provider implementation using OpenRouter API via OpenAI SDK."""

import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import base64
from openai import OpenAI

from .base import LLMProvider, LLMRequest, LLMResponse


class OpenRouterProvider(LLMProvider):
    """LLM provider using OpenRouter API via OpenAI SDK."""

    def _setup(self) -> None:
        """Set up OpenRouter client models."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for xAI Grok via OpenRouter.")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        # Use OpenAI SDK with OpenRouter endpoint
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenRouter Grok model via OpenAI SDK.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
        Returns:
            LLMResponse: The response from xAI Grok via OpenRouter
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        user_content = []
        user_content.append({"type": "text", "text": request.text})
        
        # Add images if provided
        if request.images:
            valid_images = self._validate_image_paths(request.images)
            for image_path in valid_images:
                try:
                    image_b64 = self._encode_image(image_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                **(request.additional_params or {})
            )
            duration = time.time() - start_time
            content = response.choices[0].message.content if response.choices else ""
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
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