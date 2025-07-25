"""OpenAI LLM provider implementation using langchain_openai."""

import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base import LLMProvider, LLMRequest, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider for GPT models using langchain_openai."""

    def _setup(self) -> None:
        """Set up OpenAI langchain client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        base_url = os.getenv("OPENAI_BASE_URL")
        
        # Create ChatOpenAI instance
        kwargs = {
            "model": self.model_name,
            "api_key": api_key,
        }
        
        if base_url:
            kwargs["base_url"] = base_url
            
        self.client = ChatOpenAI(**kwargs)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using langchain OpenAI.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
            
        Returns:
            LLMResponse: The response from OpenAI
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Build user message content
        user_content = []
        user_content.append({
            "type": "text",
            "text": request.text
        })
        
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
        
        messages.append(HumanMessage(content=user_content))
        
        # Make API call using langchain
        try:
            # Update client parameters for this request
            self.client.temperature = request.temperature
            self.client.max_tokens = request.max_tokens
            self.client.top_p = request.top_p
            
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
