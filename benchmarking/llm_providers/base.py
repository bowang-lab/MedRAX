"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import base64
import time
from pathlib import Path


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    text: str
    images: Optional[List[str]] = None  # List of image paths
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    raw_response: Optional[Any] = None
    

class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    This class defines the interface for all LLM providers, standardizing
    text + image input -> text output across different models and APIs.
    """

    def __init__(self, model_name: str, **kwargs):
        """Initialize the LLM provider.
        
        Args:
            model_name (str): Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Set up the provider (API keys, client initialization, etc.)."""
        pass

    @abstractmethod
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
            
        Returns:
            LLMResponse: The response from the LLM
        """
        pass

    def test_connection(self) -> bool:
        """Test the connection to the LLM provider.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple test request
            test_request = LLMRequest(
                text="Hello",
                temperature=0.5,
                max_tokens=1000
            )
            response = self.generate_response(test_request)
            return response.content is not None and len(response.content.strip()) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _validate_image_paths(self, image_paths: List[str]) -> List[str]:
        """Validate that image paths exist and are readable.
        
        Args:
            image_paths (List[str]): List of image paths to validate
            
        Returns:
            List[str]: List of valid image paths
        """
        valid_paths = []
        for path in image_paths:
            if Path(path).exists() and Path(path).is_file():
                valid_paths.append(path)
            else:
                print(f"Warning: Image path does not exist: {path}")
        return valid_paths

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model_name})" 