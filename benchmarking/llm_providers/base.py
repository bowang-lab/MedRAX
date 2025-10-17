"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import base64
from pathlib import Path
from medrax.utils.utils import load_prompts_from_file


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    text: str
    images: Optional[List[str]] = None  # List of image paths


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    chunk_history: Optional[Any] = None
    

class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    This class defines the interface for all LLM providers, standardizing
    text + image input -> text output across different models and APIs.
    """

    def __init__(self, model_name: str, system_prompt: str, **kwargs):
        """Initialize the LLM provider.
        
        Args:
            model_name (str): Name of the model to use
            system_prompt (str): System prompt identifier to load from file
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 5000)
        self.prompt_name = system_prompt
        
        # Load system prompt content from file
        try:
            prompts = load_prompts_from_file("benchmarking/system_prompts.txt")
            self.system_prompt = prompts.get(self.prompt_name, None)
            if self.system_prompt is None:
                print(f"Warning: System prompt '{system_prompt}' not found in benchmarking/system_prompts.txt.")
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            self.system_prompt = None

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
                text="Hello! What model are you? Tell me your full specification."
            )
            response = self.generate_response(test_request)
            return response.content is not None and len(response.content.strip()) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

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

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"ERROR: _encode_image failed for {image_path} (type: {type(image_path)}): {e}")
            raise
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """Detect the MIME type of an image file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: MIME type (e.g., 'image/png', 'image/jpeg')
        """
        # Get file extension
        ext = Path(image_path).suffix.lower()
        
        # Map extensions to MIME types
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
        }
        
        return mime_types.get(ext, 'image/png')  # Default to PNG for medical images

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model_name})" 