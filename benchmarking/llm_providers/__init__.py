"""LLM provider abstractions for benchmarking."""

from .base import LLMProvider, LLMRequest, LLMResponse
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .medrax_provider import MedRAXProvider
from .openrouter_provider import OpenRouterProvider
from .medgemma_provider import MedGemmaProvider

# QwenProvider is optional - only import if dependencies are compatible
try:
    from .qwen_provider import QwenProvider
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    QwenProvider = None
    print(f"QwenProvider not available: {e}")
    print("To use Qwen models, upgrade transformers: pip install --upgrade git+https://github.com/huggingface/transformers")

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "OpenAIProvider", 
    "GoogleProvider",
    "MedRAXProvider",
    "OpenRouterProvider",
    "MedGemmaProvider",
    "QwenProvider",
    "QWEN_AVAILABLE",
] 