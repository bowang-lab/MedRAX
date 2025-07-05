"""Factory for creating language model instances based on model name."""

import os
from typing import Dict, Any, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


class ModelFactory:
    """Factory for creating language model instances based on model name.
    
    This class implements a registry of language model providers and provides 
    methods to create appropriate language model instances based on the model name.
    """
    
    # Registry of model providers
    _model_providers = {
        "gpt": {
            "class": ChatOpenAI,
            "env_key": "OPENAI_API_KEY",
            "base_url_key": "OPENAI_BASE_URL"
        },
        "chatgpt": {
            "class": ChatOpenAI,
            "env_key": "OPENAI_API_KEY",
            "base_url_key": "OPENAI_BASE_URL"
        },
        "gemini": {
            "class": ChatGoogleGenerativeAI,
            "env_key": "GOOGLE_API_KEY"
        },
        "openrouter": {
            "class": ChatOpenAI,  # OpenRouter uses OpenAI-compatible interface
            "env_key": "OPENROUTER_API_KEY",
            "base_url_key": "OPENROUTER_BASE_URL",
            "default_base_url": "https://openrouter.ai/api/v1"
        },
        # Add more providers with default configurations here
    }
    
    @classmethod
    def register_provider(cls, prefix: str, model_class: Type[BaseLanguageModel], 
                          env_key: str, **kwargs) -> None:
        """Register a new model provider.
        
        Args:
            prefix (str): The prefix used to identify this model provider (e.g., 'gpt', 'gemini')
            model_class (Type[BaseLanguageModel]): The LangChain model class to use
            env_key (str): The environment variable name for the API key
            **kwargs: Additional provider-specific configuration
        """
        cls._model_providers[prefix] = {
            "class": model_class,
            "env_key": env_key,
            **kwargs
        }
    
    @classmethod
    def create_model(cls, model_name: str, temperature: float = 0.7, 
                     top_p: float = 0.95, **kwargs) -> BaseLanguageModel:
        """Create and return an instance of the appropriate language model.
        
        Args:
            model_name (str): Name of the model to create (e.g., 'gpt-4o', 'gemini-2.5-pro')
            temperature (float, optional): Temperature parameter. Defaults to 0.7.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.95.
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseLanguageModel: An initialized language model instance
            
        Raises:
            ValueError: If no provider is found for the given model name
            ValueError: If the required API key is missing
        """
        # Find the matching provider based on model name prefix
        provider_prefix = next(
            (prefix for prefix in cls._model_providers if model_name.startswith(prefix)),
            None
        )
        
        if not provider_prefix:
            raise ValueError(
                f"No provider found for model: {model_name}. "
                f"Registered providers are for: {list(cls._model_providers.keys())}"
            )
        
        provider = cls._model_providers[provider_prefix]
        model_class = provider["class"]
        env_key = provider["env_key"]
        
        # Set up provider-specific kwargs
        provider_kwargs = {}
        
        # Handle API key
        if env_key in os.environ:
            provider_kwargs["api_key"] = os.environ[env_key]
        else:
            # Log warning but don't fail - the model class might handle missing API keys differently
            print(f"Warning: Environment variable {env_key} not found. Authentication may fail.")
        
        # Check for base_url if applicable
        if "base_url_key" in provider:
            if provider["base_url_key"] in os.environ:
                provider_kwargs["base_url"] = os.environ[provider["base_url_key"]]
            elif "default_base_url" in provider:
                provider_kwargs["base_url"] = provider["default_base_url"]
        
        # Merge with any additional provider-specific settings from the registry
        for k, v in provider.items():
            if k not in ["class", "env_key", "base_url_key", "default_base_url"]:
                provider_kwargs[k] = v
        
        # Strip the provider prefix from the model name
        # For example, 'openrouter-anthropic/claude-sonnet-4' becomes 'anthropic/claude-sonnet-4'
        actual_model_name = model_name
        if model_name.startswith(f"{provider_prefix}-"):
            actual_model_name = model_name[len(provider_prefix)+1:]
            
        # Create and return the model instance
        return model_class(
            model=actual_model_name,
            temperature=temperature,
            top_p=top_p,
            **provider_kwargs,
            **kwargs
        )
    
    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered model providers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of registered providers and their configurations
        """
        # Return a copy to prevent accidental modification
        return {k: {kk: vv for kk, vv in v.items() if kk != "class"} 
                for k, v in cls._model_providers.items()}
