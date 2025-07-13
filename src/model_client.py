"""
Client for interacting with language models with extensible platform support.

Usage examples:
    # Use API-based model
    response = get_completion("Hello", platform="OpenAI", model_name="gpt-3.5-turbo")
    
    # Use local model (requires transformers and torch)
    response = get_completion("Hello", platform="local", model_name="microsoft/DialoGPT-small")
    
    # Use local model with platform=None (shortcut)
    response = get_completion("Hello", platform=None, model_name="microsoft/DialoGPT-small")
"""
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from src.constants import GenerationDefaults, MODELS
from src.exceptions import APIKeyMissingError
from src.providers import (
    LocalProvider,
    TogetherAIProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CohereProvider,
)

# Load environment variables from .env file
load_dotenv()

# Global LocalProvider instance to avoid reloading models
_local_provider_instance = None


def _get_local_provider(quantization: Optional[str] = None) -> 'LocalProvider':
    """Get or create the global LocalProvider instance."""
    global _local_provider_instance
    
    # If we don't have an instance yet, or quantization settings changed, create a new one
    if (_local_provider_instance is None or 
        _local_provider_instance.quantization != quantization):
        _local_provider_instance = LocalProvider(api_key=None, quantization=quantization)
    
    return _local_provider_instance


# Platform registry
PLATFORM_PROVIDERS = {
    "TogetherAI": TogetherAIProvider,
    "OpenAI": OpenAIProvider,
    "Anthropic": AnthropicProvider,
    "Google": GoogleProvider,
    "Cohere": CohereProvider,
    "local": LocalProvider,
}

# Environment variable mapping
PLATFORM_ENV_VARS = {
    "TogetherAI": "TOGETHER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Cohere": "COHERE_API_KEY",
    "local": None,  # No API key needed for local models
}


def get_batch_model_responses(batch_messages: List[List[Dict[str, str]]],
                              model_name: str = GenerationDefaults.MODEL_NAME,
                              max_tokens: Optional[int] = None,
                              platform: Optional[str] = "TogetherAI",
                              temperature: float = 0.0,
                              api_key: Optional[str] = None,
                              quantization: Optional[str] = None) -> List[str]:
    """Get responses for a batch of conversations (more efficient for local models)."""
    if not batch_messages:
        return []

    # For non-local platforms, fall back to individual processing
    if platform != "local":
        return [get_model_response(messages, model_name, max_tokens, platform, temperature, api_key, quantization)
                for messages in batch_messages]

    # For local platform, use batch processing
    # Resolve model name: if it's already a full model name (contains "/"), use it directly
    # Otherwise, resolve it using the MODELS dictionary
    resolved_model_name = model_name
    if "/" not in model_name:
        # This is a short key, resolve it
        if platform not in MODELS:
            raise ValueError(f"Unsupported platform: {platform}")
        platform_models = MODELS[platform]
        if model_name not in platform_models:
            raise ValueError(f"Unsupported model '{model_name}' for platform '{platform}'")
        resolved_model_name = platform_models[model_name]

    provider = _get_local_provider(quantization)
    responses = provider.get_batch_responses(batch_messages, resolved_model_name, max_tokens, temperature)
    return [response.get("parsed_response", response.get("full_response", "")) for response in responses]


def get_model_response(messages: List[Dict[str, str]],
                       model_name: str = GenerationDefaults.MODEL_NAME,
                       max_tokens: Optional[int] = None,
                       platform: Optional[str] = "TogetherAI",
                       temperature: float = 0.0,
                       api_key: Optional[str] = None,
                       quantization: Optional[str] = None) -> str:
    """Get response from the model by selecting the appropriate provider."""

    # If platform is None, use local provider
    if platform is None:
        platform = "local"

    # Resolve model name: if a short name is given, convert to full name using MODELS
    resolved_model_name = model_name
    if platform in MODELS and model_name in MODELS[platform]:
        resolved_model_name = MODELS[platform][model_name]

    # Handle local provider (no API key needed)
    if platform == "local":
        provider = _get_local_provider(quantization)
        response = provider.get_response(messages, resolved_model_name, max_tokens, temperature)
        return response.get("parsed_response", response.get("full_response", ""))

    if platform not in PLATFORM_PROVIDERS:
        supported_platforms = list(PLATFORM_PROVIDERS.keys())
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {supported_platforms}")

    # Get API key (skip for local platform)
    current_api_key = api_key if api_key is not None else os.getenv(PLATFORM_ENV_VARS[platform])
    if not current_api_key and PLATFORM_ENV_VARS[platform] is not None:
        raise APIKeyMissingError(
            f"API key for {platform} is missing. Set the {platform.upper().replace(' ', '_')}_API_KEY environment variable.")

    # Create provider and get response
    provider_class = PLATFORM_PROVIDERS[platform]
    try:
        provider = provider_class(current_api_key)
        response = provider.get_response(messages, resolved_model_name, max_tokens, temperature)
        return response.get("parsed_response", response.get("full_response", ""))
    except ImportError as e:
        raise ImportError(f"Failed to initialize {platform} provider: {e}")
    except Exception as e:
        raise RuntimeError(f"Error getting response from {platform}: {e}")


def get_completion(prompt: str,
                   model_name: str = GenerationDefaults.MODEL_NAME,
                   max_tokens: Optional[int] = None,
                   platform: Optional[str] = "TogetherAI",
                   api_key: Optional[str] = None,
                   quantization: Optional[str] = None) -> str:
    """
    Get a completion from the language model using a simple prompt.
    
    Args:
        prompt: The prompt text
        model_name: Name of the model to use
        max_tokens: Maximum number of tokens for the response
        platform: Platform to use (supported: TogetherAI, OpenAI, Anthropic, Google, Cohere, local, None)
                 If None, uses local provider
        api_key: Optional API key to use for the platform
        
    Returns:
        The model's response text
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    return get_model_response(messages=messages, model_name=model_name,
                              max_tokens=max_tokens, platform=platform, api_key=api_key,
                              quantization=quantization)


def get_supported_platforms() -> List[str]:
    """Get list of supported platforms."""
    return list(PLATFORM_PROVIDERS.keys())


def is_platform_available(platform: str) -> bool:
    """Check if a platform is available (has required dependencies)."""
    if platform not in PLATFORM_PROVIDERS:
        return False

    try:
        # Try to import required dependencies for each platform
        if platform == "TogetherAI":
            import together
        elif platform == "OpenAI":
            import openai
        elif platform == "Anthropic":
            import anthropic
        elif platform == "Google":
            import google.generativeai
        elif platform == "Cohere":
            import cohere
        elif platform == "local":
            import transformers
            import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Test the client
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")
    print(f"Supported platforms: {get_supported_platforms()}")

    for platform in get_supported_platforms():
        if is_platform_available(platform):
            print(f"✅ {platform} is available")
            try:
                if platform == "local":
                    # Test local platform with a small model
                    response = get_completion(test_prompt, platform=platform, model_name="microsoft/DialoGPT-small")
                    print(f"{platform} Response: {response[:100]}...")
                else:
                    # Only test if we have an API key
                    env_var = PLATFORM_ENV_VARS[platform]
                    if env_var and os.getenv(env_var):
                        response = get_completion(test_prompt, platform=platform)
                        print(f"{platform} Response: {response[:100]}...")
                    else:
                        print(f"   (No API key found for {env_var})")
            except Exception as e:
                print(f"   Error testing {platform}: {e}")
        else:
            print(f"❌ {platform} is not available (missing dependencies)")

    # Test with platform=None (should use local)
    print("\n--- Testing with platform=None (should use local) ---")
    if is_platform_available("local"):
        try:
            response = get_completion(test_prompt, platform=None, model_name="microsoft/DialoGPT-small")
            print(f"Local (via None) Response: {response[:100]}...")
        except Exception as e:
            print(f"Error testing local via None: {e}")
    else:
        print("Local platform not available")
