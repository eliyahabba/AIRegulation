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
from abc import abstractmethod
from typing import List, Dict, Optional, Protocol

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.constants import GenerationDefaults, MODELS
from src.exceptions import APIKeyMissingError

# Load environment variables from .env file
load_dotenv()

# Environment variable mapping
PLATFORM_ENV_VARS = {
    "TogetherAI": "TOGETHER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Cohere": "COHERE_API_KEY",
}


class ModelProvider(Protocol):
    """Protocol for model providers."""

    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        """Get response from the model provider."""
        pass


class LocalProvider:
    """Provider for local Hugging Face models."""

    def __init__(self, api_key: str = None):
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.torch = torch
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _load_model(self, model_name: str):
        """Load model and tokenizer if not already loaded or if model changed."""
        if self.model is None or self.current_model_name != model_name:
            print(f"Loading local model: {model_name}")
            self.tokenizer = self.AutoTokenizer.from_pretrained(model_name)
            self.model = self.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16 if self.device == "cuda" or self.device == "mps" else self.torch.float32,
                device_map="auto" if self.device != "cpu" else None
            )
            self.current_model_name = model_name

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        self._load_model(model_name)

        # Try to use chat template if available (for chat models like LLaMA-3-chat, Mistral-instruct)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            try:
                # Use the proper chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print(f"Debug: Using chat template.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply chat template for model {model_name}: {e}. Ensure the model supports chat templating correctly.")
        else:
            raise RuntimeError(
                f"Model {model_name} does not have a chat template. Please use a chat-tuned model (e.g., LLaMA-3-chat, Mistral-instruct) or a model with a defined chat_template.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # Move to appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with self.torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_tokens or 512,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True
            else:
                generate_kwargs["do_sample"] = False

            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new generated part
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            # If the prompt doesn't match exactly (can happen with chat templates), 
            # try to extract the last assistant response
            response = self._extract_assistant_response(full_response, messages)

        return response

    def _extract_assistant_response(self, full_response: str, original_messages: List[Dict[str, str]]) -> str:
        """Extract the assistant's response from the full generated text."""
        # Try common assistant markers
        assistant_markers = ["Assistant:", "assistant:", "ASSISTANT:", "<|assistant|>", "<|im_start|>assistant"]

        for marker in assistant_markers:
            if marker in full_response:
                parts = full_response.split(marker)
                if len(parts) > 1:
                    # Get the last assistant response
                    response = parts[-1].strip()
                    # Remove any trailing markers or special tokens
                    response = response.split("<|")[0].strip()  # Remove any trailing special tokens
                    response = response.split("User:")[0].strip()  # Remove any following user input
                    response = response.split("Human:")[0].strip()  # Remove any following human input
                    return response

        # If no markers found, try to get the last part of the response
        # This is a fallback for cases where the format is unclear
        lines = full_response.split('\n')
        if lines:
            return lines[-1].strip()

        return full_response.strip()


class TogetherAIProvider:
    """Provider for TogetherAI platform."""

    def __init__(self, api_key: str):
        try:
            from together import Together
            self.client = Together(api_key=api_key)
        except ImportError:
            raise ImportError(
                "together package is required for TogetherAI provider. Install with: pip install together")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class OpenAIProvider:
    """Provider for OpenAI platform."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider. Install with: pip install openai")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class AnthropicProvider:
    """Provider for Anthropic platform."""

    def __init__(self, api_key: str):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. Install with: pip install anthropic")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        # Convert OpenAI format to Anthropic format
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        kwargs = {
            "model": model_name,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
        }

        if system_message:
            kwargs["system"] = system_message

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class GoogleProvider:
    """Provider for Google (Gemini) platform."""

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Google provider. Install with: pip install google-generativeai")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        model = self.genai.GenerativeModel(model_name)

        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt = "\n".join(prompt_parts)

        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text


class CohereProvider:
    """Provider for Cohere platform."""

    def __init__(self, api_key: str):
        try:
            import cohere
            self.client = cohere.Client(api_key)
        except ImportError:
            raise ImportError("cohere package is required for Cohere provider. Install with: pip install cohere")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        # Convert to Cohere chat format
        chat_history = []
        message = ""

        for msg in messages:
            if msg["role"] == "system":
                # Add system message as preamble
                message = f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                if chat_history:
                    chat_history.append({"role": "USER", "message": msg["content"]})
                else:
                    message += msg["content"]
            elif msg["role"] == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg["content"]})

        kwargs = {
            "model": model_name,
            "message": message,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if chat_history:
            kwargs["chat_history"] = chat_history

        response = self.client.chat(**kwargs)
        return response.text


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


def get_model_response(messages: List[Dict[str, str]],
                       model_name: str = GenerationDefaults.MODEL_NAME,
                       max_tokens: Optional[int] = None,
                       platform: Optional[str] = "TogetherAI",
                       temperature: float = 0.0,
                       api_key: Optional[str] = None,
                       require_gpu: bool = True) -> str:
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
        provider = LocalProvider()
        if require_gpu and provider.device == "cpu":
            raise RuntimeError(
                f"GPU required for local model {model_name} but only CPU is available. Set require_gpu=False to allow CPU usage.")
        return provider.get_response(messages, resolved_model_name, max_tokens, temperature)

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
        return provider.get_response(messages, resolved_model_name, max_tokens, temperature)
    except ImportError as e:
        raise ImportError(f"Failed to initialize {platform} provider: {e}")
    except Exception as e:
        raise RuntimeError(f"Error getting response from {platform}: {e}")


def get_completion(prompt: str,
                   model_name: str = GenerationDefaults.MODEL_NAME,
                   max_tokens: Optional[int] = None,
                   platform: Optional[str] = "TogetherAI",
                   api_key: Optional[str] = None,
                   require_gpu: bool = True) -> str:
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
                              max_tokens=max_tokens, platform=platform, api_key=api_key, require_gpu=require_gpu)


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
