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
import warnings
from abc import abstractmethod
from typing import List, Dict, Optional, Protocol

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    "local": "HF_ACCESS_TOKEN",
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

    def __init__(self, api_key: str = None, quantization: str = None):
        # Suppress some transformers warnings for cleaner output
        warnings.filterwarnings("ignore", message=".*do_sample.*")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.torch = torch
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.quantization = quantization
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.hf_token = api_key or os.getenv("HF_ACCESS_TOKEN")

        # Check quantization compatibility
        if quantization and self.device == "mps":
            print("⚠️ Quantization is not fully supported on MPS, switching to CPU or consider using CUDA")
            self.device = "cpu"

        print(f"Using device: {self.device}")
        if quantization:
            print(f"Using quantization: {quantization}")

    def _load_model(self, model_name: str):
        """Load model and tokenizer if not already loaded or if model changed."""
        if self.model is None or self.current_model_name != model_name:
            print(f"Loading local model: {model_name}")
            self.tokenizer = self.AutoTokenizer.from_pretrained(model_name, token=self.hf_token)

            # Configure quantization if requested
            quantization_config = None
            if self.quantization == "8bit":
                quantization_config = self.BitsAndBytesConfig(load_in_8bit=True)
                print("📉 Using 8-bit quantization")
            elif self.quantization == "4bit":
                quantization_config = self.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch.bfloat16
                )
                print("📉 Using 4-bit quantization (NF4)")

            # Configure model loading parameters
            model_kwargs = {
                "token": self.hf_token,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device != "cpu" else None,
            }

            # Add quantization config if specified
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                # Quantized models usually work best with auto device mapping
                model_kwargs["device_map"] = "auto"
            else:
                # Only set torch_dtype for non-quantized models
                model_kwargs[
                    "torch_dtype"] = self.torch.float16 if self.device == "cuda" or self.device == "mps" else self.torch.float32

            self.model = self.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.current_model_name = model_name

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Clean up generation config to avoid warnings
            if hasattr(self.model, 'generation_config'):
                # Remove sampling parameters that cause warnings when do_sample=False
                gen_config = self.model.generation_config
                if hasattr(gen_config, 'temperature'):
                    gen_config.temperature = None
                if hasattr(gen_config, 'top_p'):
                    gen_config.top_p = None
                if hasattr(gen_config, 'top_k'):
                    gen_config.top_k = None

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

    def get_batch_responses(self, batch_messages: List[List[Dict[str, str]]], model_name: str,
                            max_tokens: Optional[int] = None, temperature: float = 0.0) -> List[str]:
        """Get responses for a batch of conversations (more efficient for local models)."""
        if not batch_messages:
            return []

        # For single item, use regular method
        if len(batch_messages) == 1:
            return [self.get_response(batch_messages[0], model_name, max_tokens, temperature)]

        self._load_model(model_name)

        # Convert all conversations to prompts
        prompts = []
        for messages in batch_messages:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(prompt)
                except Exception as e:
                    raise RuntimeError(f"Failed to apply chat template for model {model_name}: {e}")
            else:
                raise RuntimeError(f"Model {model_name} does not have a chat template.")

        # Tokenize all prompts together
        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True)

        # Move to appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate responses
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

        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            full_response = self.tokenizer.decode(output, skip_special_tokens=True)

            # Extract only the new generated part
            if full_response.startswith(prompts[i]):
                response = full_response[len(prompts[i]):].strip()
            else:
                # Fallback extraction
                response = self._extract_assistant_response(full_response, batch_messages[i])

            responses.append(response)

        return responses


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
    # Import here to avoid circular imports
    from src.constants import MODELS

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

    provider = LocalProvider(api_key, quantization)
    return provider.get_batch_responses(batch_messages, resolved_model_name, max_tokens, temperature)


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
        return LocalProvider(api_key=None, quantization=quantization).get_response(messages, resolved_model_name,
                                                                                   max_tokens, temperature)

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
