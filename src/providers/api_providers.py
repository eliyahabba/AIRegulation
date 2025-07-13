"""
API-based providers for various model platforms.
"""

from typing import List, Dict, Optional


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