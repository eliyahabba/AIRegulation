"""
Local provider for Hugging Face models.
"""

import os
import warnings
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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

        # Return the full response instead of trying to extract only the new part
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

            # Return the full response instead of trying to extract only the new part
            responses.append(full_response.strip())

        return responses 