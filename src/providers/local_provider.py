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
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> Dict[str, str]:
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

        # Extract both full response and parsed response
        full_response_clean = full_response.strip()
        parsed_response = self._extract_assistant_response(full_response_clean, messages)
        
        return {
            "full_response": full_response_clean,
            "parsed_response": parsed_response
        }

    def _extract_assistant_response(self, full_response: str, original_messages: List[Dict[str, str]]) -> str:
        """Extract the assistant's response from the full generated text."""
        
        # First, try to reconstruct the original prompt to see what was the input
        try:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                original_prompt = self.tokenizer.apply_chat_template(
                    original_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # If the full response starts with the original prompt, extract what comes after
                if full_response.startswith(original_prompt):
                    response = full_response[len(original_prompt):].strip()
                    if response:
                        return response
                
                # Alternative: try to find the prompt within the full response
                # This handles cases where there might be slight formatting differences
                prompt_index = full_response.find(original_prompt)
                if prompt_index >= 0:
                    response = full_response[prompt_index + len(original_prompt):].strip()
                    if response:
                        return response
        except Exception as e:
            # If there's an error with chat template, continue to fallback methods
            pass
        
        # Fallback: Try to extract based on common patterns
        # Look for patterns that indicate where the assistant response starts
        
        # Pattern 1: Look for the end of the last user message
        if original_messages:
            last_user_content = original_messages[-1].get('content', '')
            if last_user_content and last_user_content in full_response:
                # Find the last occurrence of the user's message
                last_user_index = full_response.rfind(last_user_content)
                if last_user_index >= 0:
                    # Extract everything after the user's message
                    after_user_msg = full_response[last_user_index + len(last_user_content):].strip()
                    if after_user_msg:
                        return after_user_msg
        
        # Pattern 2: Look for common separator tokens and role indicators
        separators = ['<|endoftext|>', '</s>', '<|im_end|>', '<|eot_id|>']
        for sep in separators:
            if sep in full_response:
                # Find the last occurrence of the separator
                last_sep_index = full_response.rfind(sep)
                if last_sep_index >= 0:
                    response = full_response[last_sep_index + len(sep):].strip()
                    if response and response != full_response.strip():
                        return response
        
        # Pattern 3: Look for role indicators (assistant, user, system)
        role_patterns = [
            'assistant\n\n',
            'assistant\n',
            'Assistant:\n',
            'Assistant: ',
            'ASSISTANT:\n',
            'ASSISTANT: '
        ]
        for pattern in role_patterns:
            if pattern in full_response:
                # Find the last occurrence of the pattern
                last_pattern_index = full_response.rfind(pattern)
                if last_pattern_index >= 0:
                    response = full_response[last_pattern_index + len(pattern):].strip()
                    if response and response != full_response.strip():
                        return response
        
        # Pattern 4: Look for standalone "assistant" at the beginning of a line
        lines = full_response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().lower() == 'assistant':
                # Extract everything after this line
                remaining_lines = lines[i+1:]
                response = '\n'.join(remaining_lines).strip()
                if response and response != full_response.strip():
                    return response
        
        # Pattern 5: If response starts with "assistant" followed by newlines, remove it
        if full_response.lower().startswith('assistant'):
            # Find the first non-empty line after "assistant"
            lines = full_response.split('\n')
            for i, line in enumerate(lines):
                if i == 0 and line.strip().lower() == 'assistant':
                    continue
                if line.strip():  # First non-empty line after assistant
                    remaining_lines = lines[i:]
                    response = '\n'.join(remaining_lines).strip()
                    if response and response != full_response.strip():
                        return response
        
        # If we can't extract cleanly, just return the full response
        # This is safer than trying to guess the format
        return full_response.strip()

    def get_batch_responses(self, batch_messages: List[List[Dict[str, str]]], model_name: str,
                            max_tokens: Optional[int] = None, temperature: float = 0.0) -> List[Dict[str, str]]:
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

            # Extract both full response and parsed response
            full_response_clean = full_response.strip()
            parsed_response = self._extract_assistant_response(full_response_clean, batch_messages[i])
            
            responses.append({
                "full_response": full_response_clean,
                "parsed_response": parsed_response
            })

        return responses 