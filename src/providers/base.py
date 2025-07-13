"""
Base protocol for model providers.
"""

from abc import abstractmethod
from typing import List, Dict, Optional, Protocol


class ModelProvider(Protocol):
    """Protocol for model providers."""

    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> Dict[str, str]:
        """Get response from the model provider."""
        pass 