"""
Evaluation package for AI Regulation datasets.
Contains specialized evaluation modules for different dataset types.
"""

from .evaluate_airbench import main as evaluate_airbench_main
from .evaluate_bbq import main as evaluate_bbq_main

__all__ = [
    'evaluate_airbench_main',
    'evaluate_bbq_main'
] 