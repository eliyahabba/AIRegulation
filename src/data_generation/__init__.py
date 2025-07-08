"""
Data Generation Module
Contains task classes for generating prompt variations for different NLP tasks.
"""

from .base_task import BaseTask
from .airbench_task import AirbenchTask
from .bbq_task import BBQTask

__all__ = ['BaseTask', 'AirbenchTask', 'BBQTask'] 