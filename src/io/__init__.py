"""File I/O sub-package.

``loader``  — reads and validates function definitions and prompts from JSON.
``writer``  — serializes results back to a JSON file on disk.
"""
from .loader import DatasetFileLoader, PromptCase
from .writer import output_results

__all__ = [
    "DatasetFileLoader",
    "PromptCase",
    "output_results",
]
