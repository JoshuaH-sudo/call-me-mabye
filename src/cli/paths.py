"""Pydantic model for resolved application file paths.

Keeping path resolution in its own model makes it easy to construct
``AppPaths`` from any source (argv, environment variables, tests) without
coupling callers to a specific argument-parsing strategy.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AppPaths(BaseModel):
    """Validated, fully-resolved paths for the three dataset files.

    All three files are required; the model raises a ``ValidationError`` if
    any field is missing or not a valid ``Path``.

    Attributes:
        function_definitions_file: JSON file that lists available functions
            and their parameter schemas.
        prompts_file: JSON file that contains the test prompts to decode.
        output_file: Destination file where results will be written.  The
            parent directory is created automatically if it does not exist.
    """

    model_config = ConfigDict(extra="forbid")

    function_definitions_file: Path
    prompts_file: Path
    output_file: Path
    debug: bool = False
