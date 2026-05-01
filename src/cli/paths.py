from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AppPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_definitions_file: Path
    prompts_file: Path
    output_file: Path
