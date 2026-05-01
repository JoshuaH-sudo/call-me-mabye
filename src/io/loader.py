import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..decoder.models import FunctionDefinition
from ..cli.paths import AppPaths


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class DatasetFileLoader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: AppPaths

    def _load_json(self, path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise RuntimeError(f"missing input file: {path}") from None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}: {exc.msg}") from exc
        except OSError as exc:
            raise RuntimeError(f"unable to read {path}: {exc}") from exc

    def load_functions(self) -> list[FunctionDefinition]:
        payload = self._load_json(self.paths.function_definitions_file)
        if not isinstance(payload, list):
            raise RuntimeError(
                "functions file must contain a JSON array: "
                f"{self.paths.function_definitions_file}"
            )
        try:
            return [
                FunctionDefinition.model_validate(item)
                for item in payload
            ]
        except ValidationError as exc:
            raise RuntimeError(
                "invalid function definition in "
                f"{self.paths.function_definitions_file}: {exc}"
            ) from exc

    def load_prompts(self) -> list[PromptCase]:
        payload = self._load_json(self.paths.prompts_file)
        if not isinstance(payload, list):
            raise RuntimeError(
                "prompt file must contain a JSON array: "
                f"{self.paths.prompts_file}"
            )
        try:
            return [PromptCase.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise RuntimeError(
                f"invalid prompt entry in {self.paths.prompts_file}: {exc}"
            ) from exc
