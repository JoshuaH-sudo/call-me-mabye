import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .constrain_decoder import FunctionDefinition


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class AppPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_definitions_file: Path
    prompts_file: Path
    output_file: Path


class DatasetFileLoader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: AppPaths

    @classmethod
    def from_argv(cls, argv: list[str]) -> "DatasetFileLoader":
        if len(argv) != 6:
            raise RuntimeError(
                "usage: python -m src --function_definitions <path> "
                "--input <path> --output <path>"
            )

        arguments = dict(zip(argv[::2], argv[1::2], strict=True))
        function_definitions_value = (
            arguments.get("--function_definitions")
            or arguments.get("--functions_definition")
            or arguments.get("--function_defintions")
        )
        if function_definitions_value is None:
            raise RuntimeError(
                "missing required argument: --function_definitions"
            )

        try:
            paths = AppPaths(
                function_definitions_file=Path(function_definitions_value),
                prompts_file=Path(arguments["--input"]),
                output_file=Path(arguments["--output"]),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"missing required argument: {exc.args[0]}"
            ) from None

        return cls(paths=paths)

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
