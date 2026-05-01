"""Dataset file loading for function definitions and test prompts.

:class:`DatasetFileLoader` is a Pydantic model that holds the resolved file
paths and exposes two loading methods.  Keeping file I/O in a dedicated
class makes it straightforward to inject alternative paths in tests.
"""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..decoder.models import FunctionDefinition
from ..cli.paths import AppPaths


class PromptCase(BaseModel):
    """A single test-prompt entry from the prompts JSON file.

    Attributes:
        prompt: The user-facing text that the decoder will turn into a
            function call.  Must be at least one character long.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class DatasetFileLoader(BaseModel):
    """Reads and validates the two input JSON files required by the pipeline.

    Use :attr:`paths` to point the loader at different files; the model
    itself carries no mutable state, so the same instance can be reused.

    Attributes:
        paths: Resolved file-system paths for the function definitions file,
            the prompts file, and the output destination.
    """

    model_config = ConfigDict(extra="forbid")

    paths: AppPaths

    def _load_json(self, path: Path) -> object:
        """Read a JSON file and return its parsed content as a Python object.

        Args:
            path: Absolute or relative path to the JSON file.

        Returns:
            The deserialised Python object (list, dict, etc.).

        Raises:
            RuntimeError: On missing file, malformed JSON, or any OS-level
                read error.
        """
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
        """Load and validate the function definitions file.

        The file must contain a JSON array where every element matches the
        :class:`~src.decoder.models.FunctionDefinition` schema.

        Returns:
            A list of validated function definitions.

        Raises:
            RuntimeError: If the file is missing, not a JSON array, or any
                element fails Pydantic validation.
        """
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
        """Load and validate the prompts file.

        The file must contain a JSON array where every element has at least
        a non-empty ``"prompt"`` string field.

        Returns:
            A list of validated prompt cases.

        Raises:
            RuntimeError: If the file is missing, not a JSON array, or any
                element fails Pydantic validation.
        """
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
