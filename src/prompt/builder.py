"""Prompt context builder for function-call schema injection.

:class:`PromptContextBuilder` enriches a raw user prompt with two kinds of
contextual signal before it is fed to the model for logit scoring:

1. **Tool/function-call schema tokens** — a structured
   ``<available_functions>`` block that lists every available function's
   name and parameter signatures
   so the model's attention is explicitly directed to the callable set.

2. **Structural hint tokens** — lightweight XML-style tags (``<instruction>``
   and ``<question>``) that delimit the task description from the user's
   request, giving the model clear role boundaries without requiring any
   changes to the underlying :class:`~llm_sdk.Small_LLM_Model`.

The output of :meth:`PromptContextBuilder.build_enriched_prompt` is a plain
``str`` that can be passed directly to ``llm.encode()``.  The raw prompt is
left unmodified and should still be supplied to the candidate builder so that
parameter extractors do not see schema text as candidate values.
"""
from pydantic import BaseModel, ConfigDict

from ..decoder.models import FunctionDefinition

# Type alias for the parameter type strings stored in ParameterDefinition.
_TYPE_DISPLAY: dict[str, str] = {
    "string": "str",
    "number": "float",
}

_INSTRUCTION = (
    "Select the correct function and extract its parameter values from the"
    " question."
)


def _display_type(raw_type: str) -> str:
    """Return a compact Python-style type label for *raw_type*.

    Falls back to the raw type string for unknown types so that new types
    added to the schema do not silently break the builder.

    Args:
        raw_type: The ``type`` field from a
            :class:`~src.decoder.models.ParameterDefinition`.

    Returns:
        A short display string such as ``"str"`` or ``"float"``.
    """
    return _TYPE_DISPLAY.get(raw_type, raw_type)


class PromptContextBuilder(BaseModel):
    """Builds enriched prompt strings for LLM logit scoring.

    This class is stateless; all public methods are pure functions of their
    arguments.  It is a :class:`~pydantic.BaseModel` so that it participates
    in the project's standard validation regime.

    Example usage::

        builder = PromptContextBuilder()
        enriched = builder.build_enriched_prompt(
            raw_prompt="What is the sum of 2 and 3?",
            functions=loaded_functions,
        )
        prefix_ids = llm.encode(enriched)[0].tolist()
    """

    model_config = ConfigDict(extra="forbid")

    def build_schema_block(
        self,
        functions: list[FunctionDefinition],
    ) -> str:
        """Format *functions* as a structured ``<available_functions>`` block.

        Each function is rendered on a single line in the form::

            name(param_name: display_type, …)

        Args:
            functions: The list of function definitions to serialise.

        Returns:
            A multi-line string enclosed in ``<available_functions>`` tags.
            Returns the empty-body tag pair when *functions* is empty.
        """
        lines: list[str] = []
        for fn in functions:
            param_parts = ", ".join(
                f"{param_name}: {_display_type(param_def.type)}"
                for param_name, param_def in fn.parameters.items()
            )
            lines.append(f"{fn.name}({param_parts}) - {fn.description}")

        body = "\n".join(lines)
        return f"<available_functions>\n{body}\n</available_functions>"

    def build_enriched_prompt(
        self,
        raw_prompt: str,
        functions: list[FunctionDefinition],
    ) -> str:
        """Compose a schema block and structural tags around *raw_prompt*.

        The resulting string has three parts, in order:

        1. An ``<available_functions>`` block produced by
           :meth:`build_schema_block`.
        2. An ``<instruction>`` tag containing a fixed task directive.
        3. A ``<question>`` tag wrapping the original *raw_prompt*.

        The output is a plain ``str`` ready to be passed to
        ``llm.encode()``; it does **not** modify *raw_prompt*, which should
        still be used as-is for candidate building and parameter extraction.

        Args:
            raw_prompt: The unmodified user question or instruction.
            functions: All function definitions available for this decode.

        Returns:
            An enriched prompt string that injects schema and structural
            context for the model's logit scoring pass.
        """
        schema_block = self.build_schema_block(functions)
        return (
            f"{schema_block}\n"
            f"<instruction>{_INSTRUCTION}</instruction>\n"
            f"<question>{raw_prompt}</question>"
        )
