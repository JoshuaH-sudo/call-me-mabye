"""Unit tests for src.prompt.builder.PromptContextBuilder."""
import pytest

from src.decoder.models import FunctionDefinition, ParameterDefinition, ReturnDefinition
from src.prompt.builder import PromptContextBuilder


def _make_function_definition(
    name: str,
    params: dict[str, str],
    description: str = "A test function.",
) -> FunctionDefinition:
    """Helper: build a minimal FunctionDefinition."""
    return FunctionDefinition(
        name=name,
        description=description,
        parameters={
            p: ParameterDefinition(type=t) for p, t in params.items()
        },
        returns=ReturnDefinition(type="number"),
    )


@pytest.fixture()
def builder() -> PromptContextBuilder:
    return PromptContextBuilder()


@pytest.fixture()
def two_functions() -> list[FunctionDefinition]:
    return [
        _make_function_definition("add", {"a": "number", "b": "number"}),
        _make_function_definition("greet_user", {"name": "string"}),
    ]


class TestBuildSchemaBlock:
    def test_wraps_in_available_functions_tags(
        self, builder: PromptContextBuilder, two_functions: list[FunctionDefinition]
    ) -> None:
        block = builder.build_schema_block(two_functions)
        assert block.startswith("<available_functions>")
        assert block.endswith("</available_functions>")

    def test_contains_function_names(
        self, builder: PromptContextBuilder, two_functions: list[FunctionDefinition]
    ) -> None:
        block = builder.build_schema_block(two_functions)
        assert "add(" in block
        assert "greet_user(" in block

    def test_number_params_displayed_as_float(
        self, builder: PromptContextBuilder
    ) -> None:
        fn = _make_function_definition("calc", {"x": "number"})
        block = builder.build_schema_block([fn])
        assert "x: float" in block

    def test_string_params_displayed_as_str(
        self, builder: PromptContextBuilder
    ) -> None:
        fn = _make_function_definition("echo", {"msg": "string"})
        block = builder.build_schema_block([fn])
        assert "msg: str" in block

    def test_unknown_type_preserved_as_is(
        self, builder: PromptContextBuilder
    ) -> None:
        fn = FunctionDefinition(
            name="exotic",
            description="Has an unusual type.",
            parameters={"val": ParameterDefinition(type="boolean")},
            returns=ReturnDefinition(type="boolean"),
        )
        block = builder.build_schema_block([fn])
        assert "val: boolean" in block

    def test_empty_function_list(self, builder: PromptContextBuilder) -> None:
        block = builder.build_schema_block([])
        assert "<available_functions>" in block
        assert "</available_functions>" in block

    def test_multiple_params_comma_separated(
        self, builder: PromptContextBuilder
    ) -> None:
        fn = _make_function_definition("multi", {"a": "number", "b": "string", "c": "number"})
        block = builder.build_schema_block([fn])
        assert "a: float, b: str, c: float" in block

    def test_no_param_function(self, builder: PromptContextBuilder) -> None:
        fn = FunctionDefinition(
            name="noop",
            description="Does nothing.",
            parameters={},
            returns=ReturnDefinition(type="number"),
        )
        block = builder.build_schema_block([fn])
        assert "noop()" in block


class TestBuildEnrichedPrompt:
    def test_contains_schema_block(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        result = builder.build_enriched_prompt("hello?", two_functions)
        assert "<available_functions>" in result
        assert "</available_functions>" in result

    def test_contains_instruction_tag(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        result = builder.build_enriched_prompt("hello?", two_functions)
        assert "<instruction>" in result
        assert "</instruction>" in result

    def test_contains_question_tag_with_raw_prompt(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        raw = "What is the sum of 2 and 3?"
        result = builder.build_enriched_prompt(raw, two_functions)
        assert f"<question>{raw}</question>" in result

    def test_schema_block_appears_before_instruction(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        result = builder.build_enriched_prompt("hello?", two_functions)
        schema_pos = result.index("<available_functions>")
        instruction_pos = result.index("<instruction>")
        assert schema_pos < instruction_pos

    def test_instruction_appears_before_question(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        result = builder.build_enriched_prompt("hello?", two_functions)
        instruction_pos = result.index("<instruction>")
        question_pos = result.index("<question>")
        assert instruction_pos < question_pos

    def test_raw_prompt_not_altered_by_enrichment(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        raw = "Reverse the string hello."
        result = builder.build_enriched_prompt(raw, two_functions)
        # The raw prompt must appear verbatim inside <question> tags.
        assert raw in result

    def test_returns_plain_string(
        self,
        builder: PromptContextBuilder,
        two_functions: list[FunctionDefinition],
    ) -> None:
        result = builder.build_enriched_prompt("test", two_functions)
        assert isinstance(result, str)
