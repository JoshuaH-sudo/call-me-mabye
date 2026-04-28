"""
Integration tests for ConstrainedDecoder using valid function definitions.

Each test group focuses on a specific parameter type (string or number) and
uses one or more of the real function definitions that appear in
functions_definition.json.  A CharLevelFakeLLM (see conftest.py) replaces
the real model so the tests run without loading any weights.

How the mock LLM works
----------------------
CharLevelFakeLLM takes a prompt string and a target JSON output string.  At
every generation step it returns logits with a score of 100.0 on the next
character of the target, steering the constrained decoder to produce that
exact output.  Because the decoder only allows tokens that advance the
grammar, we verify both that the output is grammatically valid AND that the
decoder can complete within a reasonable token budget.
"""
import json

import pytest

from src.constrain_decoder import ConstrainedDecoder
from src.decoder_pipeline.models import FunctionDefinition
from tests.conftest import CharLevelFakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_decoder(
    prompt: str,
    target_output: str,
    functions: list[FunctionDefinition],
    max_new_tokens: int = 256,
) -> dict[str, object]:
    """
    Run the constrained decoder with a CharLevelFakeLLM and return the
    parsed JSON result.

    The mock is configured to prefer ``target_output`` at every step, so
    the decoder should reproduce it exactly if the grammar allows it.
    """
    llm = CharLevelFakeLLM(prompt=prompt, target_output=target_output)
    decoder = ConstrainedDecoder(
        available_functions=functions,
        llm=llm,
        max_new_tokens=max_new_tokens,
    )
    prompt_ids = [
        ord(c) for c in prompt if ord(c) < CharLevelFakeLLM.VOCAB_SIZE
    ]
    raw = decoder.force_json_output(prompt_ids)
    return dict(json.loads(raw))


# ===========================================================================
# String parameter tests
# ===========================================================================

class TestStringParameter:
    """
    Tests covering functions whose parameters are of type ``string``.

    Functions exercised
    -------------------
    - fn_greet(name: string)
    - fn_reverse_string(s: string)
    - fn_substitute_string_with_regex(source_string, regex,
      replacement: string)
    """

    # -----------------------------------------------------------------------
    # fn_greet — single string parameter
    # -----------------------------------------------------------------------

    def test_greet_produces_valid_json(
        self, fn_greet: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Greet shrek",
            target_output='{"name":"fn_greet","parameters":{"name":"shrek"}}',
            functions=[fn_greet],
        )
        assert result["name"] == "fn_greet"
        assert "parameters" in result

    def test_greet_parameter_is_string_type(
        self, fn_greet: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Greet shrek",
            target_output='{"name":"fn_greet","parameters":{"name":"shrek"}}',
            functions=[fn_greet],
        )
        assert isinstance(result["parameters"], dict)
        assert isinstance(
            result["parameters"]["name"], str  # type: ignore[index]
        )

    def test_greet_parameter_has_correct_value(
        self, fn_greet: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Greet shrek",
            target_output='{"name":"fn_greet","parameters":{"name":"shrek"}}',
            functions=[fn_greet],
        )
        assert result["parameters"]["name"] == "shrek"  # type: ignore[index]

    def test_greet_different_name(self, fn_greet: FunctionDefinition) -> None:
        result = _run_decoder(
            prompt="Greet john",
            target_output='{"name":"fn_greet","parameters":{"name":"john"}}',
            functions=[fn_greet],
        )
        assert result["parameters"]["name"] == "john"  # type: ignore[index]

    def test_greet_has_exactly_required_parameters(
        self, fn_greet: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Greet alice",
            target_output='{"name":"fn_greet","parameters":{"name":"alice"}}',
            functions=[fn_greet],
        )
        params = result["parameters"]
        assert isinstance(params, dict)
        assert set(params.keys()) == {"name"}  # type: ignore[union-attr]

    # -----------------------------------------------------------------------
    # fn_reverse_string — single string parameter
    # -----------------------------------------------------------------------

    def test_reverse_produces_valid_json(
        self, fn_reverse_string: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Reverse the string 'hello'",
            target_output=(
                '{"name":"fn_reverse_string","parameters":{"s":"hello"}}'
            ),
            functions=[fn_reverse_string],
        )
        assert result["name"] == "fn_reverse_string"

    def test_reverse_parameter_is_string_type(
        self, fn_reverse_string: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Reverse the string 'hello'",
            target_output=(
                '{"name":"fn_reverse_string","parameters":{"s":"hello"}}'
            ),
            functions=[fn_reverse_string],
        )
        assert isinstance(
            result["parameters"]["s"], str  # type: ignore[index]
        )

    def test_reverse_parameter_has_correct_value(
        self, fn_reverse_string: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Reverse the string 'hello'",
            target_output=(
                '{"name":"fn_reverse_string","parameters":{"s":"hello"}}'
            ),
            functions=[fn_reverse_string],
        )
        assert result["parameters"]["s"] == "hello"  # type: ignore[index]

    def test_reverse_has_exactly_required_parameters(
        self, fn_reverse_string: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Reverse hello",
            target_output=(
                '{"name":"fn_reverse_string","parameters":{"s":"hello"}}'
            ),
            functions=[fn_reverse_string],
        )
        keys = set(result["parameters"].keys())  # type: ignore[union-attr]
        assert keys == {"s"}

    # -----------------------------------------------------------------------
    # fn_substitute_string_with_regex — three string parameters
    # -----------------------------------------------------------------------

    def test_substitute_produces_valid_json(
        self, fn_substitute_string_with_regex: FunctionDefinition
    ) -> None:
        target = (
            '{"name":"fn_substitute_string_with_regex",'
            '"parameters":{'
            '"source_string":"hello world",'
            '"regex":"world",'
            '"replacement":"earth"}}'
        )
        result = _run_decoder(
            prompt="Replace 'world' with 'earth' in 'hello world'",
            target_output=target,
            functions=[fn_substitute_string_with_regex],
        )
        assert result["name"] == "fn_substitute_string_with_regex"

    def test_substitute_all_parameters_are_strings(
        self, fn_substitute_string_with_regex: FunctionDefinition
    ) -> None:
        target = (
            '{"name":"fn_substitute_string_with_regex",'
            '"parameters":{'
            '"source_string":"hello",'
            '"regex":"l",'
            '"replacement":"r"}}'
        )
        result = _run_decoder(
            prompt="Replace l with r in hello",
            target_output=target,
            functions=[fn_substitute_string_with_regex],
        )
        params = result["parameters"]
        assert isinstance(params, dict)
        for key in ("source_string", "regex", "replacement"):
            assert isinstance(params[key], str), (  # type: ignore[index]
                f"expected string for parameter '{key}'"
            )

    def test_substitute_has_exactly_three_parameters(
        self, fn_substitute_string_with_regex: FunctionDefinition
    ) -> None:
        target = (
            '{"name":"fn_substitute_string_with_regex",'
            '"parameters":{'
            '"source_string":"abc",'
            '"regex":"b",'
            '"replacement":"x"}}'
        )
        result = _run_decoder(
            prompt="Replace b with x in abc",
            target_output=target,
            functions=[fn_substitute_string_with_regex],
        )
        keys = set(result["parameters"].keys())  # type: ignore[union-attr]
        assert keys == {"source_string", "regex", "replacement"}

    def test_substitute_parameter_values_correct(
        self, fn_substitute_string_with_regex: FunctionDefinition
    ) -> None:
        target = (
            '{"name":"fn_substitute_string_with_regex",'
            '"parameters":{'
            '"source_string":"cat",'
            '"regex":"c",'
            '"replacement":"b"}}'
        )
        result = _run_decoder(
            prompt="Replace c with b in cat",
            target_output=target,
            functions=[fn_substitute_string_with_regex],
        )
        params = result["parameters"]
        assert params["source_string"] == "cat"  # type: ignore[index]
        assert params["regex"] == "c"  # type: ignore[index]
        assert params["replacement"] == "b"  # type: ignore[index]


# ===========================================================================
# Number parameter tests
# ===========================================================================

class TestNumberParameter:
    """
    Tests covering functions whose parameters are of type ``number``.

    Functions exercised
    -------------------
    - fn_add_numbers(a: number, b: number)
    - fn_get_square_root(a: number)
    """

    # -----------------------------------------------------------------------
    # fn_add_numbers — two number parameters
    # -----------------------------------------------------------------------

    def test_add_produces_valid_json(
        self, fn_add_numbers: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the sum of 2 and 3?",
            target_output=(
                '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}'
            ),
            functions=[fn_add_numbers],
        )
        assert result["name"] == "fn_add_numbers"

    def test_add_parameters_are_numeric_types(
        self, fn_add_numbers: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the sum of 2 and 3?",
            target_output=(
                '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}'
            ),
            functions=[fn_add_numbers],
        )
        params = result["parameters"]
        assert isinstance(params, dict)
        for key in ("a", "b"):
            assert isinstance(  # type: ignore[index]
                params[key], (int, float)
            ), f"expected numeric type for parameter '{key}'"

    def test_add_parameter_values_correct(
        self, fn_add_numbers: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the sum of 2 and 3?",
            target_output=(
                '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}'
            ),
            functions=[fn_add_numbers],
        )
        assert result["parameters"]["a"] == 2  # type: ignore[index]
        assert result["parameters"]["b"] == 3  # type: ignore[index]

    def test_add_has_exactly_two_parameters(
        self, fn_add_numbers: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Add 265 and 345",
            target_output=(
                '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}'
            ),
            functions=[fn_add_numbers],
        )
        keys = set(result["parameters"].keys())  # type: ignore[union-attr]
        assert keys == {"a", "b"}

    def test_add_multi_digit_values(
        self, fn_add_numbers: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the sum of 265 and 345?",
            target_output=(
                '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}'
            ),
            functions=[fn_add_numbers],
        )
        assert result["parameters"]["a"] == 265  # type: ignore[index]
        assert result["parameters"]["b"] == 345  # type: ignore[index]

    # -----------------------------------------------------------------------
    # fn_get_square_root — single number parameter
    # -----------------------------------------------------------------------

    def test_sqrt_produces_valid_json(
        self, fn_get_square_root: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the square root of 16?",
            target_output=(
                '{"name":"fn_get_square_root","parameters":{"a":16}}'
            ),
            functions=[fn_get_square_root],
        )
        assert result["name"] == "fn_get_square_root"

    def test_sqrt_parameter_is_numeric_type(
        self, fn_get_square_root: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="What is the square root of 16?",
            target_output=(
                '{"name":"fn_get_square_root","parameters":{"a":16}}'
            ),
            functions=[fn_get_square_root],
        )
        assert isinstance(  # type: ignore[index]
            result["parameters"]["a"], (int, float)
        )

    def test_sqrt_parameter_value_correct(
        self, fn_get_square_root: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Calculate the square root of 144",
            target_output=(
                '{"name":"fn_get_square_root","parameters":{"a":144}}'
            ),
            functions=[fn_get_square_root],
        )
        assert result["parameters"]["a"] == 144  # type: ignore[index]

    def test_sqrt_has_exactly_one_parameter(
        self, fn_get_square_root: FunctionDefinition
    ) -> None:
        result = _run_decoder(
            prompt="Square root of 16",
            target_output=(
                '{"name":"fn_get_square_root","parameters":{"a":16}}'
            ),
            functions=[fn_get_square_root],
        )
        keys = set(result["parameters"].keys())  # type: ignore[union-attr]
        assert keys == {"a"}


# ===========================================================================
# Function selection and skipping
# ===========================================================================

class TestDecoderFunctionHandling:
    """Tests for decoder behaviour across multiple available functions."""

    def test_skips_function_with_unsupported_parameter_type(
        self,
        fn_greet: FunctionDefinition,
        fn_unsupported_type: FunctionDefinition,
    ) -> None:
        llm = CharLevelFakeLLM(
            prompt="Greet alice",
            target_output='{"name":"fn_greet","parameters":{"name":"alice"}}',
        )
        decoder = ConstrainedDecoder(
            available_functions=[fn_greet, fn_unsupported_type],
            llm=llm,
        )
        assert fn_unsupported_type.name in decoder.skipped_functions
        assert fn_greet.name not in decoder.skipped_functions

    def test_selects_string_function_from_mixed_set(
        self,
        fn_greet: FunctionDefinition,
        fn_add_numbers: FunctionDefinition,
        fn_get_square_root: FunctionDefinition,
    ) -> None:
        """
        With three functions available the decoder picks the one whose
        literal matches the highest-logit token sequence.
        """
        target = '{"name":"fn_greet","parameters":{"name":"bob"}}'
        result = _run_decoder(
            prompt="Greet bob",
            target_output=target,
            functions=[fn_greet, fn_add_numbers, fn_get_square_root],
        )
        assert result["name"] == "fn_greet"
        assert result["parameters"]["name"] == "bob"  # type: ignore[index]

    def test_selects_number_function_from_mixed_set(
        self,
        fn_greet: FunctionDefinition,
        fn_add_numbers: FunctionDefinition,
        fn_get_square_root: FunctionDefinition,
    ) -> None:
        target = '{"name":"fn_add_numbers","parameters":{"a":7,"b":8}}'
        result = _run_decoder(
            prompt="Add 7 and 8",
            target_output=target,
            functions=[fn_greet, fn_add_numbers, fn_get_square_root],
        )
        assert result["name"] == "fn_add_numbers"
        assert result["parameters"]["a"] == 7  # type: ignore[index]
        assert result["parameters"]["b"] == 8  # type: ignore[index]

    def test_output_is_always_parseable_json(
        self,
        fn_greet: FunctionDefinition,
        fn_add_numbers: FunctionDefinition,
    ) -> None:
        for prompt, target in [
            (
                "Greet world",
                '{"name":"fn_greet","parameters":{"name":"world"}}',
            ),
            (
                "Add 1 and 2",
                '{"name":"fn_add_numbers","parameters":{"a":1,"b":2}}',
            ),
        ]:
            raw_llm = CharLevelFakeLLM(prompt=prompt, target_output=target)
            decoder = ConstrainedDecoder(
                available_functions=[fn_greet, fn_add_numbers],
                llm=raw_llm,
            )
            ids = [
                ord(c) for c in prompt
                if ord(c) < CharLevelFakeLLM.VOCAB_SIZE
            ]
            raw = decoder.force_json_output(ids)
            parsed = json.loads(raw)
            assert "name" in parsed
            assert "parameters" in parsed

    def test_raises_when_no_supported_functions(
        self, fn_unsupported_type: FunctionDefinition
    ) -> None:
        llm = CharLevelFakeLLM(prompt="x", target_output="")
        with pytest.raises(RuntimeError, match="no supported functions"):
            decoder = ConstrainedDecoder(
                available_functions=[fn_unsupported_type],
                llm=llm,
            )
            decoder.force_json_output([ord("x")])
