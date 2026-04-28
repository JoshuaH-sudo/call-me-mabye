"""
Unit tests for StringParameterHandler and NumberParameterHandler.

Each class of tests is self-contained and exercises:
  - Basic acceptance of well-formed values
  - Multi-chunk (token-by-token) consumption
  - Escape-sequence handling (strings)
  - Negative numbers and floats (numbers)
  - Rejection of invalid input
  - Enforcement of the max_length / max_digits safety cap
  - is_complete / is_valid_prefix predicates
"""
from src.decoder_pipeline.parameter_handlers import (
    NumberParameterHandler,
    StringParameterHandler,
)


# ===========================================================================
# StringParameterHandler
# ===========================================================================

class TestStringParameterHandler:
    """Tests for the string parameter type."""

    # -----------------------------------------------------------------------
    # Basic acceptance
    # -----------------------------------------------------------------------

    def test_accepts_simple_string(self) -> None:
        h = StringParameterHandler()
        state, idx = h.consume_chunk(h.initial_state(), '"hello"', 0)
        assert h.is_complete(state)
        assert idx == 7

    def test_accepts_empty_string(self) -> None:
        h = StringParameterHandler()
        state, idx = h.consume_chunk(h.initial_state(), '""', 0)
        assert h.is_complete(state)
        assert idx == 2

    def test_accepts_string_with_spaces(self) -> None:
        h = StringParameterHandler()
        state, idx = h.consume_chunk(h.initial_state(), '"hello world"', 0)
        assert h.is_complete(state)
        assert idx == 13

    def test_accepts_string_with_digits(self) -> None:
        h = StringParameterHandler()
        state, _ = h.consume_chunk(h.initial_state(), '"abc123"', 0)
        assert h.is_complete(state)

    def test_accepts_string_with_special_chars(self) -> None:
        h = StringParameterHandler()
        state, _ = h.consume_chunk(h.initial_state(), '"hello-world_42!"', 0)
        assert h.is_complete(state)

    # -----------------------------------------------------------------------
    # Multi-chunk (simulates token-by-token generation)
    # -----------------------------------------------------------------------

    def test_multi_chunk_opening_then_content_then_close(self) -> None:
        h = StringParameterHandler()
        s = h.initial_state()

        # Token 1: opening quote
        s, _ = h.consume_chunk(s, '"', 0)
        assert not h.is_complete(s)
        assert s["started"]

        # Token 2: string content
        s, _ = h.consume_chunk(s, "hello", 0)
        assert not h.is_complete(s)

        # Token 3: closing quote
        s, idx = h.consume_chunk(s, '"', 0)
        assert h.is_complete(s)
        assert idx == 1

    def test_multi_chunk_single_chars(self) -> None:
        h = StringParameterHandler()
        s = h.initial_state()
        for ch in '"shrek"':
            s, _ = h.consume_chunk(s, ch, 0)
        assert h.is_complete(s)

    def test_chunk_spanning_open_and_content(self) -> None:
        """Token whose text starts with the opening quote."""
        h = StringParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"hello', 0)
        assert not h.is_complete(s)
        assert s["started"]
        assert s["content_length"] == 5

    # -----------------------------------------------------------------------
    # Escape sequences
    # -----------------------------------------------------------------------

    def test_accepts_escaped_quote_inside_string(self) -> None:
        h = StringParameterHandler()
        # "he\"llo" — the \" inside is an escape sequence, not string end
        s, idx = h.consume_chunk(h.initial_state(), '"he\\"llo"', 0)
        assert h.is_complete(s)
        assert idx == 9

    def test_accepts_escaped_backslash(self) -> None:
        h = StringParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), '"a\\\\b"', 0)
        assert h.is_complete(s)

    def test_escaped_char_counts_toward_content_length(self) -> None:
        h = StringParameterHandler(max_length=3)
        s = h.initial_state()
        # Opening quote — not counted
        s, _ = h.consume_chunk(s, '"', 0)
        # Two regular chars (count = 2)
        s, _ = h.consume_chunk(s, "ab", 0)
        assert s["content_length"] == 2
        # Backslash (count = 3, at limit)
        s, _ = h.consume_chunk(s, "\\", 0)
        assert s["content_length"] == 3
        # Escaped char is consumed even though we are at the limit
        s, _ = h.consume_chunk(s, "n", 0)  # the 'n' of '\n'
        assert not h.is_complete(s)

    # -----------------------------------------------------------------------
    # Rejection cases
    # -----------------------------------------------------------------------

    def test_rejects_without_opening_quote(self) -> None:
        h = StringParameterHandler()
        result, _ = h.consume_chunk(h.initial_state(), "hello\"", 0)
        assert result is None

    def test_rejects_control_character_inside_string(self) -> None:
        h = StringParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"', 0)  # open
        result, _ = h.consume_chunk(s, "\x01", 0)
        assert result is None

    def test_rejects_newline_inside_string(self) -> None:
        h = StringParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"', 0)
        result, _ = h.consume_chunk(s, "\n", 0)
        assert result is None

    # -----------------------------------------------------------------------
    # max_length enforcement
    # -----------------------------------------------------------------------

    def test_accepts_string_exactly_at_max_length(self) -> None:
        h = StringParameterHandler(max_length=5)
        # "hello" is exactly 5 content chars — must be accepted.
        s, _ = h.consume_chunk(h.initial_state(), '"hello"', 0)
        assert h.is_complete(s)

    def test_rejects_content_beyond_max_length(self) -> None:
        h = StringParameterHandler(max_length=5)
        # First 5 chars accepted, 6th non-quote char must be rejected.
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"hello', 0)   # 5 content chars
        assert s["content_length"] == 5
        result, _ = h.consume_chunk(s, "o", 0)   # 6th char, not a quote
        assert result is None

    def test_closing_quote_always_accepted_at_max_length(self) -> None:
        h = StringParameterHandler(max_length=5)
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"hello', 0)   # at limit
        # Closing quote must still be accepted.
        s, idx = h.consume_chunk(s, '"', 0)
        assert h.is_complete(s)
        assert idx == 1

    def test_closing_quote_always_accepted_beyond_max_length(self) -> None:
        """Escaped char may push content_length past max; close still works."""
        h = StringParameterHandler(max_length=3)
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"ab\\', 0)    # 3 chars (2 + backslash)
        s, _ = h.consume_chunk(s, "n", 0)        # escaped char, count = 4
        s, _ = h.consume_chunk(s, '"', 0)
        assert h.is_complete(s)

    # -----------------------------------------------------------------------
    # is_valid_prefix
    # -----------------------------------------------------------------------

    def test_is_valid_prefix_for_in_progress_state(self) -> None:
        h = StringParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, '"hello', 0)
        assert h.is_valid_prefix(s)

    def test_is_valid_prefix_for_initial_state(self) -> None:
        h = StringParameterHandler()
        assert h.is_valid_prefix(h.initial_state())


# ===========================================================================
# NumberParameterHandler
# ===========================================================================

class TestNumberParameterHandler:
    """Tests for the number parameter type."""

    # -----------------------------------------------------------------------
    # Basic acceptance
    # -----------------------------------------------------------------------

    def test_accepts_integer_terminated_by_comma(self) -> None:
        h = NumberParameterHandler()
        s, idx = h.consume_chunk(h.initial_state(), "42,", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "42"
        assert idx == 2  # comma NOT consumed

    def test_accepts_integer_terminated_by_closing_brace(self) -> None:
        h = NumberParameterHandler()
        s, idx = h.consume_chunk(h.initial_state(), "42}", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "42"
        assert idx == 2

    def test_accepts_single_digit(self) -> None:
        h = NumberParameterHandler()
        s, idx = h.consume_chunk(h.initial_state(), "3}", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "3"

    def test_accepts_zero(self) -> None:
        h = NumberParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), "0}", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "0"

    def test_accepts_negative_integer(self) -> None:
        h = NumberParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), "-5,", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "-5"

    def test_accepts_float(self) -> None:
        h = NumberParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), "3.14,", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "3.14"

    def test_accepts_scientific_notation(self) -> None:
        h = NumberParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), "1e10,", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "1e10"

    def test_accepts_large_integer(self) -> None:
        h = NumberParameterHandler()
        s, _ = h.consume_chunk(h.initial_state(), "265}", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "265"

    # -----------------------------------------------------------------------
    # Multi-chunk (token-by-token)
    # -----------------------------------------------------------------------

    def test_multi_chunk_digits_then_terminator(self) -> None:
        h = NumberParameterHandler()
        s = h.initial_state()

        # Token 1: digits only — not yet complete
        s, _ = h.consume_chunk(s, "42", 0)
        assert not h.is_complete(s)
        assert s["buffer"] == "42"

        # Token 2: terminator character
        s, idx = h.consume_chunk(s, "}", 0)
        assert h.is_complete(s)
        assert idx == 0  # } not consumed

    def test_multi_chunk_single_digit_tokens(self) -> None:
        h = NumberParameterHandler()
        s = h.initial_state()
        for ch in "345":
            s, _ = h.consume_chunk(s, ch, 0)
        assert not h.is_complete(s)
        assert s["buffer"] == "345"
        s, _ = h.consume_chunk(s, ",", 0)
        assert h.is_complete(s)

    def test_chunk_containing_both_digits_and_terminator(self) -> None:
        """Token text like '42}' should complete the number at '}'."""
        h = NumberParameterHandler()
        s, idx = h.consume_chunk(h.initial_state(), "42}", 0)
        assert h.is_complete(s)
        assert idx == 2

    # -----------------------------------------------------------------------
    # Rejection cases
    # -----------------------------------------------------------------------

    def test_rejects_non_numeric_start(self) -> None:
        h = NumberParameterHandler()
        result, _ = h.consume_chunk(h.initial_state(), "abc}", 0)
        assert result is None

    def test_non_number_char_terminates_valid_buffer(self) -> None:
        """
        A non-number character after a valid digit buffer completes the
        number and leaves the character unconsumed for the next segment.
        It does NOT reject the chunk — rejection only happens when the
        buffer is not yet a valid number at the point of termination.
        """
        h = NumberParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "4", 0)
        result, idx = h.consume_chunk(s, "x", 0)
        assert h.is_complete(result)
        assert result["buffer"] == "4"
        assert idx == 0  # x is NOT consumed; it belongs to the next segment

    def test_rejects_bare_minus_as_complete(self) -> None:
        """'-' alone is not a valid JSON number."""
        h = NumberParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "-", 0)
        result, _ = h.consume_chunk(s, "}", 0)
        assert result is None

    def test_rejects_bare_dot_as_complete(self) -> None:
        h = NumberParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, ".", 0)
        result, _ = h.consume_chunk(s, "}", 0)
        assert result is None

    # -----------------------------------------------------------------------
    # max_digits enforcement
    # -----------------------------------------------------------------------

    def test_accepts_number_exactly_at_max_digits(self) -> None:
        h = NumberParameterHandler(max_digits=3)
        s, _ = h.consume_chunk(h.initial_state(), "123}", 0)
        assert h.is_complete(s)
        assert s["buffer"] == "123"

    def test_rejects_digit_token_when_buffer_is_full(self) -> None:
        """After max_digits with a valid buffer, digit tokens are rejected."""
        h = NumberParameterHandler(max_digits=3)
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "123", 0)   # exactly at limit
        # A pure-digit token must be rejected so the decoder picks a
        # terminator instead.
        result, _ = h.consume_chunk(s, "4", 0)
        assert result is None

    def test_terminator_accepted_after_max_digits(self) -> None:
        """After max_digits the terminator token must still be allowed."""
        h = NumberParameterHandler(max_digits=3)
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "123", 0)
        s, idx = h.consume_chunk(s, "}", 0)
        assert h.is_complete(s)
        assert idx == 0  # } not consumed

    def test_large_default_max_digits_allows_reasonable_values(self) -> None:
        """Default max_digits (15) allows typical numbers used in tests."""
        h = NumberParameterHandler()
        for value in ["265", "345", "16", "144", "3.14", "-42", "1e10"]:
            s, _ = h.consume_chunk(h.initial_state(), value + "}", 0)
            assert h.is_complete(s), f"expected complete for {value!r}"

    # -----------------------------------------------------------------------
    # is_valid_prefix
    # -----------------------------------------------------------------------

    def test_is_valid_prefix_for_empty_buffer(self) -> None:
        h = NumberParameterHandler()
        assert h.is_valid_prefix(h.initial_state())

    def test_is_valid_prefix_after_digits(self) -> None:
        h = NumberParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "42", 0)
        assert h.is_valid_prefix(s)

    def test_is_valid_prefix_after_minus(self) -> None:
        """'-' alone is a valid prefix (number not yet complete)."""
        h = NumberParameterHandler()
        s = h.initial_state()
        s, _ = h.consume_chunk(s, "-", 0)
        assert h.is_valid_prefix(s)

    def test_is_not_valid_prefix_for_invalid_buffer(self) -> None:
        """'--' is neither a valid number nor a valid prefix."""
        h = NumberParameterHandler()
        # Force a bad buffer directly to test the predicate in isolation.
        bad_state = {"buffer": "--", "complete": False}
        assert not h.is_valid_prefix(bad_state)
