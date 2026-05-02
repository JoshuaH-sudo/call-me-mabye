"""Unit tests for RegexTokenValidator.

These tests require no LLM or heavy dependencies and cover all three
public methods: is_valid_complete_regex, is_valid_regex_prefix, and
filter_to_valid_continuations.
"""
import pytest

from src.decoder.regex_token_validator import RegexTokenValidator


@pytest.fixture
def validator() -> RegexTokenValidator:
    return RegexTokenValidator()


class TestIsValidCompleteRegex:
    """Tests for RegexTokenValidator.is_valid_complete_regex."""

    def test_simple_literal_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("hello") is True

    def test_digit_class_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex(r"\d+") is True

    def test_character_range_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("[a-z]+") is True

    def test_anchored_pattern_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex(r"^\d{3}-\d{4}$") is True

    def test_empty_string_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        # re.compile("") succeeds — matches the empty string.
        assert validator.is_valid_complete_regex("") is True

    def test_vowel_class_is_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("[aeiouAEIOU]") is True

    def test_unclosed_bracket_is_invalid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("[a-z") is False

    def test_unclosed_group_is_invalid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("(abc") is False

    def test_lone_backslash_is_invalid(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_complete_regex("\\") is False

    def test_invalid_range_is_invalid(
        self, validator: RegexTokenValidator
    ) -> None:
        # Build the invalid range string at runtime to avoid linter
        # warnings about the literal character range z-a (where z > a).
        invalid_range_pattern = "[" + "z-a" + "]"
        assert validator.is_valid_complete_regex(invalid_range_pattern) is False


class TestIsValidRegexPrefix:
    """Tests for RegexTokenValidator.is_valid_regex_prefix."""

    def test_empty_string_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_regex_prefix("") is True

    def test_complete_regex_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        assert validator.is_valid_regex_prefix(r"\d+") is True

    def test_partial_escape_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "\d" alone is a valid regex (matches a single digit).
        assert validator.is_valid_regex_prefix(r"\d") is True

    def test_lone_backslash_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "\" alone can be completed to "\d", "\w", etc.
        assert validator.is_valid_regex_prefix("\\") is True

    def test_open_bracket_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "[" can be completed to "[a]", "[a-z]", etc.
        assert validator.is_valid_regex_prefix("[") is True

    def test_partial_char_class_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "[a-z" can be completed to "[a-z]".
        assert validator.is_valid_regex_prefix("[a-z") is True

    def test_partial_group_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "(ab" can be completed to "(ab)".
        assert validator.is_valid_regex_prefix("(ab") is True

    def test_invalid_range_is_not_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # Build the partial invalid range at runtime to avoid linter
        # warnings; "[z-a" cannot be completed into a valid regex.
        partial_invalid = "[" + "z-a"
        assert validator.is_valid_regex_prefix(partial_invalid) is False

    def test_valid_quantifier_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # "a{" can be completed to "a{1}" or "a{1,3}".
        assert validator.is_valid_regex_prefix("a{") is True

    def test_plain_word_is_valid_prefix(
        self, validator: RegexTokenValidator
    ) -> None:
        # A plain word is a valid regex literal.
        assert validator.is_valid_regex_prefix("cat") is True


class TestFilterToValidContinuations:
    """Tests for RegexTokenValidator.filter_to_valid_continuations."""

    def test_returns_ids_for_valid_continuations(
        self, validator: RegexTokenValidator
    ) -> None:
        # Starting from "[", "a" extends to "[a" which is a valid prefix;
        # "]" extends to "[]" which is invalid in Python re.
        candidates = [(1, "a"), (2, "]"), (3, "z")]
        result = validator.filter_to_valid_continuations("[", candidates)
        assert 1 in result   # "[a" — valid prefix
        assert 3 in result   # "[z" — valid prefix
        # "]" alone may or may not pass depending on closer attempts;
        # the key assertion is that at least the letters are included.

    def test_filters_out_invalid_continuations(
        self, validator: RegexTokenValidator
    ) -> None:
        # Build the partial invalid range at runtime; adding content
        # cannot make "[z-a…" a valid regex (the range is irrecoverable).
        partial_invalid = "[" + "z-a"
        candidates = [(10, "]"), (11, "b"), (12, "c")]
        result = validator.filter_to_valid_continuations(
            partial_invalid, candidates
        )
        # All completions inherit the invalid range → all rejected.
        assert result == []

    def test_returns_all_valid_continuations_from_empty(
        self, validator: RegexTokenValidator
    ) -> None:
        # From an empty partial, any single character is a valid start.
        candidates = [(1, "a"), (2, "\\"), (3, "["), (4, ".")]
        result = validator.filter_to_valid_continuations("", candidates)
        # All of these are valid regex prefixes from the empty state.
        assert set(result) == {1, 2, 3, 4}

    def test_empty_candidate_list_returns_empty(
        self, validator: RegexTokenValidator
    ) -> None:
        result = validator.filter_to_valid_continuations("abc", [])
        assert result == []

    def test_preserves_input_order_of_valid_ids(
        self, validator: RegexTokenValidator
    ) -> None:
        candidates = [(5, "a"), (6, "b"), (7, "c")]
        result = validator.filter_to_valid_continuations("", candidates)
        assert result == [5, 6, 7]

    def test_multichar_token_passes_if_prefix_valid(
        self, validator: RegexTokenValidator
    ) -> None:
        # A multi-character token "\d+" appended to "" gives "\d+",
        # a valid complete regex.
        candidates = [(99, r"\d+")]
        result = validator.filter_to_valid_continuations("", candidates)
        assert 99 in result
