"""Integration tests for RegexConstrainedDecoder.

The LLM is replaced by a mock that biases the logit distribution toward
a pre-determined sequence of characters, allowing the decoder behaviour
to be verified without loading the real model.
"""
import re
from unittest.mock import MagicMock

import pytest

from src.decoder.regex_decoder import RegexConstrainedDecoder


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 300  # covers all relevant Unicode code points used in tests


def _make_regex_llm_mock(target_regex: str) -> MagicMock:
    """Return a mock Small_LLM_Model that drives toward *target_regex*.

    Encoding uses character ordinals (mod ``_VOCAB_SIZE``) so that
    ``decode([ord(c) % _VOCAB_SIZE])`` reconstructs the original
    character.  The logit function assigns score 100.0 to the next
    expected character and 0.0 to all others, guaranteeing that the
    constrained decoder always selects the correct token.

    After the target regex characters are emitted the mock returns a
    high logit for ``\\n`` (ordinal 10) to signal end-of-sequence.
    """
    # Target sequence ends with a newline to signal end-of-generation.
    target_chars = list(target_regex) + ["\n"]
    # Capture the length of the initial encoded prompt so the step index
    # can be inferred from ``len(prefix_ids) - initial_len``.
    initial_len: list[int] = []

    def fake_encode(text: str) -> MagicMock:
        ids = [ord(c) % _VOCAB_SIZE for c in text]
        if not initial_len:
            initial_len.append(len(ids))
        tensor = MagicMock()
        tensor.__getitem__ = lambda self, idx: MagicMock(
            tolist=lambda: ids
        )
        return tensor

    def fake_decode(ids: list[int]) -> str:
        if not ids:
            return ""
        return chr(ids[0] % _VOCAB_SIZE)

    def fake_logits(prefix_ids: list[int]) -> list[float]:
        logits = [0.0] * _VOCAB_SIZE
        if not initial_len:
            return logits
        step = len(prefix_ids) - initial_len[0]
        if 0 <= step < len(target_chars):
            tid = ord(target_chars[step]) % _VOCAB_SIZE
            if 0 <= tid < _VOCAB_SIZE:
                logits[tid] = 100.0
        return logits

    llm = MagicMock()
    llm.encode.side_effect = fake_encode
    llm.decode.side_effect = fake_decode
    llm.get_logits_from_input_ids.side_effect = fake_logits
    return llm


def _make_invalid_then_valid_mock(
    invalid_first: str, valid_regex: str
) -> MagicMock:
    """Mock that first tries an invalid token, then emits *valid_regex*.

    The first top-K token per step is always the invalid character given
    by *invalid_first* (which should not pass the regex prefix filter),
    while the second token is the correct character from *valid_regex*.
    This verifies that the decoder correctly skips invalid continuations.
    """
    target_chars = list(valid_regex) + ["\n"]
    initial_len: list[int] = []

    def fake_encode(text: str) -> MagicMock:
        ids = [ord(c) % _VOCAB_SIZE for c in text]
        if not initial_len:
            initial_len.append(len(ids))
        tensor = MagicMock()
        tensor.__getitem__ = lambda self, idx: MagicMock(
            tolist=lambda: ids
        )
        return tensor

    def fake_decode(ids: list[int]) -> str:
        if not ids:
            return ""
        return chr(ids[0] % _VOCAB_SIZE)

    def fake_logits(prefix_ids: list[int]) -> list[float]:
        logits = [0.0] * _VOCAB_SIZE
        if not initial_len:
            return logits
        step = len(prefix_ids) - initial_len[0]
        # Give the *invalid* character a high score (90) and the correct
        # character a slightly lower score (80).  The decoder must skip
        # the invalid one and select the correct one.
        invalid_tid = ord(invalid_first) % _VOCAB_SIZE
        logits[invalid_tid] = 90.0
        if 0 <= step < len(target_chars):
            correct_tid = ord(target_chars[step]) % _VOCAB_SIZE
            if correct_tid != invalid_tid:
                logits[correct_tid] = 80.0
        return logits

    llm = MagicMock()
    llm.encode.side_effect = fake_encode
    llm.decode.side_effect = fake_decode
    llm.get_logits_from_input_ids.side_effect = fake_logits
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateRegexOutputValidity:
    """Verify that generate_regex always returns a compilable regex."""

    @pytest.mark.parametrize(
        "target",
        [
            r"\d+",
            "[aeiouAEIOU]",
            r"\w+",
            "[a-z]+",
            r"[A-Z]+",
        ],
    )
    def test_output_compiles_as_regex(self, target: str) -> None:
        llm = _make_regex_llm_mock(target)
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("match digits")
        assert re.compile(result) is not None, (
            f"result {result!r} is not a valid regex"
        )

    def test_digit_pattern_generated(self) -> None:
        llm = _make_regex_llm_mock(r"\d+")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("replace all numbers")
        assert result == r"\d+"

    def test_vowel_class_generated(self) -> None:
        llm = _make_regex_llm_mock("[aeiouAEIOU]")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("replace all vowels")
        assert result == "[aeiouAEIOU]"

    def test_word_class_generated(self) -> None:
        llm = _make_regex_llm_mock(r"\w+")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("match word characters")
        assert result == r"\w+"


class TestInvalidTokensSkipped:
    """Verify that tokens that break regex validity are never selected."""

    def test_invalid_first_token_is_skipped(self) -> None:
        # "[z-a]" is an invalid regex (bad range). The mock puts the
        # first character of this range ("z") at the top of the logit
        # distribution but the correct character ("\") at position 2.
        # The decoder should skip "z" and select "\".
        llm = _make_invalid_then_valid_mock(
            invalid_first="z",
            valid_regex=r"\d+",
        )
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("match digits")
        # The result must be a valid regex regardless.
        assert re.compile(result) is not None

    def test_newline_token_is_not_appended_to_output(self) -> None:
        llm = _make_regex_llm_mock(r"\d+")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("replace numbers")
        assert "\n" not in result


class TestFallbackBehaviour:
    """Verify graceful degradation when the LLM cannot produce a valid regex."""

    def test_fallback_returns_match_all_regex(self) -> None:
        # Simulate a broken LLM that always returns zero logits →
        # no continuation will ever be selected → safe fallback kicks in.
        llm = MagicMock()

        def zero_encode(text: str) -> MagicMock:
            ids = [0]
            tensor = MagicMock()
            tensor.__getitem__ = lambda self, idx: MagicMock(
                tolist=lambda: ids
            )
            return tensor

        def zero_decode(ids: list[int]) -> str:
            # Return newline for all tokens → stop signal immediately,
            # but current_partial is empty → ".*" fallback.
            return "\n"

        def zero_logits(prefix_ids: list[int]) -> list[float]:
            return [0.0] * _VOCAB_SIZE

        llm.encode.side_effect = zero_encode
        llm.decode.side_effect = zero_decode
        llm.get_logits_from_input_ids.side_effect = zero_logits

        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("replace all digits")
        # The fallback must produce a compilable regex.
        assert re.compile(result) is not None
        assert result == ".*"


class TestTrailingRepetitionDetection:
    """Unit tests for _detect_trailing_repetition helper."""

    def _make_decoder(self) -> RegexConstrainedDecoder:
        # The LLM mock is never called in these tests.
        return RegexConstrainedDecoder(llm=MagicMock())

    def test_doubled_char_class_detected(self) -> None:
        decoder = self._make_decoder()
        result = decoder._detect_trailing_repetition(
            "[aeiouAEIOU][aeiouAEIOU]"
        )
        assert result == "[aeiouAEIOU]"

    def test_doubled_digit_shorthand_detected(self) -> None:
        decoder = self._make_decoder()
        result = decoder._detect_trailing_repetition(r"\d+\d+")
        assert result == r"\d+"

    def test_no_detection_for_distinct_segments(self) -> None:
        # [a-z][0-9] has different left and right halves.
        decoder = self._make_decoder()
        result = decoder._detect_trailing_repetition("[a-z][0-9]")
        assert result is None

    def test_no_detection_when_prefix_empty(self) -> None:
        # "aa" has only 2 chars; with min_segment=2 the loop range is
        # range(1, 1, -1) which is empty — strings shorter than
        # 2 * min_segment (4 chars) cannot contain a detectable doubled
        # segment, so no repetition is possible.
        decoder = self._make_decoder()
        result = decoder._detect_trailing_repetition("aa")
        assert result is None

    def test_doubled_shorthand_with_no_quantifier_detected(self) -> None:
        # \d\d doubles \d (a valid complete regex) → returns \d.
        decoder = self._make_decoder()
        result = decoder._detect_trailing_repetition(r"\d\d")
        assert result == r"\d"

    def test_no_detection_for_single_valid_regex(self) -> None:
        decoder = self._make_decoder()
        assert decoder._detect_trailing_repetition("[a-z]+") is None

    def test_no_detection_for_intentional_extension(self) -> None:
        # "[a-z][A-Z]" has two distinct halves — not a repetition.
        decoder = self._make_decoder()
        assert decoder._detect_trailing_repetition("[a-z][A-Z]") is None


class TestLookaheadAndBackreferenceGeneration:
    """Verify that lookahead and backreference patterns can be generated."""

    def test_positive_lookahead_compiles(self) -> None:
        # Mock drives toward a simple positive lookahead.
        llm = _make_regex_llm_mock(r"(?=\d)")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("position before a digit")
        assert re.compile(result) is not None

    def test_negative_lookahead_compiles(self) -> None:
        llm = _make_regex_llm_mock(r"(?!\d)")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("position not before a digit")
        assert re.compile(result) is not None

    def test_duplicate_word_pattern_generated(self) -> None:
        # The canonical duplicate-word pattern uses a capturing group
        # with a positive lookahead and a backreference.  The validator
        # must accept every prefix so the decoder can generate it fully.
        target = r"\b(\w+)\b(?=.*\b\1\b)"
        llm = _make_regex_llm_mock(target)
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("find duplicate words")
        assert result == target
        # Confirm the pattern actually matches duplicate words.
        assert re.search(result, "the cat sat on the mat") is not None
        assert re.search(result, "one two one") is not None
        assert re.search(result, "no repeats here") is None


class TestRepetitionGuardIntegration:
    """Verify the repetition guard fires during generate_regex."""

    def test_doubled_char_class_truncated(self) -> None:
        # The mock drives toward "[aeiouAEIOU][aeiouAEIOU]".
        # The repetition guard should intercept and return "[aeiouAEIOU]".
        llm = _make_regex_llm_mock("[aeiouAEIOU][aeiouAEIOU]")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("replace all vowels")
        assert result == "[aeiouAEIOU]"

    def test_single_valid_regex_not_truncated(self) -> None:
        # When the mock drives toward a pattern without repetition,
        # the guard must not fire and the full regex must be returned.
        llm = _make_regex_llm_mock(r"\d+")
        decoder = RegexConstrainedDecoder(llm=llm)
        result = decoder.generate_regex("match digits")
        assert result == r"\d+"


class TestConfigurableParameters:
    """Verify that max_tokens and top_k can be customised."""

    def test_custom_max_tokens_respected(self) -> None:
        # Use a very short regex so max_tokens=5 is enough.
        llm = _make_regex_llm_mock(".")
        decoder = RegexConstrainedDecoder(llm=llm, max_tokens=5)
        result = decoder.generate_regex("any character")
        assert re.compile(result) is not None

    def test_custom_top_k_respected(self) -> None:
        llm = _make_regex_llm_mock(r"\d+")
        # Deliberately low top_k; the correct token should still win.
        decoder = RegexConstrainedDecoder(llm=llm, top_k=50)
        result = decoder.generate_regex("match digits")
        assert re.compile(result) is not None
