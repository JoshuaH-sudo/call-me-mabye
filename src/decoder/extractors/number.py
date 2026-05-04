"""Numeric parameter extraction from prompt text.

Supports two number representations:

1. **Numeric literals** — any integer or decimal that can be matched by the
   pattern ``-?\\d+(\\.\\d+)?``, including negative values.
2. **English word numbers** — phrases such as "twenty-three", "one hundred
   and five", "negative three point one four", including compound tokens
   like "twentythree" that the tokeniser may not split.

Both sources are merged and sorted by their character position in the
original prompt so that the ordering of candidate values mirrors the
natural reading order of the text.
"""
import re

from .shared_keywords import QUESTION_INTENT_WORDS


def _find_question_segment(prompt: str) -> str:
    """Return the most intent-bearing sentence from *prompt*.

    When a prompt contains background context followed by an actual question
    or instruction (e.g. "10 + 5 = 15 and 5+10 also equals 15.  What is the
    sum of 2 and 3?"), this helper isolates the question/instruction part so
    that number extraction can focus on the relevant numbers rather than the
    noise in the preamble.

    Algorithm
    ---------
    1. Identify all character positions inside quoted substrings so that
       sentence-boundary punctuation within quotes is never treated as a split
       point.
    2. Walk the prompt character-by-character to find sentence boundaries
       (a ``.``, ``?``, or ``!`` followed by whitespace that does not fall
       inside a quoted span), collecting each sentence as a substring of the
       original prompt.
    3. Prefer the last sentence that either ends with ``?`` or whose first
       token is a recognised question/intent word.
    4. Fall back to the last sentence if no sentence matches those criteria,
       or return *prompt* unchanged when no boundary is found.

    Args:
        prompt: The raw user prompt to segment.

    Returns:
        The question or instruction sentence, stripped of leading/trailing
        whitespace.
    """
    # Step 1: collect the set of character positions that are inside quotes.
    quoted_positions: set[int] = set()
    for m in re.finditer(r'"[^"]*"|\'[^\']*\'', prompt):
        quoted_positions.update(range(m.start(), m.end()))

    # Step 2: find sentence-boundary split positions — a sentence end is a
    # `.`, `?`, or `!` that is NOT inside a quoted span, followed by at least
    # one whitespace character.
    split_positions: list[int] = []
    for m in re.finditer(r'[.?!]\s+', prompt):
        if m.start() not in quoted_positions:
            # Record the start of the *next* sentence (after the whitespace).
            split_positions.append(m.end())

    if not split_positions:
        return prompt.strip()

    # Step 3: build sentence list from the original prompt.
    starts = [0] + split_positions
    sentences: list[str] = []
    for i, start in enumerate(starts):
        end = split_positions[i] if i < len(split_positions) else len(prompt)
        segment = prompt[start:end].strip()
        if segment:
            sentences.append(segment)

    if not sentences:
        return prompt.strip()

    if len(sentences) == 1:
        return sentences[0]

    # Step 4: score each sentence.  Prefer later sentences that end with '?'
    # or begin with a recognised question/intent word.
    best_sentence = sentences[-1]
    best_score = -1
    for idx, sentence in enumerate(sentences):
        score = idx  # later sentences are preferred over earlier ones
        if sentence.rstrip().endswith("?"):
            score += len(sentences) * 2  # strong signal
        first_word = re.match(r"\w+", sentence.lower())
        if first_word and first_word.group() in QUESTION_INTENT_WORDS:
            score += len(sentences)  # moderate signal
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
"""English words for the numbers 0–19."""

TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
"""English words for multiples of ten from 20–90."""

SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
"""Scale multipliers used in English number names."""

# Words that indicate a negative sign when they precede a number phrase.
SIGNED_WORDS = {"minus", "negative"}

# All tokens that are part of a valid English number expression.
VALID_WORD_TOKENS = (
    set(UNITS) | set(TENS) | set(SCALES) | {"and", "point"} | SIGNED_WORDS
)

# Tokens that can be parts of a compound number word (e.g. "twentythree").
COMPOUND_NUMBER_PARTS = set(UNITS) | set(TENS) | set(SCALES)


class NumberParameterExtractor:
    """Extracts numeric parameter candidates from a prompt string.

    Handles both numeric literals (``-7``, ``3.14``) and English word
    numbers (``"twenty-three"``, ``"one hundred and five"``,
    ``"negative three point one four"``).
    """

    def _normalize_value(self, value: int | float) -> int | float:
        """Convert integral floats to ints for cleaner intermediate values."""
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _extract_numeric_literal_mentions(
        self,
        prompt: str,
    ) -> list[tuple[int, int | float]]:
        """Extract numeric literals like -7 and 3.14 with source positions."""
        mentions: list[tuple[int, int | float]] = []
        for match in re.finditer(r"-?\d+(?:\.\d+)?", prompt):
            raw_value = float(match.group(0))
            mentions.append((match.start(), self._normalize_value(raw_value)))
        return mentions

    def _split_compound_number_token(self, token: str) -> list[str] | None:
        """Split tokens like "twentythree" into known number word parts."""
        memo: dict[str, list[str] | None] = {}

        def solve(remaining: str) -> list[str] | None:
            if remaining == "":
                return []
            if remaining in memo:
                return memo[remaining]

            for part in COMPOUND_NUMBER_PARTS:
                if remaining.startswith(part):
                    tail = solve(remaining[len(part):])
                    if tail is not None:
                        memo[remaining] = [part] + tail
                        return memo[remaining]

            memo[remaining] = None
            return None

        parts = solve(token)
        if parts is None or len(parts) <= 1:
            return None
        return parts

    def _tokenize_word_stream(self, prompt: str) -> list[tuple[str, int]]:
        """Tokenize words and keep each start position in the prompt."""
        tokens_with_positions: list[tuple[str, int]] = []
        for word_match in re.finditer(r"[a-zA-Z]+", prompt.lower()):
            token = word_match.group(0)
            token_start = word_match.start()

            if token in VALID_WORD_TOKENS:
                tokens_with_positions.append((token, token_start))
                continue

            split_parts = self._split_compound_number_token(token)
            if split_parts is None:
                tokens_with_positions.append((token, token_start))
                continue

            for part in split_parts:
                tokens_with_positions.append((part, token_start))

        return tokens_with_positions

    def _is_numeric_word(self, token: str) -> bool:
        return token in (set(UNITS) | set(TENS) | set(SCALES))

    def _split_number_groups(self, tokens: list[str]) -> list[list[str]]:
        """Split a numeric phrase into separate groups representing numbers."""
        groups: list[list[str]] = []
        current: list[str] = []

        for index, token in enumerate(tokens):
            if token == "and" and current:
                next_token = (
                    tokens[index + 1] if index + 1 < len(tokens) else ""
                )

                # Keep "and" inside scaled numbers like
                # "one hundred and five".
                if any(tok in SCALES for tok in current):
                    current.append(token)
                    continue

                # Split at "and" for cases like
                # "twenty-three and one hundred and five".
                if self._is_numeric_word(next_token):
                    groups.append(current)
                    current = []
                    continue

            current.append(token)

        if current:
            groups.append(current)
        return groups

    def _parse_word_number(self, tokens: list[str]) -> int | float | None:
        """Parse one list of number words into a numeric value."""
        if not tokens:
            return None

        sign = 1
        if tokens[0] in SIGNED_WORDS:
            sign = -1
            tokens = tokens[1:]
        if not tokens:
            return None

        if "point" in tokens:
            point_index = tokens.index("point")
            integer_tokens = tokens[:point_index]
            fractional_tokens = tokens[point_index + 1:]
            if not fractional_tokens:
                return None

            integer_raw = self._parse_word_number(integer_tokens)
            integer_value = 0 if integer_raw is None else int(abs(integer_raw))

            fractional_digits: list[str] = []
            for token in fractional_tokens:
                if token == "and":
                    continue
                if token not in UNITS or UNITS[token] > 9:
                    return None
                fractional_digits.append(str(UNITS[token]))

            if not fractional_digits:
                return None

            fractional_part = float("0." + "".join(fractional_digits))
            return sign * (integer_value + fractional_part)

        total = 0
        current = 0
        consumed_numeric_token = False

        for token in tokens:
            if token == "and":
                continue

            if token in UNITS:
                current += UNITS[token]
                consumed_numeric_token = True
                continue

            if token in TENS:
                current += TENS[token]
                consumed_numeric_token = True
                continue

            if token == "hundred":
                current = max(1, current) * 100
                consumed_numeric_token = True
                continue

            if token in {"thousand", "million", "billion", "trillion"}:
                total += max(1, current) * SCALES[token]
                current = 0
                consumed_numeric_token = True
                continue

            return None

        if not consumed_numeric_token:
            return None

        return sign * (total + current)

    def _extract_word_number_mentions(
        self,
        prompt: str,
    ) -> list[tuple[int, int | float]]:
        """Extract number mentions expressed as words with source positions."""
        mentions: list[tuple[int, int | float]] = []
        tokens_with_positions = self._tokenize_word_stream(prompt)

        token_index = 0
        while token_index < len(tokens_with_positions):
            token = tokens_with_positions[token_index][0]
            if token not in VALID_WORD_TOKENS:
                token_index += 1
                continue

            end_index = token_index
            while end_index < len(tokens_with_positions):
                end_token = tokens_with_positions[end_index][0]
                if end_token not in VALID_WORD_TOKENS:
                    break
                end_index += 1

            token_slice = tokens_with_positions[token_index:end_index]
            phrase_tokens = [tok for tok, _ in token_slice]
            phrase_positions = [pos for _, pos in token_slice]

            cursor = 0
            for group_tokens in self._split_number_groups(phrase_tokens):
                group_len = len(group_tokens)
                group_start = phrase_positions[cursor]
                cursor += group_len

                parsed_value = self._parse_word_number(group_tokens)
                if parsed_value is None:
                    continue

                mentions.append(
                    (group_start, self._normalize_value(parsed_value))
                )

            token_index = end_index

        return mentions

    def extract_candidates(self, prompt: str) -> list[float]:
        """Extract number candidates from numeric literals and word numbers.

        Supports:
        - Integer and floating-point literals (e.g., 42, 3.14)
        - English word numbers (e.g., "twenty-three", "one hundred and fifty")
        - Negative and decimal word numbers

        Args:
            prompt: The user prompt to extract candidates from.

        Returns:
            A list of numeric candidates ordered by appearance in the prompt.
            Returns an empty list when no numeric mention is found.
        """
        # Step 1: collect number literals like "-7" and "3.5".
        mentions = self._extract_numeric_literal_mentions(prompt)

        # Step 2: collect number words like "twenty-three" and
        # "one hundred and five".
        mentions.extend(self._extract_word_number_mentions(prompt))

        # Step 3: order every mention by its position in the original prompt.
        mentions.sort(key=lambda item: item[0])

        # Step 4: emit normalized numeric values as floats, preserving order.
        ordered_values = [float(value) for _, value in mentions]

        if not ordered_values:
            return []
        return ordered_values

    def extract_candidates_from_question_segment(
        self, prompt: str
    ) -> list[float]:
        """Extract numbers only from the question/intent part of *prompt*.

        Uses :func:`_find_question_segment` to isolate the sentence that
        carries the actual instruction before delegating to
        :meth:`extract_candidates`.  This prevents numeric context that
        appears in preamble sentences from polluting the candidate set.

        Args:
            prompt: The full user prompt, which may contain background
                context sentences before the operative question.

        Returns:
            A list of numeric candidates extracted from the question segment
            only, ordered by appearance.  Falls back to the full-prompt
            extraction when the question segment yields no candidates.
        """
        question_segment = _find_question_segment(prompt)
        return self.extract_candidates(question_segment)
