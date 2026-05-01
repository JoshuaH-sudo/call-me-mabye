"""Regex pattern extraction from prompt text.

Infers candidate regular-expression strings from the prompt using two
complementary strategies:

1. **Keyword mapping** — recognises semantic intent words (e.g. "vowel",
   "digit", "uppercase") and maps them to the corresponding character-class
   patterns.
2. **Quoted literal target** — when the prompt names a specific word, text,
   or pattern using quotes, the literal is extracted and escaped so it can
   be used directly as a regex.

Both strategies fall back to a list of valid regex building-block characters
when no stronger signal is found.
"""
import re
from typing import Iterator

# Maps sets of intent keywords to the regex pattern they imply.
# Order matters: the first matching entry wins when multiple keywords appear.
KEYWORD_MAP: list[tuple[set[str], str]] = [
    ({"vowel", "vowels"}, "[aeiouAEIOU]"),
    ({"digit", "digits", "number", "numbers"}, r"\d+"),
    ({"letter", "letters", "alpha", "alphabetic"}, "[a-zA-Z]+"),
    ({"word", "words"}, r"\w+"),
    ({"whitespace", "space", "spaces"}, r"\s+"),
    ({"uppercase", "capital", "capitals"}, "[A-Z]+"),
    ({"lowercase"}, "[a-z]+"),
]

# Individual regex building-block characters and ranges yielded as a fallback
# when no keyword or literal target signal is found in the prompt.
VALID_REGEX_CHARS: list[str] = [
    "a-z",
    "A-Z",
    "0-9",
    ".",
    "^",
    "$",
    "*",
    "+",
    "?",
    "{",
    "}",
    ",",
    "(",
    ")",
    "[",
    "]",
    "|",
    "\\",
]


class RegexParameterExtractor:
    """Infers regex pattern candidates from a user prompt.

    Yields candidates in priority order:

    1. A ``re.escape``-d literal when the prompt names a specific target
       (e.g. *"pattern 'hello'"*).
    2. One or more semantic class patterns when intent keywords are present
       (e.g. *"find all vowels"* → ``[aeiouAEIOU]``).
    3. Raw regex building-block characters as a last resort.
    """

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        """Return non-empty strings found inside single or double quotes.

        Args:
            prompt: The raw user prompt.

        Returns:
            A deduplicated list of quoted string contents in order of
            appearance.  Empty strings and duplicates are omitted.
        """
        quoted: list[str] = []
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in quoted:
                quoted.append(cleaned)
        return quoted

    def _extract_literal_target(self, prompt: str) -> str | None:
        """Extract and escape a quoted literal regex target from *prompt*.

        Looks for patterns like *"word 'hello'"* or *"pattern \"abc\""* where
        the prompt explicitly names a literal target for the regex.

        Args:
            prompt: The raw user prompt.

        Returns:
            A ``re.escape``-d version of the quoted target, or ``None`` if no
            such pattern is found.
        """
        match = re.search(
            (
                r"\b(?:word|text|string|pattern)\b\s+"
                r"(?:\"([^\"\\]+)\"|'([^'\\]+)')"
            ),
            prompt,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        raw = match.group(1) or match.group(2)
        value = raw.strip()
        if not value:
            return None
        # Escape so the literal value is safe to use as a regex pattern.
        return re.escape(value)

    def extract_candidates(self, prompt: str) -> Iterator[str]:
        """Yield regex pattern candidates in priority order.

        Priority
        --------
        1. A ``re.escape``-d literal target (highest confidence).
        2. Semantic class patterns from the keyword map.
        3. A short quoted literal + raw building-block characters (fallback).
        4. Raw building-block characters only (lowest confidence).

        Args:
            prompt: The raw user prompt to extract candidates from.

        Yields:
            Regex pattern strings, deduplicated within each priority tier.
        """
        lower = prompt.lower()
        tokens = set(re.findall(r"\w+", lower))

        # Priority 1: explicit literal target.
        literal_target = self._extract_literal_target(prompt)
        if literal_target is not None:
            yield literal_target

        # Priority 2: keyword-based semantic patterns.
        results: list[str] = []
        for keywords, pattern in KEYWORD_MAP:
            if keywords & tokens:
                results.append(pattern)

        if results:
            seen: set[str] = set()
            for pattern in results:
                if pattern in seen:
                    continue
                seen.add(pattern)
                yield pattern
            return

        # Priority 3: short quoted literal + building blocks when no keywords.
        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            shortest = min(quoted, key=len)
            longest = max(quoted, key=len)
            if shortest != longest or len(shortest) <= 10:
                yield re.escape(shortest)
                yield from VALID_REGEX_CHARS
                return

        # Priority 4: raw building-block characters only.
        yield from VALID_REGEX_CHARS
