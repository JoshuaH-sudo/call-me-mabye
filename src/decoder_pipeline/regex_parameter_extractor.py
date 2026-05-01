import re
from typing import Iterator

KEYWORD_MAP: list[tuple[set[str], str]] = [
    ({"vowel", "vowels"}, "[aeiouAEIOU]"),
    ({"digit", "digits", "number", "numbers"}, r"\d+"),
    ({"letter", "letters", "alpha", "alphabetic"}, "[a-zA-Z]+"),
    ({"word", "words"}, r"\w+"),
    ({"whitespace", "space", "spaces"}, r"\s+"),
    ({"uppercase", "capital", "capitals"}, "[A-Z]+"),
    ({"lowercase"}, "[a-z]+"),
]

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
    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        quoted: list[str] = []
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in quoted:
                quoted.append(cleaned)
        return quoted

    def _extract_literal_target(self, prompt: str) -> str | None:
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
        return re.escape(value)

    def extract_candidates(self, prompt: str) -> Iterator[str]:
        lower = prompt.lower()
        tokens = set(re.findall(r"\w+", lower))

        literal_target = self._extract_literal_target(prompt)
        if literal_target is not None:
            yield literal_target

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

        # Fallback to a short quoted literal when no keyword signal exists.
        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            shortest = min(quoted, key=len)
            longest = max(quoted, key=len)
            if shortest != longest or len(shortest) <= 10:
                yield re.escape(shortest)
                yield from VALID_REGEX_CHARS
                return

        yield from VALID_REGEX_CHARS
