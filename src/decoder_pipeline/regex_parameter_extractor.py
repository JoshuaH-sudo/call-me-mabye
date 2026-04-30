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
    def extract_candidates(self, prompt: str) -> Iterator[str]:
        lower = prompt.lower()
        tokens = set(re.findall(r"\w+", lower))

        # Extract short quoted literals (single or double quotes) from the prompt.
        # A short quoted string (≤ 20 chars, not the longest quoted string) is likely
        # a literal regex target rather than the source sentence.
        quoted: list[str] = re.findall(r"['\"]([^'\"]{1,20})['\"]", prompt)
        if quoted:
            # Prefer the shortest quoted string that looks like a word/pattern target.
            shortest = min(quoted, key=len)
            longest = max(quoted, key=len)
            # Only use a quoted literal if it's not the same as the longest quote
            # (i.e., it's a short target, not the source sentence).
            if shortest != longest or len(shortest) <= 10:
                yield (
                    re.escape(shortest)
                    if re.escape(shortest) == shortest
                    else shortest
                )
                yield from VALID_REGEX_CHARS
                return

        results: list[str] = []
        for keywords, pattern in KEYWORD_MAP:
            if keywords & tokens:
                results.append(pattern)

        if results:
            yield from results
            return

        yield from VALID_REGEX_CHARS
