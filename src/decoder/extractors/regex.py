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

# Commonly requested end-to-end regexes, adapted from documentation examples.
COMMON_REGEX_PATTERNS: dict[str, str] = {
    "whole_number": r"^\d+$",
    "decimal_number": r"^\d*\.\d+$",
    "signed_whole_or_decimal": r"^[-+]?\d*(\.\d+)?$",
    "alphanumeric_no_space": r"^[a-zA-Z0-9]+$",
    "alphanumeric_with_space": r"^[a-zA-Z0-9 ]*$",
    "email": r"^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*$",
    "date_yyyy_mm_dd": (
        r"^(19|20)\d\d([- /.])(0[1-9]|1[012])"
        r"\2(0[1-9]|[12][0-9]|3[01])$"
    ),
    "date_mm_dd_yyyy": (
        r"^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])"
        r"[- /.](19|20)\d\d$"
    ),
    "date_dd_mm_yyyy": (
        r"^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])"
        r"[- /.](19|20)\d\d$"
    ),
    "time_hh_mm_12h_ampm": (
        r"^((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]))$"
    ),
    "time_hh_mm_24h": r"^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
    "time_hh_mm_ss_24h": (
        r"^(?:[01]\d|2[0123]):(?:[012345]\d):(?:[012345]\d)$"
    ),
    "duplicates": r"(\b\w+\b)(?=.*\b\1\b)",
    "phone_international_with_separators": (
        r"^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$"
    ),
    "phone_us_only_with_separators": (
        r"^(\+0?1\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$"
    ),
    "phone_unformatted": (
        r"^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
    ),
    "ssn_with_dashes": (
        r"^(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})"
        r"\d{3}-(?!00)\d{2}-(?!0{4})\d{4}$"
    ),
    "ssn_without_dashes": (
        r"^(?!219099999|078051120)(?!666|000|9\d{2})"
        r"\d{3}(?!00)\d{2}(?!0{4})\d{4}$"
    ),
    "zip_code": r"^\d{5}([\-]?\d{4})?$",
}

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

    def _extract_common_scenario_patterns(self, prompt: str) -> list[str]:
        """Infer full regexes for common validation/search scenarios."""
        lower = prompt.lower()
        results: list[str] = []

        if re.search(r"\b(email|e-mail)\b", lower):
            results.append(COMMON_REGEX_PATTERNS["email"])

        if re.search(r"\b(alphanumeric|alpha numeric)\b", lower):
            if re.search(r"\b(with|allow)\s+spaces?\b", lower):
                results.append(
                    COMMON_REGEX_PATTERNS["alphanumeric_with_space"]
                )
            else:
                results.append(COMMON_REGEX_PATTERNS["alphanumeric_no_space"])

        if re.search(r"\b(zip\s*code|zipcode|postal\s*code)\b", lower):
            results.append(COMMON_REGEX_PATTERNS["zip_code"])

        if re.search(r"\b(ssn|social\s+security)\b", lower):
            if re.search(r"\b(without|no)\s+dashes?\b", lower):
                results.append(COMMON_REGEX_PATTERNS["ssn_without_dashes"])
            elif re.search(r"\bwith\s+dashes?\b", lower):
                results.append(COMMON_REGEX_PATTERNS["ssn_with_dashes"])
            else:
                results.extend(
                    [
                        COMMON_REGEX_PATTERNS["ssn_with_dashes"],
                        COMMON_REGEX_PATTERNS["ssn_without_dashes"],
                    ]
                )

        if re.search(r"\b(phone|telephone|mobile|cell)\b", lower):
            if re.search(r"\b(unformatted|no\s+separators?)\b", lower):
                results.append(COMMON_REGEX_PATTERNS["phone_unformatted"])
            elif re.search(r"\b(us\s+only)\b", lower):
                results.append(
                    COMMON_REGEX_PATTERNS["phone_us_only_with_separators"]
                )
            else:
                results.append(
                    COMMON_REGEX_PATTERNS[
                        "phone_international_with_separators"
                    ]
                )

        if re.search(r"\b(duplicate|duplicates|repeated)\b", lower):
            results.append(COMMON_REGEX_PATTERNS["duplicates"])

        if re.search(
            r"\b(time|hh:mm|hh:mm:ss|am|pm|12-hour|24-hour)\b",
            lower,
        ):
            if re.search(r"hh:mm:ss|\bseconds\b", lower):
                results.append(COMMON_REGEX_PATTERNS["time_hh_mm_ss_24h"])
            elif re.search(r"\b(am|pm|12-hour)\b", lower):
                results.append(COMMON_REGEX_PATTERNS["time_hh_mm_12h_ampm"])
            else:
                results.append(COMMON_REGEX_PATTERNS["time_hh_mm_24h"])

        has_explicit_date_signal = bool(
            re.search(r"\bdate\b", lower)
            or re.search(
                r"\byyyy\b.*\bmm\b.*\bdd\b|yyyy[- /.]mm[- /.]dd",
                lower,
            )
            or re.search(
                r"\bmm\b.*\bdd\b.*\byyyy\b|mm[- /.]dd[- /.]yyyy",
                lower,
            )
            or re.search(
                r"\bdd\b.*\bmm\b.*\byyyy\b|dd[- /.]mm[- /.]yyyy",
                lower,
            )
        )
        if has_explicit_date_signal:
            if re.search(
                r"\byyyy\b.*\bmm\b.*\bdd\b|yyyy[- /.]mm[- /.]dd",
                lower,
            ):
                results.append(COMMON_REGEX_PATTERNS["date_yyyy_mm_dd"])
            elif re.search(
                r"\bmm\b.*\bdd\b.*\byyyy\b|mm[- /.]dd[- /.]yyyy",
                lower,
            ):
                results.append(COMMON_REGEX_PATTERNS["date_mm_dd_yyyy"])
            elif re.search(
                r"\bdd\b.*\bmm\b.*\byyyy\b|dd[- /.]mm[- /.]yyyy",
                lower,
            ):
                results.append(COMMON_REGEX_PATTERNS["date_dd_mm_yyyy"])
            else:
                results.append(COMMON_REGEX_PATTERNS["date_yyyy_mm_dd"])

        if re.search(r"\b(decimal|float|fraction)\b", lower):
            results.append(COMMON_REGEX_PATTERNS["decimal_number"])

        if re.search(r"\b(whole\s+number|integer|int)\b", lower):
            results.append(COMMON_REGEX_PATTERNS["whole_number"])

        if re.search(
            r"\b(signed|positive|negative|plus\s+or\s+minus)\b",
            lower,
        ):
            results.append(COMMON_REGEX_PATTERNS["signed_whole_or_decimal"])

        deduped: list[str] = []
        seen: set[str] = set()
        for pattern in results:
            if pattern in seen:
                continue
            seen.add(pattern)
            deduped.append(pattern)
        return deduped

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

        # Priority 2: full patterns for common scenarios.
        common_patterns = self._extract_common_scenario_patterns(prompt)
        if common_patterns:
            yield from common_patterns
            return

        # Priority 3: keyword-based semantic patterns.
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

        # Priority 4: short quoted literal + building blocks when no keywords.
        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            shortest = min(quoted, key=len)
            longest = max(quoted, key=len)
            if shortest != longest or len(shortest) <= 10:
                yield re.escape(shortest)
                yield from VALID_REGEX_CHARS
                return

        # Priority 5: raw building-block characters only.
        yield from VALID_REGEX_CHARS
