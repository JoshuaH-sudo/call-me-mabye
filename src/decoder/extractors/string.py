import re

SYMBOL_ALIASES: dict[str, str] = {
    "asterisk": "*",
    "asterisks": "*",
    "star": "*",
    "stars": "*",
    "plus": "+",
    "plus sign": "+",
    "minus": "-",
    "minus sign": "-",
    "dash": "-",
    "hyphen": "-",
    "underscore": "_",
    "pipe": "|",
    "vertical bar": "|",
    "slash": "/",
    "forward slash": "/",
    "backslash": "\\",
    "dot": ".",
    "period": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "question mark": "?",
    "exclamation mark": "!",
    "hash": "#",
    "hash sign": "#",
    "number sign": "#",
    "dollar": "$",
    "dollar sign": "$",
    "percent": "%",
    "percent sign": "%",
    "ampersand": "&",
    "at": "@",
    "at sign": "@",
    "equal": "=",
    "equals": "=",
    "equal sign": "=",
    "caret": "^",
    "tilde": "~",
    "left parenthesis": "(",
    "right parenthesis": ")",
    "left bracket": "[",
    "right bracket": "]",
    "left brace": "{",
    "right brace": "}",
}


class StringParameterExtractor:
    """Extracts string parameter candidates from prompts."""

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        quoted: list[str] = []
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in quoted:
                quoted.append(cleaned)
        return quoted

    def _extract_replacement_candidates(self, prompt: str) -> list[str]:
        candidates: list[str] = []
        with_match = re.search(
            (
                r"\bwith\b\s+"
                r"(?:\"([^\"\\]+)\"|'([^'\\]+)'|(.+?))"
                r"(?=\s+\b(?:in|on|for)\b|[.?!,;:]|$)"
            ),
            prompt,
            flags=re.IGNORECASE,
        )
        if with_match:
            raw = (
                with_match.group(1)
                or with_match.group(2)
                or with_match.group(3)
            )
            value = raw.strip()
            lowered = re.sub(r"\s+", " ", value.lower()).strip()
            symbol = SYMBOL_ALIASES.get(lowered)
            if symbol is not None:
                candidates.append(symbol)
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _extract_source_string_candidates(self, prompt: str) -> list[str]:
        quoted = self._extract_quoted_strings(prompt)
        if not quoted:
            return []
        longest = max(quoted, key=len)
        return [longest]

    def extract_candidates(
        self,
        prompt: str,
        parameter_name: str = "",
    ) -> list[str]:
        """Extract string candidates from quoted strings and bare words.

        Priority:
        1. Extract quoted strings (single and double quotes)
        2. Extract bare words (filtered by stopwords)
        3. Fallback to empty string if nothing found

        Args:
            prompt: The user prompt to extract candidates from.

        Returns:
            A list of string candidates ordered by priority.
        """
        candidates: list[str] = []

        if parameter_name == "replacement":
            candidates.extend(self._extract_replacement_candidates(prompt))
        elif parameter_name == "source_string":
            candidates.extend(self._extract_source_string_candidates(prompt))

        # Extract quoted strings (both single and double quotes)
        for quoted in self._extract_quoted_strings(prompt):
            if quoted not in candidates:
                candidates.append(quoted)

        # Extract bare words (fallback when no quotes)
        # Common stopwords to filter out
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "as",
            "if",
        }

        # Extract words using word boundary regex
        words = re.findall(r"\b\w+(?:['-]\w+)*\b", prompt)
        for word in words:
            # Skip stopwords and keep unique candidates
            if word.lower() not in stopwords and word not in candidates:
                candidates.append(word)

        # Fallback: if no candidates at all, return empty string
        if not candidates:
            candidates.append("")

        return candidates
