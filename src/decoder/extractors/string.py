"""String parameter extraction from prompt text.

Provides :class:`StringParameterExtractor`, which collects string candidates
in the following priority order:

1. **Named-parameter shortcuts** — ``"replacement"`` and ``"source_string"``
   parameters use targeted extraction rules that look at specific prompt
   patterns (e.g. "replace … with X").
2. **Quoted strings** — content inside single or double quotes is preferred
   because it is the most explicit signal in the prompt.
3. **Bare content words** — all non-stopword tokens from the prompt are added
   as lower-priority candidates.
4. **Empty string fallback** — if nothing else is found, ``""`` is returned
   to ensure the candidate list is never empty.
"""
import re

# Maps verbose symbol descriptions (as they might appear in a prompt) to
# the corresponding single-character symbol.  Used when extracting the
# "replacement" parameter so that "replace with asterisk" yields "*".
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
    """Extracts string parameter candidates from a user prompt."""

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        """Return non-empty strings found inside single or double quotes.

        Args:
            prompt: The raw user prompt.

        Returns:
            A deduplicated list of quoted contents in appearance order.
        """
        quoted: list[str] = []
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in quoted:
                quoted.append(cleaned)
        return quoted

    def _extract_replacement_candidates(self, prompt: str) -> list[str]:
        """Extract replacement-target candidates from a substitution prompt.

        Looks for the pattern *"with X (in|on|for|…)"* and resolves ``X`` to:
        - A symbol character if ``X`` matches a :data:`SYMBOL_ALIASES` entry
          (e.g. "asterisk" → ``"*"``).
        - The raw trimmed text of ``X`` otherwise.

        Both the symbol and the raw text are returned when they differ.

        Args:
            prompt: The raw user prompt.

        Returns:
            A list of 0–2 candidate strings ordered by specificity.
        """
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
            # Normalise whitespace before checking the alias table.
            lowered = re.sub(r"\s+", " ", value.lower()).strip()
            symbol = SYMBOL_ALIASES.get(lowered)
            if symbol is not None:
                candidates.append(symbol)
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _extract_source_string_candidates(self, prompt: str) -> list[str]:
        """Extract the most likely source-string candidate.

        For a ``source_string`` parameter the longest quoted string in the
        prompt is the strongest signal (it is most likely the full input to
        be processed rather than a short keyword).

        Args:
            prompt: The raw user prompt.

        Returns:
            A single-element list containing the longest quoted string, or
            an empty list if no quoted strings are found.
        """
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
        """Extract string candidates from a prompt in priority order.

        Priority
        --------
        1. Named-parameter shortcuts (``"replacement"`` and
           ``"source_string"`` use dedicated extraction helpers).
        2. Quoted strings (single and double quotes).
        3. Bare content words (stopwords excluded).
        4. Empty string ``""`` if nothing else was found.

        Args:
            prompt: The user prompt to extract candidates from.
            parameter_name: The name of the parameter being extracted.
                Enables name-specific extraction logic when provided.

        Returns:
            A deduplicated list of string candidates in priority order.
        """
        candidates: list[str] = []

        # Step 1: name-specific high-priority extraction.
        if parameter_name == "replacement":
            candidates.extend(self._extract_replacement_candidates(prompt))
        elif parameter_name == "source_string":
            candidates.extend(self._extract_source_string_candidates(prompt))

        # Step 2: quoted strings (both single and double quotes).
        for quoted in self._extract_quoted_strings(prompt):
            if quoted not in candidates:
                candidates.append(quoted)

        # Step 3: bare content words — common stopwords are removed so that
        # function/parameter names and meaningful nouns remain.
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

        # Match whole words, preserving contractions and hyphenated forms.
        words = re.findall(r"\b\w+(?:['-]\w+)*\b", prompt)
        for word in words:
            if word.lower() not in stopwords and word not in candidates:
                candidates.append(word)

        # Step 4: final fallback — guarantee at least one candidate so that
        # the decoder always has a valid JSON string to constrain against.
        if not candidates:
            candidates.append("")

        return candidates
