"""String parameter extraction from prompt text.

Provides :class:`StringParameterExtractor`, which collects string candidates
in the following priority order:

1. **Named-parameter shortcuts** — ``"replacement"`` and ``"source_string"``
   parameters use targeted extraction rules that look at specific prompt
   patterns (e.g. "replace … with X").
2. **Proper-noun candidates** — for ``"name"`` parameters, mid-sentence
   capitalised words and words following prepositions like "to"/"for" are
   extracted as high-priority candidates.
3. **Quoted strings** — content inside single or double quotes is preferred
   because it is the most explicit signal in the prompt.
4. **Bare content words** — all non-stopword tokens from the prompt are added
   as lower-priority candidates.  Structural/imperative words (e.g. "Greet",
   "Reverse") and tokens derived from the active function name are excluded.
5. **Path candidates** — strings that look like file paths (e.g.
    "/home/user/file.txt", "data.csv") are extracted as candidates because
    they are common in prompts
   and unlikely to be confused with other types of arguments.
6. **Empty string fallback** — if nothing else is found, ``""`` is returned
   to ensure the candidate list is never empty.
"""
import re

from .shared_keywords import IMPERATIVE_STRUCTURAL_WORDS, STOPWORDS

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

_IMPERATIVE_STRUCTURAL_WORDS = IMPERATIVE_STRUCTURAL_WORDS


def _build_function_name_exclusions(function_name: str) -> frozenset[str]:
    """Derive a set of tokens to exclude from the given *function_name*.

    Splits the name on underscores and removes the ``"fn"`` prefix token so
    that, for example, ``"fn_reverse_string"`` yields ``{"reverse", "string"}``
    and ``"fn_greet"`` yields ``{"greet"}``.

    Args:
        function_name: The raw function name as it appears in the schema.

    Returns:
        A frozenset of lowercase word tokens that should be excluded from
        bare-word string candidate generation for this function.
    """
    parts = function_name.lower().split("_")
    return frozenset(p for p in parts if p and p != "fn")


class StringParameterExtractor:
    """Extracts string parameter candidates from a user prompt."""

    def _unescape_quoted_text(self, value: str) -> str:
        """Interpret common escaped sequences from quoted prompt text."""
        return (
            value.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\'", "'")
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
            .replace(r"\r", "\r")
        )

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        """Return non-empty strings found inside single or double quotes.

        Args:
            prompt: The raw user prompt.

        Returns:
            A deduplicated list of quoted contents in appearance order.
        """
        quoted: list[str] = []
        quoted_matches = re.findall(
            r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'',
            prompt,
        )
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = self._unescape_quoted_text(value.strip())
            if cleaned and cleaned not in quoted:
                quoted.append(cleaned)
        return quoted

    def _extract_database_candidates(self, prompt: str) -> list[str]:
        """Extract database names from phrases like "on the X database"."""
        candidates: list[str] = []
        for match in re.finditer(
            r"\bon\s+the\s+([A-Za-z0-9_-]+)\s+database\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            value = match.group(1).strip()
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _extract_query_candidates(self, prompt: str) -> list[str]:
        """Extract SQL query candidates, prioritising quoted SQL text."""
        quoted = self._extract_quoted_strings(prompt)
        if not quoted:
            return []
        longest = max(quoted, key=len)
        return [longest]

    def _extract_encoding_candidates(self, prompt: str) -> list[str]:
        """Extract encodings from phrases like "with utf-8 encoding"."""
        candidates: list[str] = []
        for match in re.finditer(
            r"\bwith\s+([A-Za-z0-9._-]+)\s+encoding\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            value = match.group(1).strip()
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _extract_template_candidates(self, prompt: str) -> list[str]:
        """Extract the full template payload after "format template:"."""
        match = re.search(
            r"\bformat\s+template\s*:\s*(.+)$",
            prompt,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        value = match.group(1).strip()
        if not value:
            return []
        return [value]

    def _clean_path_candidate(self, value: str) -> str:
        """Trim wrappers and punctuation around an extracted path."""
        cleaned = value.strip().strip('"\'')
        cleaned = cleaned.rstrip(".,;:!?")
        return cleaned

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

    def _extract_proper_noun_candidates(self, prompt: str) -> list[str]:
        """Extract likely proper-noun argument values from *prompt*.

        Two strategies are applied in priority order:

        1. **Prepositional-phrase targets** — words immediately after
           ``to``, ``for``, ``called``, or ``named`` are strong signals that
           a proper name follows (e.g. "say hello *to Diana*", "greet
           *called John*").
        2. **Mid-sentence capitals** — a capitalised word that is *not* the
           first word of the prompt and is *not* a known structural word is
           likely a proper name (e.g. "Greet *Shrek*").

        Args:
            prompt: The raw user prompt.

        Returns:
            A deduplicated list of candidate strings ordered by extraction
            priority, or an empty list when no signals are found.
        """
        candidates: list[str] = []

        # Strategy 1: word following "to", "for", "called", or "named".
        for m in re.finditer(
            r"\b(?:to|for|called|named)\s+([A-Za-z]+)",
            prompt,
            flags=re.IGNORECASE,
        ):
            word = m.group(1)
            if word.lower() not in _IMPERATIVE_STRUCTURAL_WORDS:
                if word not in candidates:
                    candidates.append(word)

        # Strategy 2: mid-sentence capitalised words.
        words = prompt.split()
        for idx, word in enumerate(words):
            # Strip non-alphabetic and non-apostrophe characters so that
            # punctuation attached to the word does not prevent the
            # capitalisation check.  Digits are excluded because proper nouns
            # do not normally contain them.
            clean = re.sub(r"[^A-Za-z']", "", word)
            if not clean:
                continue
            # Skip the very first word (sentence-initial capital is noise).
            if idx == 0:
                continue
            if (
                clean[0].isupper()
                and clean.lower() not in _IMPERATIVE_STRUCTURAL_WORDS
            ):
                if clean not in candidates:
                    candidates.append(clean)

        return candidates

    def _extract_path_candidates(self, prompt: str) -> list[str]:
        """Extract filesystem paths while preserving special characters."""
        candidates: list[str] = []

        # Prompt-shape specific extraction for read-file style requests.
        context_match = re.search(
            (
                r"\bread\b(?:\s+the\s+file)?\s+"
                r"(?:at\s+)?(.+?)"
                r"(?=\s+with\s+[A-Za-z0-9._-]+\s+encoding\b|[.?!]|$)"
            ),
            prompt,
            flags=re.IGNORECASE,
        )
        if context_match:
            value = self._clean_path_candidate(context_match.group(1))
            if value and value not in candidates:
                candidates.append(value)

        # Windows drive paths.
        windows_pattern = re.compile(
            (
                r"(?<!\w)([A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n\s]+\\)*"
                r"[^\\/:*?\"<>|\r\n\s]+)"
                r"(?=\s+with\s+[A-Za-z0-9._-]+\s+encoding\b|[.?!]|$|\s)"
            )
        )

        # Unix absolute and home-relative paths.
        unix_pattern = re.compile(r"(?<!\w)((?:/|~/)[^\s'\",;:!?]+)")

        for match in windows_pattern.finditer(prompt):
            value = self._clean_path_candidate(match.group(1))
            if value and value not in candidates:
                candidates.append(value)

        for match in unix_pattern.finditer(prompt):
            value = self._clean_path_candidate(match.group(1))
            if value and value not in candidates:
                candidates.append(value)

        return candidates

    def extract_candidates(
        self,
        prompt: str,
        parameter_name: str = "",
        function_name: str = "",
    ) -> list[str]:
        """Extract string candidates from a prompt in priority order.

        Priority
        --------
        1. Named-parameter shortcuts (``"replacement"`` and
           ``"source_string"`` use dedicated extraction helpers).
        2. Proper-noun candidates for ``"name"`` parameters.
        3. Quoted strings (single and double quotes).
        4. Path candidates (e.g. "/home/user/file.txt", "data.csv").
        5. Bare content words (stopwords, imperative/structural words, and
           tokens derived from *function_name* are excluded).
        6. Empty string ``""`` if nothing else was found.

        Args:
            prompt: The user prompt to extract candidates from.
            parameter_name: The name of the parameter being extracted.
                Enables name-specific extraction logic when provided.
            function_name: The name of the function being called.  Used to
                derive additional tokens to exclude from bare-word candidates
                so that the function's own name words are never returned as
                argument values.

        Returns:
            A deduplicated list of string candidates in priority order.
        """
        candidates: list[str] = []

        # Step 1: name-specific high-priority extraction.
        if parameter_name == "replacement":
            candidates.extend(self._extract_replacement_candidates(prompt))
        elif parameter_name == "source_string":
            candidates.extend(self._extract_source_string_candidates(prompt))
        elif parameter_name == "database":
            return self._extract_database_candidates(prompt)
        elif parameter_name == "query":
            candidates.extend(self._extract_query_candidates(prompt))
        elif parameter_name == "encoding":
            return self._extract_encoding_candidates(prompt)
        elif parameter_name == "path":
            return self._extract_path_candidates(prompt)
        elif parameter_name == "template":
            return self._extract_template_candidates(prompt)

        # Step 2: proper-noun prioritizer for "name" parameters.
        if parameter_name == "name":
            for word in self._extract_proper_noun_candidates(prompt):
                if word not in candidates:
                    candidates.append(word)

        # Step 3: quoted strings (both single and double quotes).
        for quoted in self._extract_quoted_strings(prompt):
            if quoted not in candidates:
                candidates.append(quoted)

        # Step 4: path candidates (e.g. "/home/user/file.txt", "data.csv").
        for path in self._extract_path_candidates(prompt):
            if path not in candidates:
                candidates.append(path)

        # Step 5: bare content words — common stopwords, imperative structural
        # words, and tokens derived from the active function name are removed.
        # Build combined exclusion set: stopwords + imperative structural
        # words + tokens derived from the active function name.
        function_exclusions = _build_function_name_exclusions(function_name)
        excluded = (
            STOPWORDS | _IMPERATIVE_STRUCTURAL_WORDS | function_exclusions
        )

        # Match whole words, preserving contractions and hyphenated forms.
        words = re.findall(r"\b\w+(?:['-]\w+)*\b", prompt)
        for word in words:
            if word.lower() not in excluded and word not in candidates:
                candidates.append(word)

        # Step 6: final fallback — guarantee at least one candidate so that
        # the decoder always has a valid JSON string to constrain against.
        if not candidates:
            candidates.append("")

        return candidates
