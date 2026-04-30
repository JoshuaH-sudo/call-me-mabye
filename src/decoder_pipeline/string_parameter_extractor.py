import re


class StringParameterExtractor:
    """Extracts string parameter candidates from prompts."""

    def extract_candidates(self, prompt: str) -> list[str]:
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

        # Extract quoted strings (both single and double quotes)
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

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
