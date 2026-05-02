"""Token-level validity checker for constrained regex generation.

:class:`RegexTokenValidator` provides stateless helper methods that
determine whether a string is a valid complete regex, a valid regex
prefix (i.e. could be extended into a legal regex), and which token IDs
among a candidate set would keep the generated output on a valid path.
"""
import re
import warnings

from pydantic import BaseModel, ConfigDict

# Suffixes appended to a partial regex when testing whether it can be
# completed into a valid regex.  Ordered from most to least likely to
# recover an unclosed structural element.
_COMPLETION_ATTEMPTS: list[str] = [
    "",       # test as-is (already a valid complete regex)
    "]",      # close an open character class
    ")",      # close an open group
    "a]",     # minimal char-class content + closer
    "a)",     # minimal group content + closer
    "d",      # allow \d, \w, \s … escape sequences to complete
    "a-z]",   # range completion
    "0-9]",   # digit-range completion
    "d+",     # \d+ style completion
    "A-Z]",   # uppercase range
    "]*",     # quantified close
    ")+",     # quantified group close
    ")?",     # optional group close
    "]?",     # optional char-class close
    "]+",     # one-or-more char-class close
]

__all__ = ["RegexTokenValidator"]


class RegexTokenValidator(BaseModel):
    """Stateless validator for regex token generation constraints.

    All methods are pure functions of their arguments.  This class
    carries no mutable state and may be safely shared across decoding
    passes.
    """

    model_config = ConfigDict(extra="forbid")

    def is_valid_complete_regex(self, s: str) -> bool:
        """Return ``True`` when *s* compiles as a Python regex.

        Args:
            s: The string to test.

        Returns:
            ``True`` if ``re.compile(s)`` succeeds; ``False`` otherwise.
        """
        try:
            re.compile(s)
            return True
        except re.error:
            return False

    def is_valid_regex_prefix(self, s: str) -> bool:
        """Return ``True`` when *s* could be extended into a valid regex.

        The check proceeds in two steps:

        1. Try compiling *s* as-is.  If it succeeds it is already a
           valid (complete) regex and therefore trivially a valid prefix.
        2. For each suffix in :data:`_COMPLETION_ATTEMPTS`, try
           compiling ``s + suffix``.  If any attempt succeeds, *s* is a
           valid prefix.

        This catches unclosed character classes (``[a-z``), unclosed
        groups (``(ab``), and partial escape sequences (``\\``).

        Args:
            s: The partial regex string to test.

        Returns:
            ``True`` if *s* can be completed into a valid regex;
            ``False`` otherwise.
        """
        if not s:
            return True
        for suffix in _COMPLETION_ATTEMPTS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    re.compile(s + suffix)
                return True
            except re.error:
                continue
        return False

    def filter_to_valid_continuations(
        self,
        partial: str,
        candidate_tokens: list[tuple[int, str]],
    ) -> list[int]:
        """Return token IDs whose decoded text keeps the prefix valid.

        For each ``(token_id, token_text)`` pair, this method checks
        whether ``partial + token_text`` passes
        :meth:`is_valid_regex_prefix`.  Only token IDs that pass are
        returned.

        Args:
            partial: The regex string generated so far.
            candidate_tokens: A list of ``(token_id, decoded_text)``
                pairs to evaluate.

        Returns:
            A list of token IDs (in input order) for which
            ``partial + token_text`` is a valid regex prefix.
        """
        result: list[int] = []
        for token_id, token_text in candidate_tokens:
            if self.is_valid_regex_prefix(partial + token_text):
                result.append(token_id)
        return result
