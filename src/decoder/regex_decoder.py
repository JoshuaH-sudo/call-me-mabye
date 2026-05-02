"""LLM-driven constrained decoder for regular expression generation.

:class:`RegexConstrainedDecoder` generates a valid Python-compatible
regular expression by running a token-by-token constrained generation
loop against a language model.  At each step only tokens that keep the
accumulated output a valid regex *prefix* are considered, guaranteeing
that the final output compiles without error.

Unlike the main
:class:`~src.decoder.constrained_decoder.ConstrainedDecoder`, which
constrains against a finite set of precomputed candidates, this decoder
constrains against the open-ended space of all valid regex strings by
using :class:`~src.decoder.regex_token_validator.RegexTokenValidator`
at token-selection time.

Import this module directly rather than via the package
``__init__``; it loads ``llm_sdk`` (which depends on ``torch``) and is
therefore excluded from the lazy package surface.
"""
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from llm_sdk import Small_LLM_Model

from .extractors.regex import RegexParameterExtractor
from .regex_token_validator import RegexTokenValidator

# Template used to frame the user prompt for the regex generation pass.
# The raw *prompt* is injected at ``{prompt}``.
#
# Important: do NOT use XML-like tags (e.g. <instruction>, <question>) here.
# Qwen3-0.6B generates closing XML tags (e.g. " </regex") when the prompt
# context looks like XML and ends with a word that resembles a tag name.
# Those closing-tag strings are technically valid Python regexes (they match
# the literal characters) so the validator would accept them — producing
# useless output.  A plain-text prompt avoids this entirely.
_PROMPT_TEMPLATE = "Task: {prompt}\nRegex: "

__all__ = ["RegexConstrainedDecoder"]


def _is_stop_token(text: str) -> bool:
    """Return ``True`` when *text* signals end-of-generation.

    A token is treated as a stop signal when its decoded text is empty,
    contains a newline character, or consists entirely of whitespace.
    These are reliable indicators that the model considers generation
    complete.

    Args:
        text: The decoded text of a vocabulary token.

    Returns:
        ``True`` if the token should terminate generation.
    """
    return not text or "\n" in text or text.isspace()


class RegexConstrainedDecoder(BaseModel):
    """Generates a valid regex string using LLM-driven constrained decoding.

    Generation proceeds token-by-token.  At each step the top-K
    vocabulary tokens (by model logit score) are decoded and filtered so
    that only those whose decoded text keeps the accumulated partial
    string a valid regex prefix are considered.  The highest-scoring
    surviving token is appended to the partial output and the loop
    repeats.

    Attributes:
        llm: The language model used for logit scoring and encoding.
        validator: Stateless validator that checks regex prefix validity.
        max_tokens: Maximum number of tokens to generate before giving
            up.
        top_k: Number of top-logit vocabulary entries to evaluate per
            step.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm: SkipValidation[Small_LLM_Model]
    validator: RegexTokenValidator = Field(
        default_factory=RegexTokenValidator
    )
    max_tokens: int = 50
    top_k: int = 200

    def _build_prompt(self, prompt: str) -> str:
        """Render the instruction + question wrapper around *prompt*.

        Args:
            prompt: The raw user prompt describing the regex to generate.

        Returns:
            A formatted string ready to be encoded as the model prefix.
        """
        return _PROMPT_TEMPLATE.format(prompt=prompt)

    def _top_k_ids(self, logits: list[float]) -> list[int]:
        """Return the top-K token IDs ranked by descending logit score.

        Uses :func:`numpy.argsort` for efficient partial ordering over
        large vocabulary arrays.

        Args:
            logits: Raw model logit scores, one per vocabulary entry.

        Returns:
            A list of up to *self.top_k* token IDs in descending logit
            order.
        """
        k = min(self.top_k, len(logits))
        arr = np.array(logits, dtype=np.float32)
        # argsort is ascending; take the last k entries and reverse to
        # get descending order.
        top_ids: list[int] = np.argsort(arr)[-k:][::-1].tolist()
        return top_ids

    def _fallback(self, prompt: str) -> str:
        """Return the best candidate from the keyword-based extractor.

        Used when constrained generation fails to produce a valid regex.

        Args:
            prompt: The raw user prompt.

        Returns:
            The highest-priority regex candidate from
            :class:`~src.decoder.extractors.regex.RegexParameterExtractor`,
            or ``".*"`` if the extractor yields nothing.
        """
        extractor = RegexParameterExtractor()
        candidates = list(extractor.extract_candidates(prompt))
        return candidates[0] if candidates else ".*"

    def generate_regex(self, prompt: str) -> str:
        """Generate a valid regex string for the intent in *prompt*.

        Algorithm
        ---------
        1. Build a structured instruction prompt and encode it as the
           initial rolling token-ID context.
        2. Loop (up to *max_tokens* steps):

           a. Query the model for next-token logits.
           b. Take the top-K token IDs by logit score.
           c. Decode each token and classify it as a stop signal or a
              regex continuation.
           d. From valid continuations keep only those whose decoded text
              keeps ``current_partial + text`` a valid regex prefix.
           e. If the model's best token (top-1) is a stop signal *and*
              the current partial is a valid complete regex, return it
              immediately.
           f. If no continuations pass, expand the search window to
              ``2 * top_k`` and retry once.  If still none, return the
              current partial when valid, otherwise fall back to the
              keyword extractor.
           g. Select the highest-logit passing continuation, append its
              text to *current_partial*, and append its ID to the
              rolling context.

        3. After the loop, return *current_partial* when valid; otherwise
           fall back to the keyword extractor.

        Args:
            prompt: The raw user prompt describing the regex intent.

        Returns:
            A Python-compatible regex string that can be passed to
            ``re.compile`` without raising :exc:`re.error`.
        """
        instruction = self._build_prompt(prompt)
        encoded = self.llm.encode(instruction)
        rolling_ids: list[int] = list(encoded[0].tolist())

        current_partial = ""

        for _ in range(self.max_tokens):
            logits: list[float] = self.llm.get_logits_from_input_ids(
                rolling_ids
            )
            top_ids = self._top_k_ids(logits)

            # Early termination: the model's single best token is a
            # stop signal and what we have so far is a valid complete
            # regex.  Only the top-1 token is checked here to avoid
            # spurious stops caused by low-probability stop tokens
            # appearing anywhere in the top-K window.
            if top_ids:
                top_text: str = self.llm.decode([top_ids[0]])
                stripped = current_partial.strip()
                if (
                    _is_stop_token(top_text)
                    and stripped
                    and self.validator.is_valid_complete_regex(stripped)
                ):
                    return stripped

            # Collect valid continuations from all non-stop top-K tokens.
            continuations: list[tuple[int, str, float]] = []
            for tid in top_ids:
                text: str = self.llm.decode([tid])
                if not _is_stop_token(text) and (
                    self.validator.is_valid_regex_prefix(
                        current_partial + text
                    )
                ):
                    continuations.append((tid, text, logits[tid]))

            # If no continuations survived the initial top_k window,
            # double the search window and retry once.
            if not continuations:
                expanded_k = min(2 * self.top_k, len(logits))
                arr = np.array(logits, dtype=np.float32)
                expanded_ids: list[int] = (
                    np.argsort(arr)[-expanded_k:][::-1].tolist()
                )
                # Skip tokens already evaluated in the initial top_k
                # pass (whether they were stop signals or failed the
                # prefix validity check).
                already_evaluated: set[int] = set(top_ids)
                for tid in expanded_ids:
                    if tid in already_evaluated:
                        continue
                    text = self.llm.decode([tid])
                    if not _is_stop_token(text):
                        if self.validator.is_valid_regex_prefix(
                            current_partial + text
                        ):
                            continuations.append(
                                (tid, text, logits[tid])
                            )

                if not continuations:
                    stripped = current_partial.strip()
                    if stripped and self.validator.is_valid_complete_regex(
                        stripped
                    ):
                        return stripped
                    return self._fallback(prompt)

            # Select the continuation with the highest logit score.
            best_tid, best_text, _ = max(
                continuations, key=lambda t: t[2]
            )
            current_partial += best_text
            rolling_ids.append(best_tid)

        # Post-loop: strip BPE leading/trailing whitespace and return when
        # the result is a valid complete regex.
        current_partial = current_partial.strip()
        if current_partial and self.validator.is_valid_complete_regex(
            current_partial
        ):
            return current_partial
        return self._fallback(prompt)
