"""LLM-backed token selector for constrained decoding.

:class:`TokenSelector` wraps the model's logit API and implements the
core constraint-enforcement step: given a set of *allowed* token IDs, it
zeroes out the logits for every disallowed token and returns the ID with the
highest remaining score.

This guarantees that the model's probability mass flows entirely to tokens
that keep the output on a valid JSON path, while still respecting the
model's learned preferences among those valid continuations.
"""
from llm_sdk import Small_LLM_Model

from .types import AllowedTokenIds, Logits, TokenIds


class TokenSelector:
    """Selects the highest-scoring token from a constrained allowed set.

    Attributes:
        llm: The language model used to obtain next-token logit scores.
    """

    def __init__(self, llm: Small_LLM_Model) -> None:
        """Attach a language model for logit retrieval.

        Args:
            llm: An initialised :class:`~llm_sdk.Small_LLM_Model` instance.
        """
        self.llm = llm

    def force_token(
        self,
        prefix_ids: TokenIds,
        allowed_token_ids: AllowedTokenIds,
    ) -> int:
        """Return the allowed token ID with the highest model logit score.

        Algorithm
        ---------
        1. Query the model for raw logits given *prefix_ids*.
        2. Create a ``-inf``-filled copy of the logit vector.
        3. Copy the original logit score only for each allowed token ID.
        4. Return the index (= token ID) with the maximum value.

        Masking all disallowed tokens to ``-inf`` ensures that ``argmax``
        will always return one of the allowed IDs, never an out-of-range or
        schema-invalid token.

        Args:
            prefix_ids: The token ID sequence seen so far (prompt +
                previously generated tokens).
            allowed_token_ids: Token IDs that the :class:`PrefixMatcher`
                determined are valid next-step continuations.

        Returns:
            The single token ID with the highest logit among allowed tokens.

        Raises:
            RuntimeError: If any allowed token ID is outside the vocabulary
                range returned by the model.
        """
        logits: Logits = self.llm.get_logits_from_input_ids(prefix_ids)

        # Validate allowed IDs before masking to catch schema/tokeniser
        # mismatches early with a clear error.
        for allowed_token_id in allowed_token_ids:
            if allowed_token_id < 0 or allowed_token_id >= len(logits):
                raise RuntimeError(
                    "allowed token id is out of vocabulary bounds"
                )

        # Start with -inf everywhere; only allowed tokens get real scores.
        constrained_logits = [float("-inf")] * len(logits)
        for allowed_token_id in allowed_token_ids:
            constrained_logits[allowed_token_id] = logits[allowed_token_id]

        # argmax over the constrained logit vector.
        return max(
            range(len(constrained_logits)),
            key=lambda index: constrained_logits[index],
        )
