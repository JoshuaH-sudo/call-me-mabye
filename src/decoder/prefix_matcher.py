"""Prefix-based token constraint checker.

At each step of constrained decoding the :class:`PrefixMatcher` answers
two questions:

1. Which token IDs can legally follow the sequence generated so far?
   (answered by :meth:`~PrefixMatcher.next_allowed_token_ids`)
2. Has the generated sequence already reached a complete candidate?
   (answered by :meth:`~PrefixMatcher.is_complete_output`)

The approach is purely string-matching: a token ID is *allowed* if
appending it to the generated sequence still results in a prefix of at
least one encoded output candidate.
"""
from .types import AllowedTokenIds, EncodedOutputCandidates, TokenIds


class PrefixMatcher:
    """Finds valid next tokens by checking against precomputed candidates.

    All logic is stateless; the same instance can be reused across many
    decoding steps and prompts without any reset.
    """

    def next_allowed_token_ids(
        self,
        generated_ids: TokenIds,
        encoded_output_candidates: EncodedOutputCandidates,
    ) -> AllowedTokenIds:
        """Return the set of token IDs that legally extend *generated_ids*.

        For every precomputed candidate the method checks whether
        *generated_ids* is a strict prefix of that candidate.  If it is,
        the token immediately following the prefix is added to the allowed
        set (deduplication is handled via *seen_token_ids*).

        Args:
            generated_ids: Token IDs produced so far in the current decoding
                loop (grows by one at each step).
            encoded_output_candidates: All valid output candidates encoded as
                token-ID sequences.

        Returns:
            A deduplicated list of token IDs that can come next without
            violating any candidate constraint.

        Raises:
            RuntimeError: If no candidate can be extended from the current
                prefix, which indicates a bug in the candidate-building or
                token-selection logic.
        """
        allowed_token_ids: AllowedTokenIds = []
        seen_token_ids: set[int] = set()

        for candidate_ids in encoded_output_candidates:
            prefix_length = len(generated_ids)

            # Skip candidates that are already fully consumed or shorter than
            # the current generated sequence.
            if prefix_length >= len(candidate_ids):
                continue

            # Skip candidates whose prefix does not match what was generated.
            if candidate_ids[:prefix_length] != generated_ids:
                continue

            next_token_id = candidate_ids[prefix_length]

            # Deduplicate: the same token may appear at this position in
            # multiple candidates, but we only need it once.
            if next_token_id in seen_token_ids:
                continue

            seen_token_ids.add(next_token_id)
            allowed_token_ids.append(next_token_id)

        if not allowed_token_ids:
            raise RuntimeError(
                "no valid constrained JSON continuation available"
            )
        return allowed_token_ids

    def is_complete_output(
        self,
        generated_ids: TokenIds,
        encoded_output_candidates: EncodedOutputCandidates,
    ) -> bool:
        """Return ``True`` when *generated_ids* equals a full candidate.

        This is the loop-exit condition: once the generated sequence matches
        an entire candidate the decoded string is guaranteed to be valid JSON
        that conforms to one of the precomputed function-call schemas.

        Args:
            generated_ids: Token IDs generated so far.
            encoded_output_candidates: All valid output candidates.

        Returns:
            ``True`` if *generated_ids* is an exact match for at least one
            candidate; ``False`` otherwise.
        """
        return any(
            generated_ids == candidate
            for candidate in encoded_output_candidates
        )
