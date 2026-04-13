class PrefixMatcher:
    """Finds valid next tokens from candidate-token prefixes."""

    def next_allowed_token_ids(
        self,
        generated_ids: list[int],
        encoded_output_candidates: list[list[int]],
    ) -> list[int]:
        allowed_token_ids: list[int] = []
        seen_token_ids: set[int] = set()

        for candidate_ids in encoded_output_candidates:
            prefix_length = len(generated_ids)
            if prefix_length >= len(candidate_ids):
                continue
            if candidate_ids[:prefix_length] != generated_ids:
                continue

            next_token_id = candidate_ids[prefix_length]
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
        generated_ids: list[int],
        encoded_output_candidates: list[list[int]],
    ) -> bool:
        return any(
            generated_ids == candidate
            for candidate in encoded_output_candidates
        )
