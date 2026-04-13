from llm_sdk import Small_LLM_Model

from .types import AllowedTokenIds, Logits, TokenIds


class TokenSelector:
    """Scores allowed tokens and returns the best one under constraints."""

    def __init__(self, llm: Small_LLM_Model):
        self.llm = llm

    def force_token(
        self,
        prefix_ids: TokenIds,
        allowed_token_ids: AllowedTokenIds,
    ) -> int:
        logits: Logits = self.llm.get_logits_from_input_ids(prefix_ids)
        for allowed_token_id in allowed_token_ids:
            if allowed_token_id < 0 or allowed_token_id >= len(logits):
                raise RuntimeError(
                    "allowed token id is out of vocabulary bounds"
                )

        constrained_logits = [float("-inf")] * len(logits)
        for allowed_token_id in allowed_token_ids:
            constrained_logits[allowed_token_id] = logits[allowed_token_id]

        return max(
            range(len(constrained_logits)),
            key=lambda index: constrained_logits[index],
        )
