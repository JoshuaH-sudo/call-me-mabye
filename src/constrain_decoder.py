from typing import Any, cast

from llm_sdk import Small_LLM_Model
from .decoder_pipeline import (
    CandidateBuilder,
    FunctionDefinition,
    ParameterDefinition,
    PrefixMatcher,
    ReturnDefinition,
    TokenSelector,
)

__all__ = [
    "ParameterDefinition",
    "ReturnDefinition",
    "FunctionDefinition",
    "ConstrainedDecoder",
]


class ConstrainedDecoder:
    """
    Decodes model outputs while constraining choices
    to known function definitions.

    Token and ID flow (high level):

    text -> tokenizer -> token_ids -> model logits -> constrained choice

    Example:
    "* hello *" -> [42, 991, 17] -> logits over vocab -> keep only one id

    Output-shape flow:

    function definitions
        -> build fixed JSON candidates
        -> encode each candidate into token ids
        -> at each generation step, keep only tokens that continue
           one of those candidates
        -> final decoded text always matches one complete candidate

    Candidate example:
    {"name":"get_weather","parameters":{"city":"","days":0}}
    """

    available_functions: list[FunctionDefinition]
    encoded_output_candidates: list[list[int]]
    output_candidates: list[str]
    llm: Small_LLM_Model
    candidate_builder: CandidateBuilder
    prefix_matcher: PrefixMatcher
    token_selector: TokenSelector

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ):
        self.llm = llm
        self.available_functions = available_functions
        self.candidate_builder = CandidateBuilder()
        self.prefix_matcher = PrefixMatcher()
        self.token_selector = TokenSelector(llm=llm)
        self.output_candidates = self.candidate_builder.build_base_candidates(
            available_functions
        )
        self.encoded_output_candidates = [
            self._encode_text(candidate)
            for candidate in self.output_candidates
        ]

    def _encode_text(self, text: str) -> list[int]:
        return cast(list[int], self.llm.encode(text)[0].tolist())

    def rebuild_candidates_for_prompt(self, prompt: str) -> None:
        self.output_candidates = (
            self.candidate_builder.build_prompt_candidates(
                available_functions=self.available_functions,
                prompt=prompt,
            )
        )
        self.encoded_output_candidates = [
            self._encode_text(candidate)
            for candidate in self.output_candidates
        ]

    def _next_allowed_token_ids(
        self,
        generated_ids: list[int],
    ) -> list[int]:
        return self.prefix_matcher.next_allowed_token_ids(
            generated_ids=generated_ids,
            encoded_output_candidates=self.encoded_output_candidates,
        )

    def _is_complete_output(self, generated_ids: list[int]) -> bool:
        return self.prefix_matcher.is_complete_output(
            generated_ids=generated_ids,
            encoded_output_candidates=self.encoded_output_candidates,
        )

    def _force_token(
        self,
        prefix_ids: list[int],
        allowed_token_ids: list[int],
    ) -> int:
        return self.token_selector.force_token(
            prefix_ids=prefix_ids,
            allowed_token_ids=allowed_token_ids,
        )

    def force_json_output(
        self,
        prefix_input_ids: list[int],
    ) -> str:
        generated_ids: list[int] = []
        rolling_prefix = list(prefix_input_ids)

        # Token-by-token constrained decoding:
        #
        # prompt ids --------------------------+
        #                                      v
        # [prompt prefix] + [generated prefix] -> logits
        #                                      -> keep only tokens that still
        #                                         match at least one JSON
        #                                         candidate
        #                                      -> pick best remaining token
        #                                      -> append and repeat
        #
        # The loop stops only when generated_ids exactly equals one full
        # candidate sequence, so the decoded string is guaranteed to match the
        # precomputed JSON structure.
        while not self._is_complete_output(generated_ids):
            allowed_token_ids = self._next_allowed_token_ids(generated_ids)
            selected_token_id = self._force_token(
                prefix_ids=rolling_prefix,
                allowed_token_ids=allowed_token_ids,
            )
            generated_ids.append(selected_token_id)
            rolling_prefix.append(selected_token_id)

        return self.llm.decode(generated_ids)

    def decode(self, input_ids: Any) -> str:
        prefix_ids = cast(list[int], input_ids[0].tolist())
        return self.force_json_output(prefix_ids)
