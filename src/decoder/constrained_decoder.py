"""Top-level constrained decoder that wires together all decoding components.

The :class:`ConstrainedDecoder` is the primary public interface for
running a complete function-call decode pass.  It delegates each
responsibility to a focused sub-component:

* :class:`~src.decoder.candidate_builder.CandidateBuilder` — selects the
  best-matching function and enumerates all plausible JSON candidates for
  the given prompt.
* :class:`~src.decoder.prefix_matcher.PrefixMatcher` — determines which
  token IDs are still valid at each generation step by checking the current
  prefix against all pre-encoded candidates.
* :class:`~src.decoder.token_selector.TokenSelector` — scores the allowed
  tokens using real model logits and returns the highest-scoring one.

Token and ID flow (high level)::

    text  →  tokenizer  →  token_ids  →  model logits  →  constrained choice

Output-shape flow::

    function definitions
        → build fixed JSON candidates
              (e.g. ``{"name":"add","parameters":{"a":1}}``)
        → encode each candidate into token-ID sequences
        → at each generation step, keep only token IDs that extend
          at least one pre-encoded candidate sequence
        → pick the model's preferred token among those survivors
        → repeat until generated_ids == one full candidate sequence
"""
from typing import cast

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, ParameterDefinition, ReturnDefinition
from .candidate_builder import CandidateBuilder
from .prefix_matcher import PrefixMatcher
from .token_selector import TokenSelector
from .types import (
    AllowedTokenIds,
    EncodedOutputCandidates,
    TokenIds,
)

__all__ = [
    "ParameterDefinition",
    "ReturnDefinition",
    "FunctionDefinition",
    "ConstrainedDecoder",
]


class ConstrainedDecoder:
    """Decodes a prompt into a schema-valid function-call JSON string.

    Unlike free-form text generation, every token emitted by this decoder
    is guaranteed to continue exactly one of the precomputed JSON candidates,
    so the final output is always parseable and structurally correct.

    Attributes:
        available_functions: All function definitions the decoder may choose
            from when building candidates for a prompt.
        llm: The language model used for both encoding (prompt → token IDs)
            and scoring (token IDs → logits).
        candidate_builder: Builds and ranks JSON output candidates per prompt.
        prefix_matcher: Checks which token IDs are valid at each step.
        token_selector: Selects the best token from the allowed set using
            real model logits.
    """

    available_functions: list[FunctionDefinition]
    llm: Small_LLM_Model
    candidate_builder: CandidateBuilder
    prefix_matcher: PrefixMatcher
    token_selector: TokenSelector

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ) -> None:
        """Initialise the decoder with a function schema list and an LLM.

        Args:
            available_functions: Validated function definitions to decode
                against.  Must contain at least one definition.
            llm: An initialised :class:`~llm_sdk.Small_LLM_Model` instance.
        """
        self.llm = llm
        self.available_functions = available_functions
        self.candidate_builder = CandidateBuilder()
        self.prefix_matcher = PrefixMatcher()
        self.token_selector = TokenSelector(llm=llm)

    def _encode_text(self, text: str) -> list[int]:
        """Tokenise *text* and return its token-ID sequence as a plain list.

        Args:
            text: Any UTF-8 string to tokenise.

        Returns:
            A flat list of integer token IDs (no special tokens prepended).
        """
        return cast(TokenIds, self.llm.encode(text)[0].tolist())

    def _next_allowed_token_ids(
        self,
        generated_ids: TokenIds,
        encoded_output_candidates: EncodedOutputCandidates,
    ) -> AllowedTokenIds:
        """Delegate to :class:`~src.decoder.prefix_matcher.PrefixMatcher`.

        Args:
            generated_ids: Token IDs emitted so far in the current decode.
            encoded_output_candidates: All pre-encoded JSON candidate
                sequences.

        Returns:
            Token IDs that legally extend the current generated sequence.
        """
        return self.prefix_matcher.next_allowed_token_ids(
            generated_ids=generated_ids,
            encoded_output_candidates=encoded_output_candidates,
        )

    def _is_complete_output(
        self,
        generated_ids: TokenIds,
        encoded_output_candidates: EncodedOutputCandidates,
    ) -> bool:
        """Return ``True`` when *generated_ids* matches a full candidate.

        Args:
            generated_ids: Token IDs emitted so far.
            encoded_output_candidates: All pre-encoded JSON candidates.

        Returns:
            ``True`` when decoding is finished; ``False`` to keep looping.
        """
        return self.prefix_matcher.is_complete_output(
            generated_ids=generated_ids,
            encoded_output_candidates=encoded_output_candidates,
        )

    def _force_token(
        self,
        prefix_ids: TokenIds,
        allowed_token_ids: AllowedTokenIds,
    ) -> int:
        """Delegate token scoring to
        :class:`~src.decoder.token_selector.TokenSelector`.

        Args:
            prefix_ids: The full token sequence seen by the model so far
                (original prompt IDs + generated IDs).
            allowed_token_ids: The constrained set of valid next tokens.

        Returns:
            The token ID with the highest model logit among allowed tokens.
        """
        return self.token_selector.force_token(
            prefix_ids=prefix_ids,
            allowed_token_ids=allowed_token_ids,
        )

    def apply_decoder(
        self,
        prefix_input_ids: TokenIds,
        prompt: str,
    ) -> str:
        """Run the full constrained decoding loop for one prompt.

        Steps
        -----
        1. Ask :class:`~src.decoder.candidate_builder.CandidateBuilder` to
           select the best-matching function and build all plausible JSON
           candidate strings.
        2. Encode every candidate string into a token-ID sequence so the
           prefix matcher can work at the token level.
        3. Loop token-by-token:
           a. Ask the prefix matcher which token IDs can come next.
           b. Ask the token selector to score them and pick the best one.
           c. Append the selected token to both ``generated_ids`` and the
              rolling model context (``rolling_prefix``).
        4. Stop when ``generated_ids`` exactly equals one complete candidate.
        5. Decode the final token sequence back to a JSON string.

        Args:
            prefix_input_ids: Token IDs of the encoded prompt (used as the
                initial rolling context for logit queries).
            prompt: The raw prompt text (used by the candidate builder for
                function selection and parameter extraction).

        Returns:
            A JSON string of the form ``{"name":"...","parameters":{...}}``
            that is guaranteed to match one of the precomputed candidates.
        """
        # Step 1: build JSON candidates and print them for visibility.
        output_candidates = self.candidate_builder.build_prompt_candidates(
            available_functions=self.available_functions,
            prompt=prompt,
        )
        print("========================================")
        print("Prompt:")
        print(prompt)
        print("Generated output candidates:")
        for candidate in output_candidates:
            print(candidate)
        print("========================================")

        # Step 2: encode every candidate string into token IDs once upfront
        # so the inner loop only does integer-list comparisons.
        encoded_output_candidates: EncodedOutputCandidates = [
            self._encode_text(candidate) for candidate in output_candidates
        ]

        # generated_ids accumulates the output tokens one at a time.
        generated_ids: TokenIds = []
        # rolling_prefix is the full context fed to the model at each step:
        # it starts as the prompt token IDs and grows by one token per loop.
        rolling_prefix = list(prefix_input_ids)

        # Step 3: token-by-token constrained generation loop.
        #
        # Visual overview:
        #   [prompt prefix] + [generated prefix]
        #          |                  |
        #          +------------------+
        #                   |
        #            model logits
        #                   |
        #          keep only allowed IDs
        #                   |
        #          pick highest-logit ID
        #                   |
        #       append to generated_ids and rolling_prefix
        #
        # The loop exits only when generated_ids equals a full candidate, so
        # the result is always a syntactically valid JSON function-call string.
        while not self._is_complete_output(
            generated_ids, encoded_output_candidates
        ):
            allowed_token_ids = self._next_allowed_token_ids(
                generated_ids, encoded_output_candidates
            )
            selected_token_id = self._force_token(
                prefix_ids=rolling_prefix,
                allowed_token_ids=allowed_token_ids,
            )
            generated_ids.append(selected_token_id)
            rolling_prefix.append(selected_token_id)

        # Step 4: convert the final token-ID sequence back to a string.
        return self.llm.decode(generated_ids)
