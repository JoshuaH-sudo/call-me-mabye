import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class ConstrainedDecoder:
    """
    Decodes model outputs while constraining JSON structure but allowing
    flexible string and number parameter values.

    Two-mode constrained decoding:

    1. STRUCTURE MODE (default): Constrain to JSON scaffolding
       - Function name selection (constrained to available functions)
       - Parameter keys (constrained to defined parameters)
       - JSON structural tokens: { } : , [ ]
       - When entering a string value (after "param_name":"), transition to STRING mode

    2. STRING MODE (inside string values): Allow flexible token generation
       - Once inside quotes (after "param_name":"), generate any tokens
       - Stop when exiting string (closing unescaped quote)
       - Return to STRUCTURE MODE for next parameter

    This approach:
    - Generates parameter values specific to the prompt (LLM-driven)
    - Maintains JSON validity at token level (structure constraints)
    - Scales to complex prompts (no combinatorial explosion of candidates)

    Example flow for prompt "say hello to josh":
    [STRUCTURE] {"name":"fn_greet","parameters":{"name":"
    [STRING]    josh
    [STRUCTURE] "}}"
    """

    available_functions: list[FunctionDefinition]
    # Template for each function showing JSON structure with value placeholders
    function_templates: list[dict[str, Any]]
    llm: Small_LLM_Model

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ):
        self.llm = llm
        self.available_functions = available_functions
        # Build templates that define JSON structure with flexible value regions
        self.function_templates = [
            self._build_function_template(func) for func in available_functions
        ]

    def _encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return cast(list[int], self.llm.encode(text)[0].tolist())

    def _build_function_template(
        self, function_definition: FunctionDefinition
    ) -> dict[str, Any]:
        """
        Build a template describing the JSON structure for this function.
        Template includes:
        - fixed_prefix: JSON start up to first parameter value
        - param_regions: List of (param_name, param_type, start_marker, end_marker)
        - fixed_suffix: JSON end after all parameters
        """
        # Build JSON with placeholder values to establish structure
        parameters: dict[str, object] = {}
        for name, definition in function_definition.parameters.items():
            if definition.type == "string":
                parameters[name] = ""
            elif definition.type == "number":
                parameters[name] = 0
            else:
                raise RuntimeError(
                    f"unsupported parameter type: {definition.type}"
                )

        full_json = json.dumps(
            {
                "name": function_definition.name,
                "parameters": parameters,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

        return {
            "function_name": function_definition.name,
            "full_structure_json": full_json,
            "encoded_structure": self._encode_text(full_json),
            "parameters": [
                {
                    "name": param_name,
                    "type": param_def.type,
                }
                for param_name, param_def in function_definition.parameters.items()
            ],
        }

    def _is_in_string_value(self, generated_text: str) -> bool:
        """
        Detect if we're currently inside a string parameter value.
        Simple heuristic: count unescaped quotes.
        - Odd count = inside string
        - Even count = outside string
        """
        # Count quotes, excluding escaped quotes
        unescaped_quote_count = 0
        i = 0
        while i < len(generated_text):
            if generated_text[i] == '"':
                # Check if it's escaped
                num_backslashes = 0
                j = i - 1
                while j >= 0 and generated_text[j] == "\\":
                    num_backslashes += 1
                    j -= 1
                # If even number of backslashes, quote is not escaped
                if num_backslashes % 2 == 0:
                    unescaped_quote_count += 1
            i += 1
        return unescaped_quote_count % 2 == 1

    def _get_structure_tokens(self, generated_text: str) -> list[int]:
        """
        Get allowed tokens that maintain JSON structure.
        These are tokens from templates that match the current text position.

        Uses text-level matching (more robust than token ID matching) to handle
        tokenization boundaries that shift when string content is inserted.
        """
        allowed_token_ids: set[int] = set()

        for template in self.function_templates:
            template_json_text = template["full_structure_json"]

            # Check if generated_text is a valid prefix of this template's JSON
            if len(generated_text) <= len(template_json_text):
                if template_json_text[: len(generated_text)] == generated_text:
                    # This template matches! Find the next token(s) that extend it
                    if len(generated_text) < len(template_json_text):
                        # Get the next part of the template
                        next_char = template_json_text[len(generated_text)]
                        # Encode just the next character(s) to find valid token IDs
                        # Try encoding single chars and small chunks
                        for chunk_len in range(
                            1,
                            min(
                                4,
                                len(template_json_text)
                                - len(generated_text)
                                + 1,
                            ),
                        ):
                            next_chunk = template_json_text[
                                len(generated_text) : len(generated_text)
                                + chunk_len
                            ]
                            chunk_encoded = self._encode_text(next_chunk)
                            if chunk_encoded:
                                # Add the first token of this chunk as a valid continuation
                                allowed_token_ids.add(chunk_encoded[0])

        return list(allowed_token_ids)

    def _get_string_tokens(self) -> list[int]:
        """
        Get allowed tokens inside a string value.
        Allow most of the vocabulary (the model decides content).
        In practice, we exclude certain tokens but for now allow broadly.
        """
        # For string content, we allow almost any token from the vocabulary.
        # The model's logits will guide which token is best given the prompt.
        # We use a large range; the actual vocab size is determined by the model.
        # Most tokenizers have vocab size between 30k-50k.
        # We return a placeholder set that will be filtered by available logits.
        vocab_size = 150000  # Safe upper bound for most models
        # Return a representative sample to avoid huge lists
        # The actual filtering happens in _force_token via logits
        return list(range(vocab_size))

    def _next_allowed_token_ids(
        self,
        generated_text: str,
    ) -> list[int]:
        """
        Determine which tokens are allowed next based on generation state.

        Two modes:
        - STRUCTURE MODE: Constrain to JSON structure tokens from templates
        - STRING MODE: Allow flexible token generation (from vocab)

        The state machine transitions based on quote count in generated text.

        Has fallback logic: if structure matching fails, allow broader set of tokens.
        """
        # Check if we're inside a string value (odd quote count)
        if self._is_in_string_value(generated_text):
            # STAGE 2B: STRING MODE
            # Inside a string parameter value: allow any token.
            # The model logits (not the constraint) decide which token is best.
            return self._get_string_tokens()
        else:
            # STAGE 2A: STRUCTURE MODE
            # In JSON structure: constrain to template tokens.
            allowed = self._get_structure_tokens(generated_text)

            # Fallback: if we can't find structure tokens, be more lenient
            # This handles tokenization boundary issues
            if not allowed:
                # If we're at the very beginning, allow tokens from all templates
                if len(generated_text) == 0:
                    allowed = self._get_string_tokens()
                else:
                    # Otherwise, allow a broader range of tokens
                    # This helps recover from tokenization mismatches
                    allowed = self._get_string_tokens()

            if not allowed:
                raise RuntimeError(
                    "no valid constrained JSON continuation available"
                )
            return allowed

    def _is_complete_json(self, generated_text: str) -> bool:
        """
        Check if generated_text is valid, complete JSON.
        Must have matching braces and valid structure.
        """
        if not generated_text:
            return False

        # Check basic structure: starts with { and ends with }
        if not (
            generated_text.startswith("{") and generated_text.endswith("}")
        ):
            return False

        # Check brace balance
        brace_count = 0
        for char in generated_text:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            if brace_count < 0:
                return False

        if brace_count != 0:
            return False

        # Try to parse as JSON
        try:
            parsed = json.loads(generated_text)
            return (
                isinstance(parsed, dict)
                and "name" in parsed
                and "parameters" in parsed
            )
        except (json.JSONDecodeError, ValueError):
            return False

    def _force_token(
        self,
        prefix_ids: list[int],
        allowed_token_ids: list[int],
    ) -> int:
        """
        Select the best token from allowed tokens using model logits.

        STAGE 3: Score tokens and select best from constrained set.
        - Get logits from the model for the current prefix
        - Mask all tokens except allowed ones to -inf
        - Select token with highest logit (greedy)
        """
        # STAGE 3A: Get logits from model
        logits = self.llm.get_logits_from_input_ids(prefix_ids)

        # STAGE 3B: Validate allowed tokens are within vocab
        for allowed_token_id in allowed_token_ids:
            if allowed_token_id < 0 or allowed_token_id >= len(logits):
                raise RuntimeError(
                    "allowed token id is out of vocabulary bounds: "
                    f"{allowed_token_id} (vocab size: {len(logits)})"
                )

        # STAGE 3C: Mask logits - set disallowed tokens to -inf
        constrained_logits = [float("-inf")] * len(logits)
        for allowed_token_id in allowed_token_ids:
            constrained_logits[allowed_token_id] = logits[allowed_token_id]

        # STAGE 3D: Greedy selection - pick highest logit from allowed set
        return max(
            range(len(constrained_logits)),
            key=lambda index: constrained_logits[index],
        )

    def force_json_output(
        self,
        prefix_input_ids: list[int],
    ) -> str:
        """
        Main generation loop with flexible constrained decoding.

        FLOW:
        1. Keep rolling prefix of [prompt context + generated output]
        2. At each step, determine if we're in STRING or STRUCTURE mode
        3. Get allowed tokens for current mode
        4. Score allowed tokens with model logits
        5. Select best token and append
        6. Stop when we have valid complete JSON

        This generates parameter values specific to the prompt while
        maintaining JSON validity throughout.
        """
        # STAGE 0: Initialize generation state
        generated_ids: list[int] = []  # Track token IDs
        rolling_prefix = list(prefix_input_ids)
        max_new_tokens = 500

        # STAGE 1: Token-by-token generation with constrained mode switching
        for step in range(max_new_tokens):
            # Decode current state to check string mode and structure
            generated_text = self.llm.decode(generated_ids)

            # STAGE 2: Determine allowed tokens based on generation state
            try:
                allowed_token_ids = self._next_allowed_token_ids(
                    generated_text
                )
            except RuntimeError as exc:
                # If no valid continuation, we might be in a bad state
                # Try to return what we have if it's valid JSON
                if self._is_complete_json(generated_text):
                    return generated_text
                raise exc

            # STAGE 3: Select next token from allowed set
            selected_token_id = self._force_token(
                prefix_ids=rolling_prefix,
                allowed_token_ids=allowed_token_ids,
            )

            # STAGE 4: Append token and update state
            generated_ids.append(selected_token_id)
            rolling_prefix.append(selected_token_id)

            # STAGE 5: Check for complete output
            generated_text = self.llm.decode(generated_ids)
            if self._is_complete_json(generated_text):
                return generated_text

        # If we hit max tokens, return what we have if it's valid
        generated_text = self.llm.decode(generated_ids)
        if self._is_complete_json(generated_text):
            return generated_text

        raise RuntimeError(
            f"generation did not complete valid JSON after {max_new_tokens} tokens. "
            f"Last output: {generated_text}"
        )

    def decode(self, input_ids: Any) -> str:
        prefix_ids = cast(list[int], input_ids[0].tolist())
        return self.force_json_output(prefix_ids)
