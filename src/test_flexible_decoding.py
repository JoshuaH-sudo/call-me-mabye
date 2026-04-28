import json

import llm_sdk
from .constrain_decoder import (
    ConstrainedDecoder,
    FunctionDefinition,
    ParameterDefinition,
    ReturnDefinition,
)

# STAGE 0: Define available functions
example_functions = [
    FunctionDefinition(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={"name": ParameterDefinition(type="string")},
        returns=ReturnDefinition(type="string"),
    ),
    FunctionDefinition(
        name="fn_subtract_numbers",
        description="Subtract one number from another and return the result.",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
        returns=ReturnDefinition(type="number"),
    ),
    FunctionDefinition(
        name="fn_count_characters",
        description="Count the number of characters in a string.",
        parameters={"s": ParameterDefinition(type="string")},
        returns=ReturnDefinition(type="number"),
    ),
]

# STAGE 1: Initialize LLM and decoder
llm = llm_sdk.Small_LLM_Model()
decoder = ConstrainedDecoder(available_functions=example_functions, llm=llm)

# STAGE 2: Test with different prompts
test_cases = [
    "say hello to josh",
    "greet the user named alice",
    "count characters in the word hello",
]

print("=" * 70)
print("FLEXIBLE CONSTRAINED DECODING TEST")
print("=" * 70)

for prompt_text in test_cases:
    print(f"\nPrompt: {prompt_text!r}")
    print("-" * 70)

    # Encode function definitions and prompt as context
    function_context = json.dumps(
        [
            {
                "name": f.name,
                "description": f.description,
                "parameters": {
                    name: {"type": param_def.type}
                    for name, param_def in f.parameters.items()
                },
            }
            for f in example_functions
        ],
        separators=(",", ":"),
        sort_keys=True,
    )
    function_context_token_ids = llm.encode(function_context)[0].tolist()
    prompt_token_ids = llm.encode(prompt_text)[0].tolist()

    # Combine context: function definitions + prompt
    prefix_input_ids = function_context_token_ids + prompt_token_ids

    # STAGE 3: Generate function call using flexible constrained decoding
    try:
        result_json = decoder.force_json_output(prefix_input_ids)
        result = json.loads(result_json)

        print(f"Function: {result['name']}")
        print(f"Parameters: {result['parameters']}")
        print(f"Raw JSON: {result_json}")

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("✓ Flexible constrained decoding complete!")
print("=" * 70)
