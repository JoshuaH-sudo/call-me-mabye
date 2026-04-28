# Flexible Constrained Decoding - Implementation Complete

## Summary

Successfully implemented **Approach 1: Flexible Constrained Decoding** that allows the LLM to generate actual parameter values specific to the prompt while maintaining JSON validity.

## Key Changes

### Before (Rigid Approach)
- Generated fixed candidates with default values: `{"name":"fn_greet","parameters":{"name":""}}`
- Decoder stopped as soon as any candidate was complete
- Parameter values were always empty strings or zeros
- Result: `{"name":"fn_greet","parameters":{"name":""}}`

### After (Flexible Approach)
- Builds JSON templates that track structure but allow value regions to vary
- Implements a **state machine** with two modes:
  - **STRUCTURE MODE**: Constrain tokens to JSON scaffolding (braces, colons, function names, parameter keys)
  - **STRING MODE**: Allow flexible token generation inside string values (decoder enters when inside quotes)
- Parameter values are LLM-driven based on prompt context
- Result: `{"name":"fn_greet","parameters":{"name":"josh"}}` for prompt "say hello to josh"

## Test Results

```
Prompt: 'say hello to josh'
✓ Function: fn_greet
✓ Parameters: {'name': 'josh'}        # Generated from prompt context!

Prompt: 'greet the user named alice'
✓ Function: fn_greet
✓ Parameters: {'name': 'alice'}       # Generated from prompt context!
```

## Implementation Details

### Core Components

1. **State Detection** (`_is_in_string_value`)
   - Counts unescaped quotes in generated text
   - Odd count = inside string value
   - Even count = in JSON structure

2. **Token Constraint Logic**
   - **STRUCTURE MODE** (`_get_structure_tokens`): Tokens from template JSON structure
   - **STRING MODE** (`_get_string_tokens`): Broad vocabulary range (filtered by logits)

3. **Main Generation Loop** (`force_json_output`)
   - **STAGE 0**: Initialize generation state
   - **STAGE 1**: Token-by-token loop (max 500 tokens)
   - **STAGE 2**: Determine allowed tokens (structure vs. string mode)
   - **STAGE 3**: Score tokens with model logits
   - **STAGE 4**: Append best token and update state
   - **STAGE 5**: Check for complete valid JSON

4. **Completion Detection** (`_is_complete_json`)
   - Validates JSON structure (matching braces)
   - Parses as valid JSON
   - Checks for required keys: "name" and "parameters"

### Advantages Over Rigid Approach

| Factor | Rigid (Before) | Flexible (After) |
|--------|----------------|-----------------|
| **Scaling** | $M^N$ candidates for N params, M examples | 1 template per function (linear) |
| **Parameter Values** | Fixed defaults (empty strings, zeros) | LLM-driven from prompt context |
| **Complex Prompts** | Can't adapt to prompt-specific context | Generates values specific to prompt |
| **Correctness** | High (but incorrect values) | High (and correct values) |

## Code Comments

Every stage of the new implementation includes clear comments explaining:
- What mode the decoder is in (STRUCTURE vs. STRING)
- Why constraints are relaxed or tightened
- How the state machine transitions
- When generation stops

Example comments in code:
```python
# STAGE 2B: STRING MODE
# Inside a string parameter value: allow any token.
# The model logits (not the constraint) decide which token is best.

# STAGE 2A: STRUCTURE MODE
# In JSON structure: constrain to template tokens.
```

## Files Modified

- `src/constrain_decoder.py`: Complete refactor of the `ConstrainedDecoder` class
  - Replaced rigid candidate matching with flexible templates
  - Added state machine logic
  - Improved comments and documentation

## Testing

Created `src/test_flexible_decoding.py` demonstrating:
- Multiple test cases with different prompts
- Correct parameter value extraction
- JSON output validation

## Next Steps

1. Test with the full function calling dataset
2. Handle edge cases (escaped quotes, nested structures)
3. Optimize token allowlist for string mode (current: broad range)
4. Validate against function definitions in `app.py`
