*This project has been created as part of the 42 curriculum by JoshuaH-sudo.*

# call-me-maybe

## Description

**call-me-maybe** is a function-calling pipeline that translates natural-language prompts into schema-valid JSON function calls using a small language model (Qwen/Qwen3-0.6B, ~600 M parameters).

Given a prompt like `"What is the sum of 40 and 2?"`, the system does **not** answer the question directly. Instead it produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The key technical challenge — and the core learning goal — is achieving near-perfect reliability with a tiny model. The solution is **constrained decoding**: at every token-generation step the model's logit vector is masked so that only tokens that continue a valid JSON function-call candidate survive. This guarantees 100 % parseable, schema-compliant output regardless of the model's raw generation tendencies.

## Instructions

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Installation

```bash
make install
# or: uv sync
```

### Running

Run with default input/output paths (`data/input/` → `data/output/`):

```bash
make run
# or: uv run python -m src
```

Run with custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other commands

| Command | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make run` | Execute the pipeline with default paths |
| `make debug` | Run under Python's built-in `pdb` debugger |
| `make lint` | Run `flake8` + `mypy` with standard flags |
| `make lint-strict` | Run `mypy --strict` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, etc. |

### Input format

**`functions_definition.json`** — array of available functions:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

**`function_calling_tests.json`** — array of prompts:
```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

### Output format

**`function_calls.json`** — array of results (one object per prompt):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Algorithm Explanation

### Constrained Decoding

The pipeline generates the output JSON **one token at a time**, constrained to stay on a valid path:

1. **Candidate generation** (`CandidateBuilder`) — For every available function, parameter values are extracted from the prompt (numbers via regex, strings via quote detection and word extraction) and serialized into a finite set of compact JSON strings, e.g. `{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}`.

2. **Token encoding** — Every candidate string is encoded into a token-ID sequence using the model's tokenizer. These sequences are computed once per prompt.

3. **Prefix matching** (`PrefixMatcher`) — At each decoding step the matcher checks which token IDs can legally follow the already-generated prefix. A token is *allowed* if and only if it continues at least one pre-encoded candidate sequence.

4. **Logit masking** (`TokenSelector`) — The model's full logit vector (one float per vocabulary entry) is obtained. All disallowed tokens are masked to −∞. The token with the highest remaining score is selected. This is where the LLM drives the decision — including function selection, since all functions' candidates are present and the model's probability distribution favours whichever function best matches the prompt.

5. **Loop** — Steps 3–4 repeat until the generated token-ID sequence exactly equals one complete candidate. The result is then decoded back to a string and validated against the function schema.

### Why All Functions Are Included

Rather than using a heuristic (word overlap, keyword matching) to pre-select one function, the pipeline builds candidates for **all** available functions. The LLM's logits naturally assign higher probability to the token path that matches the prompt's intent. This ensures function selection is entirely model-driven, as required by the project specification.

## Design Decisions

| Decision | Rationale |
|---|---|
| Finite precomputed candidates | Makes prefix matching cheap (integer-list comparisons) and guarantees the output is always parseable. |
| All-function candidate set | Delegates function selection to the LLM via logit scoring; no heuristic needed. |
| Sliding-window for numeric params | Preserves natural left-to-right number ordering (e.g. "add 3 and 7" → a=3, b=7). |
| Cross-product for string/mixed params | Covers all plausible value combinations while staying bounded by `max_candidates_per_function`. |
| Pydantic models throughout | Validates all inputs and outputs at the boundary; failures surface as clear RuntimeError messages. |
| `argparse` for CLI | Provides standard `--help`, optional flags, and default paths without manual string parsing. |

## Performance Analysis

On a typical CPU run with Qwen/Qwen3-0.6B:

- **JSON validity**: 100 % — every output is parseable and schema-compliant by construction.
- **Function selection accuracy**: Near-perfect for unambiguous prompts; the LLM's logit scoring over all candidate prefixes is a strong signal.
- **Processing speed**: Roughly 5–30 seconds per prompt on CPU depending on candidate count and sequence length. The full default test set completes well within the 5-minute target.

Bottleneck: each generated token requires one forward pass through the model. The candidate count directly affects the prefix-matching step (O(candidates × prefix_length)) but this is negligible compared to the model forward pass.

## Challenges Faced

- **Tokenisation boundaries** — A single JSON character (e.g. `{`, `"`, `:`) may be split across multiple tokens or merged with adjacent characters differently depending on context. Pre-encoding full candidates sidesteps the need to reason about individual characters.
- **Number representation** — The same numeric value can appear as `2`, `2.0`, `2.00`, etc. The number extractor normalises floats and includes both integer and float representations to cover tokenizer differences.
- **LLM-driven function selection without prompting** — The project forbids heuristic selection but also forbids relying on prompt-only JSON generation. The solution — include all functions in the candidate set and let logit scoring decide — satisfies both constraints.

## Testing Strategy

Test the pipeline manually with the provided datasets:

```bash
# Standard test set
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

# Edge-case test set
uv run python -m src \
  --functions_definition data/input/functions_definition_edge_cases.json \
  --input data/input/function_calling_tests_edge_cases.json \
  --output data/output/function_calls_edge_cases.json
```

Validate the output:

```bash
python3 -c "
import json, sys
with open('data/output/function_calls.json') as f:
    results = json.load(f)
for r in results:
    assert set(r) == {'prompt', 'name', 'parameters'}, f'bad keys: {r}'
print(f'All {len(results)} results are valid.')
"
```

Edge cases to check: empty strings, large/negative numbers, special characters in strings, ambiguous prompts, multi-parameter functions.

## Example Usage

```bash
$ uv run python -m src --input data/input/function_calling_tests.json \
    --functions_definition data/input/functions_definition.json \
    --output data/output/function_calls.json

========================================
Prompt:
What is the sum of 2 and 3?
Generated output candidates:
{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}
{"name":"fn_greet","parameters":{"name":"2"}}
...
========================================
```

Output file `data/output/function_calls.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model used by this project.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — tokenization and model inference.
- [Outlines: structured generation](https://github.com/dottxt-ai/outlines) — reference implementation of constrained decoding (not used directly; the project reimplements the core idea).
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — the output format this pipeline guarantees.
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — used for all input/output validation.

### AI Usage

AI tools (GitHub Copilot / Claude) were used for the following tasks in this project:

- **Gap analysis**: reviewing the project against the subject PDF to identify non-compliant areas (e.g. heuristic function selection, missing CLI defaults).
- **Boilerplate generation**: initial docstring templates and type-hint scaffolding, reviewed and corrected manually.
- **Debugging assistance**: identifying why certain prompts produced wrong candidates (tokenisation edge cases with numbers).

All AI-generated content was reviewed, tested, and understood before being committed. No code was copy-pasted without verification.
