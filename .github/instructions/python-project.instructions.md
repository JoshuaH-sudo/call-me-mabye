---
applyTo: "**/*.py"
description: "Python coding rules for call-me-maybe. Use when editing loaders, decoder, CLI orchestration, and validation models."
---

# Scope
- Apply these rules to Python files only.
- Keep behavior aligned with project subject requirements.

# Python Rules
- Use Python 3.10+ compatible syntax.
- Keep code flake8 and mypy compatible.
- Add type hints for function parameters and return types.
- Handle failures with clear user-facing RuntimeError messages.
- Prefer context managers for file/resource handling.

# Validation Rules
- All newly created classes must use pydantic BaseModel.
- Use ConfigDict(extra="forbid") for structured input/output models unless explicitly required otherwise.

# Model And Decoder Rules
- Use llm_sdk as the model interface; do not call private llm_sdk attributes/methods.
- In src code, do not add direct usage of forbidden model-stack libraries (dspy, transformers, huggingface, torch, outlines, and similar).
- Keep function choice LLM-driven.
- Constrained decoding must enforce JSON structure and schema validity at token selection time.

# CLI And IO Rules
- Support CLI usage with optional paths and sensible defaults.
- Validate JSON inputs robustly, including malformed JSON and missing files.
- Do not hardcode behavior to example datasets.

# Output Rules
- Output must be valid JSON.
- Each result object must contain exactly: prompt, name, parameters.
- Parameter names and types must match function definitions exactly.
- Do not emit extra keys or prose in output JSON.
