---
description: "Workspace instructions for call-me-maybe. Use when implementing Python code, constrained decoding, data loading, validation, CLI behavior, and project documentation."
---

# Purpose
Build a reliable function-calling pipeline that converts prompts into schema-valid JSON function calls using constrained decoding.

# Hard Rules
- Use Python 3.10+ compatible code.
- Follow flake8 and mypy-compatible typing in all source code.
- All newly created classes must use pydantic validation.
- Handle errors gracefully with clear user-facing messages.
- Manage resources safely (prefer context managers for files and similar resources).
- In src code, do not use private llm_sdk attributes or methods.
- Function selection must be LLM-driven, not heuristic-only logic.
- Do not edit the llm_sdk library code directly; use it as a black-box interface.

# Project Constraints
- Allowed core libraries in src include json, numpy, and pydantic.
- Do not add direct usage of forbidden model-stack libraries in src (dspy, transformers, huggingface, torch, outlines, and similar). Use llm_sdk as the interface to the model.
- Default model target must remain compatible with Qwen/Qwen3-0.6B.

# CLI And IO Requirements
- Expected command shape:
  uv run python -m src --functions_definition <path> --input <path> --output <path>
- Support default data paths when optional arguments are omitted.
- Read and validate JSON input files with robust error handling for missing files and malformed JSON.
- Do not hardcode behavior to sample input data.

# Output Requirements
- Output must be valid JSON.
- Each result object must contain exactly:
  prompt, name, parameters
- Parameter names and types must match function_definitions exactly.
- No extra keys or free-form prose in output JSON.

# Decoder Expectations
- Implement constrained decoding at token-selection time.
- Enforce both JSON structure and schema validity during generation.
- Do not rely on prompting alone for structured output correctness.

# Reliability Targets
- Aim for near-perfect function-call correctness.
- Ensure 100% parseable JSON output.
- Keep processing time reasonable for full prompt sets.

# Coding Workflow
- For new features: define pydantic models first, then parsing/loading, then orchestration.
- For refactors: preserve external behavior unless the user asks for a behavior change.
- For bugfixes: identify root cause, implement minimal fix, and verify diagnostics.
- For reviews: report findings first, ordered by severity.

# Definition Of Done
- No diagnostics in touched files.
- Type hints are present and consistent.
- New classes use pydantic BaseModel with explicit validation intent.
- Error handling paths are covered for missing/invalid inputs.
- Behavior aligns with the project subject requirements.
