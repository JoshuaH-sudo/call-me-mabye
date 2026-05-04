"""Validation helpers for decoded function-call payloads.

After the constrained decoder emits a JSON string the app must verify that:

1. The JSON parses into a structure that matches :class:`FunctionCallPayload`.
2. The selected function name exists in the loaded function index.
3. The decoded parameter names exactly match the schema (no missing, no extra).
4. Every parameter value has the correct Python type for its declared schema
    type (``"string"`` → ``str``, ``"number"`` → ``int | float``,
    ``"integer"`` → ``int``).

:func:`validate_function_payload` performs all four checks in order and
raises a descriptive :exc:`RuntimeError` on the first failure so that the
caller can surface a clear message without inspecting raw exception fields.
"""
from ..decoder.models import FunctionDefinition
from .function_call import FunctionCallPayload


def is_valid_parameter_value(
    parameter_type: str,
    value: object,
) -> bool:
    """Return ``True`` when *value* is an acceptable Python type
    for *parameter_type*.

    Supported schema types:

    * ``"string"``  → must be a :class:`str`.
    * ``"number"``  → must be an :class:`int` or :class:`float`, but **not**
      a :class:`bool` (JSON booleans deserialize as Python bools which are a
      subclass of ``int``, so the explicit exclusion is necessary).
        * ``"integer"`` → must be a :class:`int`, but **not** a :class:`bool`.

    Args:
        parameter_type: The type string from the function schema.
        value: The decoded parameter value to check.

    Returns:
        ``True`` if the value matches the expected type, ``False`` otherwise.

    Raises:
        RuntimeError: If *parameter_type* is not one of the supported types.
    """
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "number":
        # Exclude bool explicitly: JSON true/false become Python True/False
        # which are instances of int, but should not be treated as numbers.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if parameter_type == "integer":
        # Exclude bool explicitly for the same reason as number above.
        return isinstance(value, int) and not isinstance(value, bool)
    raise RuntimeError(f"unsupported parameter type: {parameter_type}")


def validate_function_payload(
    payload: object,
    function_index: dict[str, FunctionDefinition],
) -> FunctionCallPayload:
    """Validate a decoded payload dict against the function schema index.

    Performs four sequential checks:

    1. **Structure** — ``payload`` must conform to
       :class:`~src.models.function_call.FunctionCallPayload`.
    2. **Name** — the selected function name must exist in *function_index*.
    3. **Parameter names** — the decoded parameter set must exactly match the
       schema's parameter set (no missing, no extra keys).
    4. **Parameter types** — every value must satisfy
       :func:`is_valid_parameter_value` for its declared schema type.

    Args:
        payload: Raw Python object (typically from ``json.loads``) to validate.
        function_index: Mapping of function name → definition, built from
            the loaded function definitions file.

    Returns:
        A validated :class:`~src.models.function_call.FunctionCallPayload`.

    Raises:
        RuntimeError: On any of the four validation failures described above.
    """
    # Step 1: structural validation via Pydantic.
    validated_payload = FunctionCallPayload.model_validate(payload)

    # Step 2: verify the function name was in the loaded definitions.
    function_definition = function_index.get(validated_payload.name)
    if function_definition is None:
        raise RuntimeError(
            "decoder selected an unknown function name: "
            f"{validated_payload.name}"
        )

    # Step 3: check that decoded parameter names exactly match the schema.
    expected_parameter_names = set(function_definition.parameters)
    actual_parameter_names = set(validated_payload.parameters)
    if actual_parameter_names != expected_parameter_names:
        missing_names = sorted(
            expected_parameter_names - actual_parameter_names
        )
        extra_names = sorted(actual_parameter_names - expected_parameter_names)
        raise RuntimeError(
            "decoded parameters do not match function schema for "
            f"{validated_payload.name}: missing={missing_names}, "
            f"extra={extra_names}"
        )

    # Step 4: verify every parameter value has the correct Python type.
    for (
        parameter_name,
        parameter_definition,
    ) in function_definition.parameters.items():
        value = validated_payload.parameters[parameter_name]
        if not is_valid_parameter_value(parameter_definition.type, value):
            raise RuntimeError(
                "decoded parameter has the wrong type for "
                f"{validated_payload.name}.{parameter_name}: expected "
                f"{parameter_definition.type}, got {type(value).__name__}"
            )

    return validated_payload
