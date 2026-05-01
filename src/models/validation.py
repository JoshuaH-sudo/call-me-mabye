from ..decoder.models import FunctionDefinition
from .function_call import FunctionCallPayload


def is_valid_parameter_value(
    parameter_type: str,
    value: object,
) -> bool:
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise RuntimeError(f"unsupported parameter type: {parameter_type}")


def validate_function_payload(
    payload: object,
    function_index: dict[str, FunctionDefinition],
) -> FunctionCallPayload:
    validated_payload = FunctionCallPayload.model_validate(payload)

    function_definition = function_index.get(validated_payload.name)
    if function_definition is None:
        raise RuntimeError(
            "decoder selected an unknown function name: "
            f"{validated_payload.name}"
        )

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
