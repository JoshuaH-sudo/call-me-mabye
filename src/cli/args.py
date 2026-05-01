from pathlib import Path

from .paths import AppPaths


def parse_args(argv: list[str]) -> AppPaths:
    """Parse CLI arguments and return resolved application paths.

    Expected shape:
        --functions_definition <path> --input <path> --output <path>

    Raises:
        RuntimeError: When required arguments are missing or unrecognized.
    """
    if len(argv) != 6:
        raise RuntimeError(
            "usage: python -m src --function_definitions <path> "
            "--input <path> --output <path>"
        )

    arguments = dict(zip(argv[::2], argv[1::2], strict=True))
    function_definitions_value = (
        arguments.get("--function_definitions")
        or arguments.get("--functions_definition")
        or arguments.get("--function_defintions")
    )
    if function_definitions_value is None:
        raise RuntimeError(
            "missing required argument: --function_definitions"
        )

    try:
        return AppPaths(
            function_definitions_file=Path(function_definitions_value),
            prompts_file=Path(arguments["--input"]),
            output_file=Path(arguments["--output"]),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"missing required argument: {exc.args[0]}"
        ) from None
