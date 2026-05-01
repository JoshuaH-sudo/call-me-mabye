"""CLI argument parsing for the function-calling pipeline.

Converts raw ``sys.argv`` tokens into a validated
:class:`~src.cli.paths.AppPaths` instance.  Accepts minor spelling
variants of ``--function_definitions`` so that the CLI is forgiving of
common typos in automation scripts.
"""
from pathlib import Path

from .paths import AppPaths


def parse_args(argv: list[str]) -> AppPaths:
    """Parse CLI arguments and return resolved application paths.

    Expects exactly six positional tokens arranged as three key-value pairs::

        --functions_definition <path> --input <path> --output <path>

    The function-definitions flag is accepted in three spellings to avoid
    breaking callers that use the legacy ``--functions_definition`` form or
    the common ``--function_defintions`` typo.

    Args:
        argv: Argument list, typically ``sys.argv[1:]``.

    Returns:
        An :class:`AppPaths` instance with all three paths set.

    Raises:
        RuntimeError: When the argument count is wrong, a required key is
            absent, or the function-definitions flag is not recognised.
    """
    if len(argv) != 6:
        raise RuntimeError(
            "usage: python -m src --function_definitions <path> "
            "--input <path> --output <path>"
        )

    # Pair up flags and values:
    # ["-a", "1", "-b", "2"] -> {"-a": "1", "-b": "2"}
    arguments = dict(zip(argv[::2], argv[1::2], strict=True))

    # Accept three spelling variants of the function-definitions flag so that
    # existing scripts with minor typos continue to work.
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
