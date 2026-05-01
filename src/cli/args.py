"""CLI argument parsing for the function-calling pipeline.

Converts raw ``sys.argv`` tokens into a validated
:class:`~src.cli.paths.AppPaths` instance.  All three path flags are
optional; when omitted the program reads from ``data/input/`` and writes
to ``data/output/`` by default.
"""
import argparse
from pathlib import Path
from typing import NoReturn

from .paths import AppPaths

_DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
_DEFAULT_INPUT = "data/input/function_calling_tests.json"
_DEFAULT_OUTPUT = "data/output/function_calls.json"


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises RuntimeError instead of calling sys.exit.

    This keeps error handling consistent with the rest of the pipeline,
    where every recoverable failure surfaces as a RuntimeError caught and
    printed by ``main()``.
    """

    def error(self, message: str) -> NoReturn:
        """Raise RuntimeError with *message* instead of exiting.

        Args:
            message: The argparse error description.

        Raises:
            RuntimeError: Always, carrying *message* as its argument.
        """
        raise RuntimeError(message)


def parse_args(argv: list[str]) -> AppPaths:
    """Parse CLI arguments and return resolved application paths.

    All three path arguments are optional.  When an argument is omitted
    the corresponding default path inside ``data/`` is used so that the
    program can be run without any flags::

        uv run python -m src

    The function-definitions flag is accepted in three spellings to avoid
    breaking callers that use the legacy form or the common typo.

    Args:
        argv: Argument list, typically ``sys.argv[1:]``.

    Returns:
        An :class:`AppPaths` instance with all three paths set.

    Raises:
        RuntimeError: When an unrecognised argument is supplied.
    """
    parser = _ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        "--function_definitions",
        "--function_defintions",
        dest="functions_definition",
        default=_DEFAULT_FUNCTIONS,
        metavar="PATH",
        help=(
            "path to the function definitions JSON file "
            f"(default: {_DEFAULT_FUNCTIONS})"
        ),
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        metavar="PATH",
        help=(
            "path to the prompts JSON file "
            f"(default: {_DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "path for the output JSON file "
            f"(default: {_DEFAULT_OUTPUT})"
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="print candidate lists and decoding details to stdout",
    )

    args = parser.parse_args(argv)
    return AppPaths(
        function_definitions_file=Path(args.functions_definition),
        prompts_file=Path(args.input),
        output_file=Path(args.output),
        debug=args.debug,
    )
