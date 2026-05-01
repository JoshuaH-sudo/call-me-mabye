"""Result serialisation — writes the final function-call list to disk.

The output file is created (with any missing parent directories) and
written as a pretty-printed JSON array so that it is easy to read and
diff in version control.
"""
import json
from pathlib import Path

from ..models.function_call import FunctionCallResult


def output_results(
    output_file: Path,
    results: list[FunctionCallResult],
) -> None:
    """Serialise *results* to a JSON file at *output_file*.

    Creates any missing parent directories before writing.  The file is
    UTF-8 encoded, indented with two spaces, and terminated with a newline
    so that POSIX tools treat it as a well-formed text file.

    Args:
        output_file: Destination path.  The file is created or overwritten.
        results: List of function-call results to serialise.

    Raises:
        RuntimeError: On any OS-level write failure (permissions, full disk,
            etc.).
    """
    try:
        # Ensure the destination directory exists before opening the file.
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(
                [item.model_dump(mode="json") for item in results],
                handle,
                ensure_ascii=False,
                indent=2,
            )
            # Append a trailing newline so the file is POSIX-compliant.
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"unable to write output file {output_file}: {exc}")
