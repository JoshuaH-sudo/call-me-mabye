"""Result serialisation — writes the final function-call list to disk.

The output file is created (with any missing parent directories) and
written as a pretty-printed JSON array so that it is easy to read and
diff in version control.
"""
import json
import tempfile
from pathlib import Path

from ..models.function_call import FunctionCallResult


def ensure_output_writable(output_file: Path) -> None:
    """Fail fast if the output destination is not writable.

    This preflight check is intentionally separate from ``output_results`` so
    the application can surface permission issues before doing model work.

    Args:
        output_file: Destination path for the final JSON output.

    Raises:
        RuntimeError: If the output file cannot be created/written, the
            destination directory cannot be created, or the destination path
            is not a regular file.
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"unable to prepare output directory {output_file.parent}: {exc}"
        ) from exc

    if output_file.exists():
        if not output_file.is_file():
            raise RuntimeError(
                f"output path is not a regular file: {output_file}"
            )
        try:
            with output_file.open("a", encoding="utf-8"):
                pass
            return
        except OSError as exc:
            raise RuntimeError(
                f"output file is not writable: {output_file}: {exc}"
            ) from exc

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            prefix=".cmm-write-check-",
            delete=True,
        ):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"output directory is not writable: {output_file.parent}: {exc}"
        ) from exc


def output_results(
    output_file: Path,
    results: list[FunctionCallResult],
) -> None:
    """Serialize *results* to a JSON file at *output_file*.

    Creates any missing parent directories before writing.  The file is
    UTF-8 encoded, indented with two spaces, and terminated with a newline
    so that POSIX tools treat it as a well-formed text file.

    Args:
        output_file: Destination path.  The file is created or overwritten.
        results: List of function-call results to serialize.

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
