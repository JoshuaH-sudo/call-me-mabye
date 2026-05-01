import json
from pathlib import Path

from ..models.function_call import FunctionCallResult


def output_results(
    output_file: Path,
    results: list[FunctionCallResult],
) -> None:
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(
                [item.model_dump(mode="json") for item in results],
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"unable to write output file {output_file}: {exc}")
