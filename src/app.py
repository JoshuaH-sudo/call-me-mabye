import sys

from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import BaseModel, ConfigDict
from .constrain_decoder import ConstrainedDecoder, FunctionDefinition
from .data_loader import DatasetFileLoader, PromptCase


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_count: int
    prompt_count: int
    average_prompt_length: float


def summarize_dataset(
    functions: list[FunctionDefinition], prompts: list[PromptCase]
) -> DatasetSummary:
    prompt_lengths = np.array(
        [len(item.prompt) for item in prompts], dtype=np.float64
    )
    average_length = (
        float(prompt_lengths.mean()) if prompt_lengths.size else 0.0
    )
    return DatasetSummary(
        function_count=len(functions),
        prompt_count=len(prompts),
        average_prompt_length=average_length,
    )


def main() -> int:
    try:
        loader = DatasetFileLoader.from_argv(sys.argv[1:])
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        functions = loader.load_functions()
        prompts = loader.load_prompts()
        summary = summarize_dataset(functions, prompts)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(available_functions=functions, llm=llm)
    encoded = llm.encode("hello")
    logits = llm.get_logits_from_input_ids(encoded[0].tolist())
    returned_text = llm.decode([[24342]])
    path_to_merges_file = llm.get_path_to_merges_file()
    path_to_tokenizer_file = llm.get_path_to_tokenizer_file()
    path_to_vocab_file = llm.get_path_to_vocab_file()
    print(f"Encoded: {encoded}")
    print(f"Logits: {logits[:5]}...")
    print(f"Decoded: {returned_text}")
    print(f"Path to merges file: {path_to_merges_file}")
    print(f"Path to tokenizer file: {path_to_tokenizer_file}")
    print(f"Path to vocab file: {path_to_vocab_file}")
    print(
        "Function constraints encoded: "
        f"{len(decoder.encoded_function_definitions)}"
    )

    print("call-me-maybe scaffold")
    print(f"Functions loaded: {summary.function_count}")
    print(f"Prompts loaded: {summary.prompt_count}")
    print(f"Average prompt length: {summary.average_prompt_length:.2f}")
    return 0
