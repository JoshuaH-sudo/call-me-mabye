from pydantic import BaseModel, ConfigDict


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_count: int
    prompt_count: int
    average_prompt_length: float


class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]


class FunctionCallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, object]
