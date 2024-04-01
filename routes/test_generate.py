from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from generate import gpt_agent_exec
from utils.code_utils import enforce_code_snippet

test_generate_router = APIRouter()


class TestGenerateInput(BaseModel):
    input: str
    test_context: Optional[str]


class TestGenerateOutput(BaseModel):
    input: str
    code: str
    reasoning: str


@test_generate_router.post("/generate-test/", tags=["test", "generate"])
async def generate_test(input: TestGenerateInput):
    output = TestGenerateOutput.model_validate(
        gpt_agent_exec.invoke(
            {"input": f"Generate code for the following input : { input.input }"}
        )
    )
    output.code = enforce_code_snippet(output.code)
    return output
