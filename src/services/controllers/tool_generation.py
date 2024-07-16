import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import ToolGenerationRequest, ToolGenerationResponse
from ...core import run_tool_gen, llm

router = APIRouter()

async def get_dummy_data():
    dummy_response = ToolGenerationResponse(
        name = "convert_currency",
        description = "Tool to convert currency",
        arguments = {
            "amount": {
                "required": True,
                "type": "integer",
                "description": "Amount of money need to be exchanged"
            },
            "source_currency": {
                "required": True,
                "type": "string",
                "description": "Source type of currency (USD for example)"
            },
                "target_currency": {
                "required": True,
                "type": "string",
                "description": "Target type of currency (VND for example)"
            }
        }
    )
    return dummy_response

@router.post('/tool-generation', status_code=status.HTTP_200_OK, response_model=ToolGenerationResponse)
async def tool_generation_task(request_data: ToolGenerationRequest):
    try:
        start = time.time()
        response = run_tool_gen(
            llm,
            request_data.prompt
        )
        print(f"Processing time: {time.time() - start}")
        return ToolGenerationResponse(response=response)
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))
