import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import QATaskRequest, QATaskResponse
from ...core import run_qa, llm
router = APIRouter()

async def get_dummy_data():
    dummy_response = QATaskResponse(
            response="This is response",
            citations="This is citation"
    )
    return dummy_response

@router.post('/qa-task', status_code=status.HTTP_200_OK, response_model=QATaskResponse)
async def qa_task(request_data: QATaskRequest):
    try:
        start = time.time()
        response = run_qa(llm, request_data.prompt, request_data.datas)
        print(f"Processing time: {time.time() - start}")
        return response
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))