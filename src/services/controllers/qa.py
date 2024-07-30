import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import QATaskRequest, QATaskResponse
from ...core import run_qa, llm, select_potential_context, embedding
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
        if len(request_data.datas) > 10:
            related_context = select_potential_context(
                embedding,
                request_data.prompt,
                request_data.datas,
            )
        else:
            related_context = request_data.datas
        print(f"Number of data sources after filter: {len(related_context)}")
        response = run_qa(llm, request_data.prompt, related_context)
        print(f"Processing time: {time.time() - start}")
        return response
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))