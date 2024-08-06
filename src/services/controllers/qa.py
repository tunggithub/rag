import traceback
import time
from typing import Union
from fastapi import APIRouter, HTTPException, status
from ..schemas import QATaskRequest, QATaskResponse, VisionQATaskResponse
from ...core import run_qa, llm, vlm, select_potential_context, embedding, run_vision_qa
router = APIRouter()

async def get_dummy_data():
    dummy_response = QATaskResponse(
            response="This is response",
            citations="This is citation"
    )
    return dummy_response

def _refine_qa_response(datas, response):
    citations = []
    source_to_context = {}
    for sample in datas:
        source_to_context[sample.source] = sample.context
    for citation in response.citations:
        citations.append({
            "source": citation,
            "context": source_to_context[citation]
        })
    final_response = QATaskResponse(
        response = response.response,
        citations = citations
    )
    return final_response
 

@router.post('/qa-task', status_code=status.HTTP_200_OK, response_model=Union[QATaskResponse, VisionQATaskResponse])
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
        if not request_data.files:
            response = run_qa(llm, request_data.prompt, related_context)
            print(f"Processing time: {time.time() - start}")
            return _refine_qa_response(related_context, response)
        else:
            response = await run_vision_qa(vlm, request_data.prompt, request_data.files)
            final_response = VisionQATaskResponse(response=response)
            return final_response
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))