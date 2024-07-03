import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import ConversationTaskRequest, ConversationTaskResponse
from ...core import run_conversation, llm

router = APIRouter()

async def get_dummy_data():
    dummy_response = ConversationTaskResponse(
            response="This is response"
    )     
    return dummy_response


@router.post('/conversation', status_code=status.HTTP_200_OK, response_model=ConversationTaskResponse)
async def conversation_task(request_data: ConversationTaskRequest):
    try:
        start = time.time()
        assistant_message = run_conversation(llm, request_data.message_history)
        response = ConversationTaskResponse(response=assistant_message.content)
        print(f"Processing time: {time.time() - start}")
        return response
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))