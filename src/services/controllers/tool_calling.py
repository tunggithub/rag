import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import ToolCallingRequest, ToolCallingResponse, ExtractedArguments, ChatMessage
from ...core import run_tool_calling, llm

router = APIRouter()

async def get_dummy_data():
    chat_message = ChatMessage(
        role = 'assistant',
        content = 'This is chat message from assistant'
    )
    extracted_arguments = ExtractedArguments(
        role = 'tool call',
        content = {
            "name": "get_movie_recommendations",
            "arguments": {
                "genre": "action",
                "year": 2015
            }
        }
    )
    dummy_response = ToolCallingResponse(
        argument_extract = extracted_arguments,
        message = chat_message
    )
    return dummy_response


@router.post('/tool-calling', status_code=status.HTTP_200_OK, response_model=ToolCallingResponse)
async def tool_call_task(request_data: ToolCallingRequest):
    try:
        start = time.time()
        result = run_tool_calling(
            llm,
            request_data.tools,
            request_data.message_history
        )
        tool, message = result['tool'], result['chat']
        extracted_arguments = ExtractedArguments(
            role = "tool call",
            content = {
                "name": tool[0]['type'],
                "arguments": tool[0]['args']
            }
        )
        chat_message = ChatMessage(
            role='assistant',
            content=message
        )
        tool_call_response = [extracted_arguments, chat_message]
        response = ToolCallingResponse(
            response = tool_call_response
        )
        print(f"Processing time: {time.time() - start}")
        return response
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))