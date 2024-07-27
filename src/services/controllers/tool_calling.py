import traceback
import time
from fastapi import APIRouter, HTTPException, status
from ..schemas import ToolCallingRequest, ToolCallingResponse, ExtractedArguments, ChatMessage
from ...core import run_tool_calling, llm
from typing import Dict, List
from pydantic import BaseModel, Field
from strenum import StrEnum
import json

router = APIRouter()


class Role(StrEnum):
    """One of ASSISTANT|USER to identify who the message is coming from."""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool call"
    TOOL_RESPONSE = "tool response"


class Message(BaseModel):
    """A list of previous messages between the user and the model, meant to give the model conversational context for responding to the user's message."""

    role: Role = Field(
        title="One of the ChatRole's to identify who the message is coming from.",
    )
    content: str | dict | list = Field( # TODO the dict/list was added to support json loading the function calls. this should maybe be done inside  a ToolMessage type
        title="Contents of the chat message.",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        """Create a ChatMessage object from a dictionary."""
        return cls(role=Role(data['role']), content=data['content'])
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}

def messages_from_list(data_list: List[Dict[str, str]]):
    messages = [Message.from_dict(item) for item in data_list]
    return messages


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


@router.post('/tool-calling', status_code=status.HTTP_200_OK)
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
        dict_response = response.to_dict()
        dict_response['response'] = json.dumps(dict_response['response'])
        return dict_response
    except Exception as err:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(err))