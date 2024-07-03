from pydantic import BaseModel, validator
from typing import Dict, List, Any, Union

class ToolSample(BaseModel):
    name: str 
    description: str 
    arguments: Dict[str, Dict[str, str]]

    @validator('arguments')
    def validate_argument_format(cls, v):
        for argument in v:
            keys = list(v[argument].keys())
            if keys != ['required', 'type', 'description']:
                raise ValueError(f"Invalid format of function call argument")
        return v

class ChatMessage(BaseModel):
    role: str 
    content: str 

    @validator('role')
    def validate_chat_role(cls, v):
        if v not in ['user', 'assistant', 'tool call']:
            raise ValueError(f"Invalid role {v} for chat message")
        return v

class ToolCallingRequest(BaseModel):
    tools: List[ToolSample]
    message_history: List[ChatMessage]


class ExtractedArguments(BaseModel):
    role: str 
    content: Dict[str, Any]

    @validator('role')
    def validate_chat_role(cls, v):
        if v not in ['user', 'assistant', 'tool call']:
            raise ValueError(f"Invalid role {v} for chat message")
        return v
    
    @validator('content')
    def validate_tool_argument_extract(cls, v):
        if 'name' not in v:
            raise ValueError(f"Missing name of tool")
        if 'arguments' not in v:
            raise ValueError(f"Missing extracted arguments")
        for argument in v['arguments']:
            if not isinstance(argument, str):
                raise TypeError(f"Argument name must be string")

        return v

class ToolCallingResponse(BaseModel):
    response: List[Union[ChatMessage, ExtractedArguments]]

