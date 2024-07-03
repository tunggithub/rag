from pydantic import BaseModel, validator
from typing import List, Dict 
from .tool_calling import ChatMessage


class ConversationTaskRequest(BaseModel):
    message_history: List[ChatMessage]

class ConversationTaskResponse(BaseModel):
    response: str
