from pydantic import BaseModel, validator
from typing import Dict, List, Any

class ToolGenerationRequest(BaseModel):
    prompt: str 

class Parameter(BaseModel):
    required: str
    description: str
    type: str

class ToolGenerationContent(BaseModel):
    name: str 
    description: str
    arguments: Dict[str, Parameter]

class ToolGenerationResponse(BaseModel):
    response: ToolGenerationContent

