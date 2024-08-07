from pydantic import BaseModel, validator
from typing import Dict, List, Any

class ToolGenerationRequest(BaseModel):
    prompt: str 

class Parameter(BaseModel):
    required: bool
    description: str
    type: str

    def to_dict(self) -> Dict[str, str]:
        return {"required": self.required, "description": self.description, "type": self.type}

class ToolGenerationContent(BaseModel):
    name: str 
    description: str
    arguments: Dict[str, Parameter]

    def to_dict(self) -> Dict:
        output = {"name": self.name, "description": self.description, "arguments": {}}
        for argument in self.arguments:
            output['arguments'][argument] = self.arguments[argument].to_dict()
        return output
        
class ToolGenerationResponse(BaseModel):
    response: ToolGenerationContent
    
    def to_dict(self) -> Dict[str, Dict]:
        return {"response": self.response.to_dict()}
        

