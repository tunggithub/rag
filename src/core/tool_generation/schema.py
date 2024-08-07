from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union, Dict


class Parameter(BaseModel):
    description: str = Field(description="Description of the parameter.")
    required: bool = Field(description="Is this parameter required to be provided by the user, or can it be set to a default value? If yes, return true; otherwise, return false.")
    type: str = Field(description="Data type of the parameter.")
    
    def to_dict(self) -> Dict[str, str]:
        return {"description": self.description, "required": self.required, "type": self.type}

class Function(BaseModel):
    name: str = Field(description="Name of the function")
    arguments: Dict[str, Parameter] = Field(description="Arguments of the function.")
    description: str = Field(description="Function description")

    def to_dict(self) -> Dict:
        output = {
            "name": self.name,
            "description": self.description,
            "arguments": {}
        }
        for argument in self.arguments:
            output['arguments'][argument] = self.arguments[argument].to_dict()
        return output