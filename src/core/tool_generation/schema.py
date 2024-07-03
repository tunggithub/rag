from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union, Dict


class Parameter(BaseModel):
    description: str = Field(description="Description of the parameter.")
    required: str = Field(description="Is this parameter required to be provided by the user, or can it be set to a default value? If yes, return true; otherwise, return false.")
    type: str = Field(description="Data type of the parameter.")

class Function(BaseModel):
    name: str = Field(description="Name of the function")
    arguments: Dict[str, Parameter] = Field(description="Arguments of the function.")
    description: str = Field(description="Function description")