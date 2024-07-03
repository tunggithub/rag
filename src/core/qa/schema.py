from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Answer(BaseModel):
    response: str = Field(description="The answer to the question")
    citations: List[str] = Field(description="List of useful sources to help you answer the question")
