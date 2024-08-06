from langchain_core.pydantic_v1 import BaseModel, Field

class VisionQAAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    
