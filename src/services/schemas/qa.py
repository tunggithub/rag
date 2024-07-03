from pydantic import BaseModel, Field, validator
from typing import List

class DataSource(BaseModel):
    source: str 
    context: str 

class QATaskRequest(BaseModel):
    prompt: str 
    datas: List[DataSource]

class QATaskResponse(BaseModel):
    response: str 
    citations: List[str] 