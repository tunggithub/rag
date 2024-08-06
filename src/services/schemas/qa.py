import re
from pydantic import BaseModel, Field, validator
from typing import List, Dict

class DataSource(BaseModel):
    source: str 
    context: str 

class QATaskRequest(BaseModel):
    prompt: str 
    datas: List[DataSource] = []
    files: List[Dict[str, str]] = []  

    @validator('files')
    def validate_files_content(cls, v):
        for file in v:
            if set(file.keys()) != set(['content', 'type']):
                raise ValueError("File element should contains 2 keys: 'content' and 'type'")
            if file['type'] != "image":
                raise ValueError("Only support file 'type' image")
            if not re.match(r"^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$", file['content']):
                raise ValueError("Invalid base64 image format")
        return v

class QATaskResponse(BaseModel):
    response: str 
    citations: List[Dict[str, str]] 

class VisionQATaskResponse(BaseModel):
    response: str 