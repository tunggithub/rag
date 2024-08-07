from langchain_core.pydantic_v1 import BaseModel, Field, validator
from datetime import datetime

def _try_to_refine(value):
        last_value = value.split(" ")[-1][:-1]
        print(f"Original value: {value}")
        try:
            refined_value = float(last_value)
            print(f"Refine value: {refined_value}")
            return refined_value
        except Exception:
            try:
                refined_value = datetime.strptime(last_value, '%m/%d/%Y')
                return refined_value 
            except Exception:
                return value

class VisionQAAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    
    @validator("answer")
    def must_be_number_or_date(cls, v):
        try:
            float_value = float(v)
        except Exception:
            try:
                date_value = datetime.strptime(v, '%m/%d/%Y')
            except Exception:
                v = _try_to_refine(v)
        return v

    
