from langchain_core.messages import HumanMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from typing import List
import base64
import requests
import json

async def _build_content(question: str, images: List[str]):
    content = [
        {"type": "text", "text": question}
    ]
    for image in images:
        sample = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image['content']}"}
        }
        content.append(sample)
    return content

async def run_vision_qa(llm: BaseLLM, question: str, images: List[str]) -> str:
    content = await _build_content(question, images)
    msg = llm.invoke(
             [HumanMessage(content=content)]
    )
    return msg.content


if __name__ == "__main__":
    llm = ChatOpenAI(model='gpt-4o-mini', api_key='')
    data_path = "/home/trungpham/Project/source-code/rag/vision_qa_data/sample.json"
    datas = json.load(open(data_path))
    for sample in datas:
        question = sample['question']
        image = sample['image']
        print(run_vision_qa(llm, question, image))