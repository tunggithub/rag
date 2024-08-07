import traceback
import base64
import requests
import json
from langchain_core.messages import HumanMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from typing import List
from langchain_core.runnables.base import RunnableSequence
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .schema import VisionQAAnswer
from .prompt import system_prompt



def _get_chain(llm: BaseLLM) -> RunnableSequence:
    parser = PydanticOutputParser(pydantic_object=VisionQAAnswer)
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(system_prompt)),
            (
                "human",
                [
                    {"type": "text", "text": "{question}"},
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{base64_image}",
                    },
                ],
            )
        ]
    )
    
    chain = (
        chat_template | llm | parser
    )
    return chain


async def run_vision_qa(llm: BaseLLM, question: str, base64_image: str):
    chain = _get_chain(llm)
    num_try = 10
    for i in range(num_try):
        try:
            result = chain.invoke({
                'question': question,
                'base64_image': base64_image
            })
            return result
        except Exception:
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    llm = ChatOpenAI(model='gpt-4o-mini', api_key='')
    data_path = "/home/trungpham/Project/source-code/rag/vision_qa_data/sample.json"
    datas = json.load(open(data_path))
    for sample in datas:
        question = sample['question']
        image = sample['image']
        print(run_vision_qa(llm, question, image))