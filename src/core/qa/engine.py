from typing import List, Dict
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence
from operator import itemgetter

from src.core.qa.prompt import system_prompt, data_prompt_template, user_prompt_template
from src.core.qa.schema import Answer
from src.core.embedding import select_potential_context


def _get_chain(llm: BaseLLM) -> RunnableSequence:
    parser = PydanticOutputParser(pydantic_object=Answer)

    format_data = lambda data: "\n".join([data_prompt_template.format(context=d.context, source=d.source) for d in data])
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(system_prompt)),
            HumanMessagePromptTemplate.from_template(
                template=user_prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()}),
        ]
    )

    chain = (
        {
            "question": itemgetter("question"),
            "data": itemgetter("data") | RunnableLambda(format_data)
        }
        | chat_template
        | llm
        | parser
    )
    return chain 



def run_qa(llm: BaseLLM, question: str, data: List[Dict[str, str]]) -> Answer:
    chain = _get_chain(llm)
    while True:
        try:
            return chain.invoke({"question": question, "data": data})
        except:
            pass


if __name__ == "__main__":
    import os
    import yaml
    import json
    from src.core.llms.factory import LLMFactory, LLMConfiguration
    from src.services.schemas.qa import DataSource

    def _load_yaml(config_file: str) -> dict:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
        return data

    llm_config = LLMConfiguration(**_load_yaml("./src/core/configs/dolphin.yaml"))
    llm = LLMFactory.create(llm_config)

    test_set = json.load(open("./test_data/qa_sample.json"))
    for i in range(10):
        for data in test_set:
            question = data["prompt"]
            data = data["datas"]

            data = [DataSource(**d) for d in data]

            response = run_qa(llm, question, data)
            print(response)
            print("-----")
