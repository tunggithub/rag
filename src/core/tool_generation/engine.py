from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables.base import RunnableSequence
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


from src.core.tool_generation.schema import Function
from src.core.tool_generation.prompt import system_prompt, user_prompt_template


def _get_chain(llm: BaseLLM) -> RunnableSequence:
    parser = PydanticOutputParser(pydantic_object=Function)

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(system_prompt)),
            HumanMessagePromptTemplate.from_template(
                template=user_prompt_template),
        ]
    )

    chain = chat_template | llm | parser
    return chain


def run_tool_gen(llm: BaseLLM, context: str) -> Function:
    chain = _get_chain(llm)
    while True:
        try:
            return chain.invoke({"context": context})
        except:
            pass


if __name__ == "__main__":
    import os
    import yaml
    from src.core.llms.factory import LLMFactory, LLMConfiguration

    def _load_yaml(config_file: str) -> dict:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
        return data

    llm_config = LLMConfiguration(**_load_yaml("./src/core/configs/dolphin.yaml"))
    llm = LLMFactory.create(llm_config)


    context = "how many day from today to the next lunar new year"

    print(run_tool_gen(llm, context))