from typing import List, Dict
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence
from langchain.output_parsers import PydanticOutputParser
from operator import itemgetter

from src.core.conversation_unfiltering.prompt import system_prompt

def _flatten_chat_message_dict(message_history: List[Dict[str, str]]):
    """
    List of message to List of Langchain BaseMessage
    """
    result = []
    for message in message_history:
        if message.role == "user":
            result.append(HumanMessage(message.content))
        elif message.role == "assistant":
            result.append(AIMessage(message.content))
        else:
            raise Exception("role must be assistant or user")
    return result


def _get_chain(llm: BaseLLM) -> RunnableSequence:
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="conversation")
        ]
    )

    chain = (
        {"conversation": itemgetter("message_history") | RunnableLambda(_flatten_chat_message_dict)}
        | chat_template
        | llm
    )
    return chain 



def run_conversation(llm: BaseLLM, message_history: List[Dict[str, str]]) -> str:
    chain = _get_chain(llm)
    return chain.invoke({"message_history": message_history})


if __name__ == "__main__":
    import os
    from src.core.llms.factory import LLMFactory, LLMConfiguration
    from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(model="gpt-4o")
    llm = ChatOpenAI()

    message_history = [
        {
            "role": "user",
            "content": "hello"
        },
        {
            "role": "assistant",
            "content": "hi"
        },
        {
            "role": "user",
            "content": "1+1=?"
        },
    ]

    print(run_conversation(llm, message_history))