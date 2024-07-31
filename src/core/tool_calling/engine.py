import json
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables.base import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers.string import StrOutputParser
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import SystemMessage
from typing import List, Dict
from operator import itemgetter


from src.core.tool_calling.prompt import system_prompt, tool_choice_system_prompt, tool_choice_system_prompt


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

def _convert_tools(tools: List[Dict]) -> List[Dict]:
    """convert to openai tool template
    """
    result = []
    for tool in tools:
        tool_ = {}
        arguments_ = {}
        required_ = []

        for k, v in tool.arguments.items():
            arguments_[k] = {}
            arguments_[k]["type"] = v["type"]
            arguments_[k]["description"] = v["description"]
            if v["required"] == "True":
                required_.append(k)

        tool_["type"] = "function"
        tool_["function"] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": arguments_,
                "required": required_
            }
        }
        result.append(tool_)

    return result


def _get_tool_choice_chain(llm) -> RunnableSequence:
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=tool_choice_system_prompt),
            MessagesPlaceholder(variable_name="conversation"),
            HumanMessagePromptTemplate.from_template(
                template=tool_choice_system_prompt),
        ]
    )
    return {
        "conversation": itemgetter("message_history") | RunnableLambda(_flatten_chat_message_dict),
        "tools": itemgetter("tools")
        } | chat_template | llm | StrOutputParser()


def _get_chain(llm: BaseLLM, llm_with_tools: BaseLLM) -> RunnableSequence:
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="conversation")
        ]
    )

    format_conversation = {"conversation": itemgetter("message_history") | RunnableLambda(_flatten_chat_message_dict)} | chat_template
    tool_chain = format_conversation | llm_with_tools | JsonOutputToolsParser()
    chat_chain = format_conversation | llm | StrOutputParser()


    chain = RunnableParallel(tool=tool_chain, chat=chat_chain)
    return chain


def run_tool_calling(llm: BaseLLM, tools: List[Dict], message_history: List[Dict]) -> Dict:
    openai_tools = _convert_tools(tools)
    if len(tools) > 1:
        tool_choice_chain = _get_tool_choice_chain(llm)
        tool_name = None
        num_try = 5
        i = 0
        while tool_name is None:
            i = i + 1
            if i > num_try:
                tool_name = tools[0].name
            response = tool_choice_chain.invoke(
                {
                    "message_history": message_history,
                    "tools": json.dumps(openai_tools, indent=4)
                }
            )
            for tool in tools:
                if tool.name in response:
                    tool_name = tool.name
                    break
    else:
        tool_name = tools[0].name

    llm_with_tools = llm.bind(
        tools=openai_tools,
        tool_choice= {"type": "function", "function": {"name": tool_name}}
    )
    chain = _get_chain(llm, llm_with_tools)
    return chain.invoke({"message_history": message_history})


if __name__ == "__main__":
    import os
    import yaml
    import json
    from src.core.llms.factory import LLMFactory, LLMConfiguration
    from src.services.schemas.tool_calling import ToolSample, ChatMessage

    def _load_yaml(config_file: str) -> dict:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
        return data

    llm_config = LLMConfiguration(**_load_yaml("./src/core/configs/dolphin.yaml"))
    llm = LLMFactory.create(llm_config)

    test_set = json.load(open("./test_data/tool_call_sample.json"))
    for i in range(10):
        for data in test_set:
            tools = data["tools"]

            message_history = data["message_history"]

            tools = [ToolSample(**tool) for tool in tools]
            message_history = [ChatMessage(**message) for message in message_history]

            print(run_tool_calling(llm, tools, message_history))
            print("-----")
