from .qa import run_qa
from .tool_calling import run_tool_calling
from .tool_generation import run_tool_gen
from .conversation_unfiltering import run_conversation
from .configs import llm_settings 
from langchain_openai import ChatOpenAI


try:
    llm = ChatOpenAI(
        base_url=llm_settings['base_url'],
        api_key=llm_settings['api_key'],
        model=llm_settings["model_name"]
    )
except Exception as err:
    raise Exception(f"Fail to init LLM due to {str(err)}")