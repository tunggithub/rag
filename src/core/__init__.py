from .qa import run_qa
from .tool_calling import run_tool_calling
from .tool_generation import run_tool_gen
from .conversation_unfiltering import run_conversation
from .configs import llm_settings, embedding_settings
from .embedding import select_potential_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


try:
    llm = ChatOpenAI(**llm_settings['selfhost'])
except Exception as err:
    raise Exception(f"Fail to init LLM due to {str(err)}")


try:
    embedding = OpenAIEmbeddings(api_key=embedding_settings['api_key'])
except Exception as err:
    raise Exception(f"Fail to init embedding model due to {str(err)}")