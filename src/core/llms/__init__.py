from .factory import LLMFactory, LLMConfiguration
from langchain_openai import ChatOpenAI


LLMFactory.register('openai', ChatOpenAI)
