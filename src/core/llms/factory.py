from langchain_core.language_models.llms import BaseLLM
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LLMConfiguration:
    """Define LLM Configuration
    """
    name: str
    settings: Optional[Dict] = field(default_factory=dict)


class LLMFactory:
    """LLM Factory Class
    """
    engines = dict()

    @classmethod
    def register(cls, key: str, llm_class: BaseLLM) -> None:
        """Register llm class
        """
        if key in cls.engines:
            raise Exception(f"LLM: key {key} is existed")
        else:
            cls.engines[key] = llm_class

    @classmethod
    def create(cls, config: LLMConfiguration) -> BaseLLM:
        """Create a LLM instance
        """
        if config.name not in cls.engines:
            raise Exception(f"LLM: {config.name} is not registered")
        return cls.engines[config.name](**config.settings)
