from soc.config import LLM_BACKEND, OLLAMA_CONFIG
from soc.explain.llm_base import BaseLLM
from soc.explain.llm_mock import MockLLM
from soc.explain.llm_ollama import OllamaLLM


def get_llm() -> BaseLLM:
    """
    Factory function to return the selected LLM backend.
    """

    if LLM_BACKEND == "mock":
        return MockLLM()

    if LLM_BACKEND == "ollama":
        return OllamaLLM(**OLLAMA_CONFIG)

    raise ValueError(f"Unsupported LLM backend: {LLM_BACKEND}")
