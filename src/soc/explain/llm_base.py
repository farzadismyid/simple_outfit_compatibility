from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for explanation LLM backends.

    Any future backend, OpenAI, Ollama, Hugging Face, etc,
    should implement the same generate() interface.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from a prompt.
        """
        raise NotImplementedError
