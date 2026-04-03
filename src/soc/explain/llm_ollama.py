import json
from urllib import request, error

from soc.explain.llm_base import BaseLLM


class OllamaLLM(BaseLLM):
    """
    Ollama backend for local text generation.

    Uses the local Ollama API:
    http://localhost:11434/api/generate
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        keep_alive: str = "5m",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }

        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is installed, "
                "running, and serving on http://localhost:11434."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

        text = response_data.get("response")
        if not text:
            raise RuntimeError(
                "Ollama response did not contain generated text in the "
                "'response' field."
            )

        return text.strip()
