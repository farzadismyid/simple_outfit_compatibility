from abc import ABC, abstractmethod
from pathlib import Path


class BaseCaptioner(ABC):
    """
    Abstract base class for image captioning backends.
    """

    @abstractmethod
    def caption(self, image_path: str | Path) -> str:
        """
        Generate a short caption or visual summary for an image.
        """
        raise NotImplementedError
