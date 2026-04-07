from soc.caption.caption_base import BaseCaptioner
from soc.caption.caption_mock import MockCaptioner
from soc.caption.caption_florence import FlorenceCaptioner


CAPTION_BACKEND = "florence"  # options: "mock", "florence"


def get_captioner() -> BaseCaptioner:
    """
    Factory function for selecting the captioning backend.
    """

    if CAPTION_BACKEND == "mock":
        return MockCaptioner()

    if CAPTION_BACKEND == "florence":
        return FlorenceCaptioner()

    raise ValueError(f"Unsupported caption backend: {CAPTION_BACKEND}")
