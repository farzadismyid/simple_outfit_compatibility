from soc.caption.caption_base import BaseCaptioner
from soc.caption.caption_mock import MockCaptioner


def get_captioner() -> BaseCaptioner:
    """
    Factory function for selecting the captioning backend.

    For now, only the mock captioner is available.
    """
    return MockCaptioner()
