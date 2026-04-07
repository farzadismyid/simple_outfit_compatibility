from pathlib import Path

from soc.caption.caption_base import BaseCaptioner


class MockCaptioner(BaseCaptioner):
    """
    Simple mock captioner for testing the pipeline
    before connecting a real VLM.
    """

    def caption(self, image_path: str | Path) -> str:
        return (
            "A fashion outfit image with a generally coordinated look. "
            "The visible style appears suitable for recommending complementary items."
        )
