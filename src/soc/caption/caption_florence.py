from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from soc.caption.caption_base import BaseCaptioner


class FlorenceCaptioner(BaseCaptioner):
    """
    Florence-2 captioner backend.

    Uses a prompt-based vision-language model for image captioning.
    Runs on CPU by default for consistency with the project setup.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str = "cpu",
        max_new_tokens: int = 80,
        task_prompt: str = "<CAPTION>",
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.task_prompt = task_prompt

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def caption(self, image_path: str | Path) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.task_prompt,
            images=image,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height),
        )

        caption = parsed.get(self.task_prompt, "").strip()

        if not caption:
            raise RuntimeError("Florence did not return a caption.")

        return caption
