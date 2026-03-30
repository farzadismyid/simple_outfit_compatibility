from pathlib import Path
import numpy as np
import torch
import open_clip

from soc.data.loaders import load_image


class CLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ):
#     device: str | None = None,
# ):
#     if device is None:
#         if torch.cuda.is_available():
#             device = "cuda"
#         else:
#             device = "cpu"

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        image = load_image(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()
