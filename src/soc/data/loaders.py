from pathlib import Path
from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")
