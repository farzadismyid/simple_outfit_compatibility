from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")
