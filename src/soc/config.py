from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CATALOG_IMG_DIR = DATA_DIR / "catalog_images"
QUERY_IMG_DIR = DATA_DIR / "query_images"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR = OUTPUT_DIR / "figures"

CATALOG_CSV = METADATA_DIR / "catalog.csv"
CATALOG_EMBEDDINGS = EMBEDDINGS_DIR / "catalog_embeddings.npy"
CATALOG_EMBEDDINGS_META = EMBEDDINGS_DIR / "catalog_embeddings_meta.csv"

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

# =========================
# LLM CONFIGURATION
# =========================

LLM_BACKEND = "ollama"  # options: "mock", "ollama"

OLLAMA_CONFIG = {
    "model_name": "llama3.2:3b",
    "base_url": "http://localhost:11434",
    "timeout": 120,
    "keep_alive": "5m",
}
