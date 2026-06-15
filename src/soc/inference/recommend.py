from pathlib import Path
import numpy as np
import pandas as pd

from soc.config import CATALOG_CSV, CATALOG_EMBEDDINGS
from soc.data.catalog import load_catalog
from soc.models.clip_encoder import CLIPEncoder
from soc.models.retriever import OutfitRetriever


def recommend_from_image(
    query_image_path: str | Path,
    top_k: int = 5,
    target_category: str | None = None,
) -> pd.DataFrame:
    catalog_df = load_catalog(CATALOG_CSV)
    catalog_embeddings = np.load(CATALOG_EMBEDDINGS)

    encoder = CLIPEncoder()
    retriever = OutfitRetriever(catalog_df, catalog_embeddings)

    query_embedding = encoder.encode_image(query_image_path)
    results = retriever.recommend(
        query_embedding=query_embedding,
        top_k=top_k,
        target_category=target_category,
    )
    return results
