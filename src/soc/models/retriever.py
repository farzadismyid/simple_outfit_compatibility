import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from soc.data.categories import validate_category


class OutfitRetriever:
    def __init__(self, catalog_df: pd.DataFrame, catalog_embeddings: np.ndarray):
        self.catalog_df = catalog_df.reset_index(drop=True)
        self.catalog_embeddings = catalog_embeddings

        if len(self.catalog_df) != len(self.catalog_embeddings):
            raise ValueError("Catalog dataframe and embeddings must have same length")

    def recommend(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        target_category: str | None = None,
    ) -> pd.DataFrame:
        df = self.catalog_df.copy()
        emb = self.catalog_embeddings

