from pathlib import Path
from typing import Any
import pandas as pd


def build_explanation_payload(
    query_image_path: str | Path,
    target_category: str | None,
    recommendations_df: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Convert recommendation results into a clean structured payload
    that can later be passed to a prompt builder / LLM.

    Expected recommendation columns:
    - item_id
    - category
    - score

    Optional columns:
    - color
    - style
    - image_path
    """

    query_image_path = str(Path(query_image_path))

    df = recommendations_df.copy().head(top_k)

    payload = {
        "query": {
            "image_path": query_image_path,
            "target_category": target_category,
        },
        "recommendations": [],
    }

    for rank, row in enumerate(df.to_dict(orient="records"), start=1):
        item = {
            "rank": rank,
            "item_id": row.get("item_id"),
            "category": row.get("category"),
            "score": _safe_float(row.get("score")),
            "color": row.get("color"),
            "style": row.get("style"),
            "image_path": row.get("image_path"),
        }
        payload["recommendations"].append(item)

    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
