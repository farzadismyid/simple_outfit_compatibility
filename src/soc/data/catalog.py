from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["item_id", "image_path", "category"]


def load_catalog(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in catalog CSV: {missing}")

    return df


def validate_catalog_images(
    df: pd.DataFrame, base_dir: str | Path | None = None
) -> pd.DataFrame:
    base_dir = Path(base_dir) if base_dir else None

    exists_list = []
    for rel_path in df["image_path"]:
        img_path = Path(rel_path)
        if base_dir is not None:
            img_path = base_dir / img_path
        exists_list.append(img_path.exists())

    df = df.copy()
    df["exists"] = exists_list
    return df
