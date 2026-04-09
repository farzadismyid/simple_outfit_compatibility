from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from soc.config import PROJECT_ROOT


def _open_image(image_path: str | Path):
    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def show_query_and_recommendations(
    query_image_path: str | Path,
    recommendations_df: pd.DataFrame,
    explanation: str | None = None,
    query_caption: str | None = None,
    top_k: int = 5,
    figsize_per_item: tuple[int, int] = (4, 4),
):
    """
    Visualize query image and top-k recommended items.

    Displays:
    - query image
    - optional query caption
    - recommended images
    - rank, score, category, optional color/style
    - optional explanation text below the figure
    """

    df = recommendations_df.copy().head(top_k).reset_index(drop=True)

    n_items = len(df)
    total_panels = n_items + 1
    ncols = min(3, total_panels)
    nrows = math.ceil(total_panels / ncols)

    fig_width = figsize_per_item[0] * ncols
    extra_height = 0
    if query_caption:
        extra_height += 1.0
    if explanation:
        extra_height += 1.5
    fig_height = figsize_per_item[1] * nrows + extra_height

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    query_img = _open_image(query_image_path)
    axes[0].imshow(query_img)
    axes[0].axis("off")
    axes[0].set_title("Query Image", fontsize=11)

    for i, row in df.iterrows():
        ax = axes[i + 1]

        img = _open_image(row["image_path"])
        ax.imshow(img)
        ax.axis("off")

        title_lines = [
            f"Rank {i + 1}",
            f"score={row['score']:.3f}" if pd.notna(row.get("score")) else "score=N/A",
        ]

        if "category" in row and pd.notna(row["category"]):
            title_lines.append(f"cat={row['category']}")
        if "color" in row and pd.notna(row["color"]):
            title_lines.append(f"color={row['color']}")
        if "style" in row and pd.notna(row["style"]):
            title_lines.append(f"style={row['style']}")

        ax.set_title("\n".join(title_lines), fontsize=10)

    for j in range(total_panels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    y_pos = 0.02

    if explanation:
        fig.subplots_adjust(bottom=0.20)
        fig.text(
            0.02,
            y_pos,
            f"Explanation: {explanation}",
            ha="left",
            va="bottom",
            fontsize=10,
            wrap=True,
        )
        y_pos += 0.10

    if query_caption:
        fig.subplots_adjust(bottom=0.32 if explanation else 0.18)
        fig.text(
            0.02,
            y_pos,
            f"Query caption: {query_caption}",
            ha="left",
            va="bottom",
            fontsize=10,
            wrap=True,
        )

    plt.show()
