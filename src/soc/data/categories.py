import pandas as pd


def get_available_categories(df: pd.DataFrame) -> list[str]:
    """
    Return sorted list of unique categories in the catalog.
    """
    categories = df["category"].dropna().str.lower().unique().tolist()
    return sorted(categories)


def validate_category(
    df: pd.DataFrame,
    target_category: str | None,
) -> str | None:
    """
    Validate user requested category.

    - normalizes to lowercase
    - checks existence
    - raises clear error if invalid
    """

    if target_category is None:
        return None

    target_category = target_category.lower()

    available = get_available_categories(df)

    if target_category not in available:
        raise ValueError(
            f"Invalid category '{target_category}'. "
            f"Available categories: {available}"
        )

    return target_category
