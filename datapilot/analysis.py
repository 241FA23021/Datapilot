import pandas as pd
from .utils import get_column_types

def profile_dataset(df: pd.DataFrame) -> dict:
    """Summarize dataset quality and structure."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing = df.isna().sum()
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    col_types = get_column_types(df)

    recs = []
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        recs.append(f"Identified {dup_count} duplicate rows in the dataset. Recommended action: Remove duplicates to prevent training bias.")
    if len(col_types['numerical']) == 0:
        recs.append("Warning: No numeric features detected. Quantitative distribution analysis is limited.")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "row_count": len(df),
        "col_count": len(df.columns),
        "memory_usage_mb": memory_mb,
        "duplicates": dup_count,
        "missing": missing[missing > 0].sort_values(ascending=False),
        "missing_values": missing.to_dict(),
        "numeric_columns": col_types['numerical'],
        "categorical_columns": col_types['categorical'],
        "summary": df.describe(include="all").T,
        "recommendations": recs
    }

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Alias for backward compatibility with existing modules."""
    return profile_dataset(df)