import pandas as pd

def validate_classification_readiness(df, target_col):
    if not target_col or target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    if df[target_col].nunique() < 2:
        return False, "Target column must have at least 2 distinct classes for classification."
    return True, "Dataset is fully ready for predictive modeling and classification."