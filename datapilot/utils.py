import pandas as pd

def validate_csv(uploaded_file, chunksize=None):
    if uploaded_file is None:
        return False, "No file uploaded", None
    try:
        # Read efficiently; if large, can read in chunks or optimize low_memory
        df = pd.read_csv(uploaded_file, low_memory=False)
        if df.empty:
            return False, "Uploaded CSV file is empty", None
        return True, "CSV file validated and optimized successfully", df
    except Exception as e:
        return False, f"Error parsing CSV: {str(e)}", None

def get_column_types(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {"numerical": num_cols, "categorical": cat_cols}