import os
import pandas as pd

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_observations(df: pd.DataFrame, excel_path: str, overwrite: bool = False):
    if (not overwrite) and os.path.exists(excel_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {excel_path} (set overwrite: true)")
    df.to_excel(excel_path, index=False)