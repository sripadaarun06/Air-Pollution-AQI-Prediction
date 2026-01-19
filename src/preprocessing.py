import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "city_day.csv"

def load_csv(path: Path = CSV_PATH) -> pd.DataFrame: 
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_col: str):
    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")

    df = df.dropna(subset=[target_col])

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    if num_cols:
        mask = pd.Series(True, index=df.index)
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask &= df[col].between(lower, upper)
        df = df[mask]

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    numeric_X = X.select_dtypes(include=["number"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_X) if not numeric_X.empty else numeric_X.values

    return X_scaled, y, X.columns, scaler

if __name__ == "__main__":
    try:
        df = load_csv()
    except FileNotFoundError as e:
        print(e)
    else:
        print(f"Loaded {len(df)} rows from {CSV_PATH}")
        target = "AQI"
        if target in df.columns:
            X_scaled, y, cols, scaler = preprocess_data(df, target)
            print("Preprocessing done. Numeric features:", list(cols))
        else:
            print(f"Please set the correct target column. Available columns: {list(df.columns)[:10]}")

