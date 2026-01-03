import pandas as pd # type: ignore

lags_default = {1, 3, 6, 12}

def load_macro_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.set_index("observation_date").sort_index()
    return df

def add_core_infl_6m_ann(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cpi = df["CPILFESL"]
    cpi_lag6 = cpi.shift(6)
    infl = 200 * ((cpi / cpi_lag6) - 1)
    df["core_infl_6m_ann"] = infl
    return df

def add_label_infl_up_6m(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.0) -> pd.DataFrame:
   df.copy()
   infl = df["core_infl_6m_ann"]
   future_infl = infl.shift(-horizon)
   delta = future_infl - infl
   y = (delta > threshold).astype(int)
   df["y_infl_up_6m"] = y
   return df

def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int] = None) -> pd.DataFrame:
    df = df.copy()
    if lags is None:
        lags = lags_default
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def build_xy(df: pd.DataFrame, feature_cols: list[str], label_col: str = "y_infl_up_6m"):
    df = df.copy()
    df = df.dropna(subset=feature_cols + [label_col])
    X = df[feature_cols]
    y = df[label_col].astype(int)
    dates = df.index
    return X, y, dates

def make_dataset(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.0, lags: list[int] = None):
    df = add_core_infl_6m_ann(df)
    df = add_label_infl_up_6m(df, horizon=horizon, threshold=threshold)

    df = df.copy()
    df["spread"] = df["DGS10"] - df["DGS2"]

    predictors = ["UNRATE", "FEDFUNDS", "INDPRO", "DGS2", "DGS10", "spread"]
    df = add_lag_features(df, cols=predictors, lags=lags)

    
    if lags is None:
        lags = lags_default
    feature_cols = [f"{col}_lag{lag}" for col in predictors for lag in lags]

    X, y, dates = build_xy(df, feature_cols=feature_cols, label_col="y_infl_up_6m")
    return X, y, dates, df
