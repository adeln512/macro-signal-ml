import pandas as pd # type: ignore
from features.make_dataset import make_dataset

macro_df = pd.read_csv(
    "data/data/processed/macro_monthly.csv",
    parse_dates=["observation_date"]
)
macro_df = macro_df.set_index("observation_date").sort_index()

X, y, dates, full_df = make_dataset(
    macro_df,
    horizon=6,
    threshold=0.0
)

print(
    full_df[["CPILFESL", "core_infl_6m_ann", "y_infl_up_6m"]].head(12)
)
