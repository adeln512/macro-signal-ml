import pandas as pd # type: ignore

indicators = {"CPILFESL": "CPILFESL (1).csv","UNRATE": "UNRATE.csv", "FEDFUNDS": "FEDFUNDS (1).csv", "DGS2": "DGS2.csv", "DGS10": "DGS10.csv", "INDPRO": "INDPRO.csv"}

indicators_dict = {}
for key, value in indicators.items():
    df = pd.read_csv(value)
    df["observation_date"] = pd.to_datetime(df['observation_date'])
    df = df.set_index("observation_date")
    df.columns = [key]
    indicators_dict[key] = df

for k, df in indicators_dict.items():
    if k in {"DGS2", "DGS10"}:
        df = df.resample("MS").mean()
        indicators_dict[k] = df
    #print(k, df.shape, df.index.min(), df.index.max())

macro_df = indicators_dict["CPILFESL"]
for key in list(indicators_dict.keys())[1:]:
    macro_df = macro_df.join(indicators_dict[key], how="inner")

macro_df = macro_df.sort_index()
macro_df.to_csv("data/processed/macro_monthly.csv")

macro_df = pd.read_csv("data/processed/macro_monthly.csv", parse_dates=["observation_date"], index_col="observation_date")

#print(macro_df.shape)
print(macro_df.head())



    




