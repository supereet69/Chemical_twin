import pandas as pd
import numpy as np


def load_and_preprocess(path, dt=10):
    df = pd.read_csv(path)

    # normalize time column
    if "time_s" not in df.columns and "time" in df.columns:
        df["time_s"] = df["time"]

    df = df.sort_values(["run_id", "time_s"]).reset_index(drop=True)

    out = []
    for rid in df["run_id"].unique():
        dfr = df[df["run_id"] == rid].reset_index(drop=True)

        # downsample every N rows
        step = max(1, int(dt))
        dfr = dfr.iloc[::step].reset_index(drop=True)

        # recompute dt from time
        dfr["dt"] = dfr["time_s"].diff().fillna(dt)
        dfr.loc[dfr["dt"] <= 0, "dt"] = dt
        dfr["dt"] = dfr["dt"].fillna(dt).clip(lower=1)

        # smooth turbidity (NOISE FIX)
        if "turbidity" in dfr.columns:
            dfr["turbidity"] = dfr["turbidity"].rolling(5, min_periods=1).mean()

        # smooth TDS lightly (optional, helps stability)
        if "tds" in dfr.columns:
            dfr["tds"] = dfr["tds"].rolling(3, min_periods=1).mean()

        out.append(dfr)

    df2 = pd.concat(out, ignore_index=True)
    return df2
