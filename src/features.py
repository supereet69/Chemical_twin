import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame):
    df = df.copy()

    required = ["run_id", "time_s", "acid_M", "h2o2_ml", "temp_C", "tds", "turbidity", "recovery", "dt"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.sort_values(["run_id", "time_s"]).reset_index(drop=True)

    # targets (rates)
    df["tds_rate"] = df.groupby("run_id")["tds"].diff() / df["dt"]
    df["turbidity_rate"] = df.groupby("run_id")["turbidity"].diff() / df["dt"]
    df["recovery_rate"] = df.groupby("run_id")["recovery"].diff() / df["dt"]

    df[["tds_rate", "turbidity_rate", "recovery_rate"]] = (
        df[["tds_rate", "turbidity_rate", "recovery_rate"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # rolling means (good for stability)
    df["tds_mean"] = df.groupby("run_id")["tds"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["turb_mean"] = df.groupby("run_id")["turbidity"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # interactions
    df["acid_tds"] = df["acid_M"] * df["tds"]
    df["acid_turb"] = df["acid_M"] * df["turbidity"]
    df["h2o2_tds"] = df["h2o2_ml"] * df["tds"]

    FEATURES = [
        "acid_M", "h2o2_ml", "temp_C",
        "time_s", "tds", "turbidity", "recovery",
        "tds_mean", "turb_mean",
        "acid_tds", "acid_turb", "h2o2_tds",
    ]

    TARGETS = ["tds_rate", "turbidity_rate", "recovery_rate"]
    return df, FEATURES, TARGETS
