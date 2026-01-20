# final_model.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import DATA_PATH
from features import add_features


FINAL_MODEL_PATH = "models/final_outcome_model.pkl"

BM_CONST = 0.5  # your constant proxy (0.5g/100mL)


def compute_DI_eff(df_run: pd.DataFrame) -> pd.DataFrame:
    """
    Automatic mixing baseline detection:
    baseline = tds at the time when |tds_rate| becomes stable.
    """
    df_run = df_run.sort_values("t_rel_s").reset_index(drop=True).copy()

    # stability rule
    thr = 3.0  # ppm/s threshold (tune if needed)
    win = 20   # samples

    stable = (df_run["tds_rate"].abs() < thr).rolling(win, min_periods=win).mean()
    idx = stable[stable == 1.0].index

    if len(idx) == 0:
        mix_end_idx = min(60, len(df_run) - 1)
    else:
        mix_end_idx = int(idx[0])

    baseline = float(df_run.loc[mix_end_idx, "tds"])

    df_run["tds_baseline_mix"] = baseline
    df_run["DI_eff"] = (df_run["tds"] - baseline) / BM_CONST
    df_run["mix_end_t"] = float(df_run.loc[mix_end_idx, "t_rel_s"])

    return df_run


def build_final_dataset(df_feat: pd.DataFrame, early_window_s: float = 180.0) -> pd.DataFrame:
    rows = []

    for rid, g in df_feat.groupby("run_id"):
        g = g.sort_values("t_rel_s").reset_index(drop=True)

        g = compute_DI_eff(g)

        # take first early_window_s seconds as "available data"
        g_early = g[g["t_rel_s"] <= early_window_s]
        if len(g_early) < 10:
            continue

        # features summarizing early behavior
        row = {}
        row["run_id"] = rid
        row["acid_M"] = float(g["acid_M"].iloc[0])
        row["h2o2_ml"] = float(g["h2o2_ml"].iloc[0])

        row["tds_start"] = float(g_early["tds"].iloc[0])
        row["turb_start"] = float(g_early["turbidity"].iloc[0])
        row["temp_mean_early"] = float(g_early["temp_C"].mean())

        row["tds_min_early"] = float(g_early["tds"].min())
        row["tds_max_early"] = float(g_early["tds"].max())
        row["tds_slope_early"] = float((g_early["tds"].iloc[-1] - g_early["tds"].iloc[0]) / max(g_early["t_rel_s"].iloc[-1], 1))

        row["turb_mean_early"] = float(g_early["turbidity"].mean())
        row["turb_slope_early"] = float((g_early["turbidity"].iloc[-1] - g_early["turbidity"].iloc[0]) / max(g_early["t_rel_s"].iloc[-1], 1))

        # targets: final values
        row["final_tds"] = float(g["tds"].iloc[-1])
        row["final_turbidity"] = float(g["turbidity"].iloc[-1])
        row["final_DI_eff"] = float(g["DI_eff"].iloc[-1])

        rows.append(row)

    return pd.DataFrame(rows)


def train_final_model():
    df = pd.read_csv(DATA_PATH)
    df_feat = add_features(df)

    final_df = build_final_dataset(df_feat, early_window_s=180.0)

    X_cols = [
        "acid_M", "h2o2_ml",
        "tds_start", "turb_start",
        "temp_mean_early",
        "tds_min_early", "tds_max_early", "tds_slope_early",
        "turb_mean_early", "turb_slope_early",
    ]

    y_cols = ["final_tds", "final_turbidity", "final_DI_eff"]

    X = final_df[X_cols]
    y = final_df[y_cols]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    preds = model.predict(X)
    mae_tds = mean_absolute_error(y["final_tds"], preds[:, 0])
    mae_turb = mean_absolute_error(y["final_turbidity"], preds[:, 1])
    mae_di = mean_absolute_error(y["final_DI_eff"], preds[:, 2])

    print("=== Final Outcome Model (trained on available runs) ===")
    print(f"Final TDS MAE: {mae_tds:.2f}")
    print(f"Final Turbidity MAE: {mae_turb:.2f}")
    print(f"Final DI_eff MAE: {mae_di:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "X_cols": X_cols}, FINAL_MODEL_PATH)
    print(f"Saved -> {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train_final_model()
