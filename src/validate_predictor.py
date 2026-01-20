import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/data.csv"
MODEL_PATH = "models/predictor_model.pkl"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # time relative
    if "time_rel" in df.columns:
        df["t_rel_s"] = df["time_rel"]
    elif "time_s" in df.columns:
        df["t_rel_s"] = df["time_s"] - df.groupby("run_id")["time_s"].transform("min")
    else:
        raise ValueError("Dataset must contain time_rel or time_s")

    df = df.sort_values(["run_id", "t_rel_s"]).reset_index(drop=True)

    # dt
    if "dts" in df.columns:
        df["dt_s"] = df["dts"]
    else:
        df["dt_s"] = df.groupby("run_id")["t_rel_s"].diff().fillna(0)

    # avoid divide by zero
    df.loc[df["dt_s"] <= 0, "dt_s"] = np.nan

    # previous values
    df["tds_prev"] = df.groupby("run_id")["tds"].shift(1)
    df["turb_prev"] = df.groupby("run_id")["turbidity"].shift(1)
    df["temp_prev"] = df.groupby("run_id")["temp_C"].shift(1)

    # rates
    df["tds_rate"] = (df["tds"] - df["tds_prev"]) / df["dt_s"]
    df["turb_rate"] = (df["turbidity"] - df["turb_prev"]) / df["dt_s"]
    df["temp_rate"] = (df["temp_C"] - df["temp_prev"]) / df["dt_s"]

    df[["tds_rate", "turb_rate", "temp_rate"]] = df[["tds_rate", "turb_rate", "temp_rate"]].replace([np.inf, -np.inf], np.nan)
    df[["tds_rate", "turb_rate", "temp_rate"]] = df[["tds_rate", "turb_rate", "temp_rate"]].fillna(0)

    # next-step targets
    df["tds_next"] = df.groupby("run_id")["tds"].shift(-1)
    df["turb_next"] = df.groupby("run_id")["turbidity"].shift(-1)

    df = df.dropna(subset=["tds_next", "turb_next"]).reset_index(drop=True)
    return df


def evaluate_next_step(df_run, model, feature_cols):
    X = df_run[feature_cols]
    y_true_tds = df_run["tds_next"].values
    y_true_turb = df_run["turb_next"].values

    y_pred = model.predict(X)
    y_pred_tds = y_pred[:, 0]
    y_pred_turb = y_pred[:, 1]

    # baseline: next = current
    baseline_tds = df_run["tds"].values
    baseline_turb = df_run["turbidity"].values

    mae_model_tds = mean_absolute_error(y_true_tds, y_pred_tds)
    mae_model_turb = mean_absolute_error(y_true_turb, y_pred_turb)

    mae_base_tds = mean_absolute_error(y_true_tds, baseline_tds)
    mae_base_turb = mean_absolute_error(y_true_turb, baseline_turb)

    return (mae_model_tds, mae_model_turb, mae_base_tds, mae_base_turb)


def rollout_forecast(df_run_raw, model, feature_cols):
    """
    Roll forward predictions to create a full predicted curve.
    Uses real temp profile (temp_C) from the run to avoid drift from missing heat model.
    """
    df = df_run_raw.sort_values("t_rel_s").reset_index(drop=True).copy()

    # start with actual initial state
    pred_tds = [df.loc[0, "tds"]]
    pred_turb = [df.loc[0, "turbidity"]]

    # we will iterate over time points and predict next
    for i in range(len(df) - 1):
        row = df.loc[i].copy()

        # replace current state with predicted state (closed-loop simulation)
        row["tds"] = pred_tds[-1]
        row["turbidity"] = pred_turb[-1]

        # approximate rates using previous predicted point
        if i == 0:
            row["tds_rate"] = 0
            row["turb_rate"] = 0
            row["temp_rate"] = 0
        else:
            dt = df.loc[i, "dt_s"] if df.loc[i, "dt_s"] > 0 else 1.0
            row["tds_rate"] = (pred_tds[-1] - pred_tds[-2]) / dt
            row["turb_rate"] = (pred_turb[-1] - pred_turb[-2]) / dt
            row["temp_rate"] = (df.loc[i, "temp_C"] - df.loc[i-1, "temp_C"]) / dt

        X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        y_next = model.predict(X_row)[0]

        pred_tds.append(y_next[0])
        pred_turb.append(y_next[1])

    return np.array(pred_tds), np.array(pred_turb)


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    model = joblib.load(MODEL_PATH)

    feature_cols = [
        "acid_M", "h2o2_ml",
        "t_rel_s", "temp_C",
        "tds", "turbidity",
        "tds_rate", "turb_rate", "temp_rate"
    ]

    # --- evaluate next-step + baseline ---
    print("\n=== NEXT-STEP VALIDATION (per run) ===")
    for rid in sorted(df["run_id"].unique()):
        df_run = df[df["run_id"] == rid].copy()
        mae_model_tds, mae_model_turb, mae_base_tds, mae_base_turb = evaluate_next_step(df_run, model, feature_cols)

        print(f"Run {rid}:")
        print(f"  Model MAE  -> TDS: {mae_model_tds:.2f}, Turb: {mae_model_turb:.2f}")
        print(f"  Baseline   -> TDS: {mae_base_tds:.2f}, Turb: {mae_base_turb:.2f}")

    # --- rollout plot on run 4 ---
    rid = 4
    df_run4 = df[df["run_id"] == rid].copy()

    pred_tds, pred_turb = rollout_forecast(df_run4, model, feature_cols)

    t = df_run4["t_rel_s"].values
    actual_tds = df_run4["tds"].values
    actual_turb = df_run4["turbidity"].values

    # Plot TDS
    plt.figure()
    plt.plot(t, actual_tds, label="Actual TDS")
    plt.plot(t, pred_tds, label="Predicted TDS")
    plt.xlabel("Time (s)")
    plt.ylabel("TDS")
    plt.title("Run 4: Actual vs Predicted (Rollout) - TDS")
    plt.legend()
    plt.show()

    # Plot Turbidity
    plt.figure()
    plt.plot(t, actual_turb, label="Actual Turbidity")
    plt.plot(t, pred_turb, label="Predicted Turbidity")
    plt.xlabel("Time (s)")
    plt.ylabel("Turbidity")
    plt.title("Run 4: Actual vs Predicted (Rollout) - Turbidity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
