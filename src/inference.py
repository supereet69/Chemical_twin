import joblib
import numpy as np
import pandas as pd

from preprocessing import load_and_preprocess
from features import build_features

MODEL_PATH = "./models/chemical_rate_model.pkl"
DATA_PATH = "./data/data.csv"


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def forecast_step(model, FEATURES, row):
    # row is a Series with engineered columns
    X = row[FEATURES].values.reshape(1, -1)
    tds_rate, turb_rate, rec_rate = model.predict(X)[0]
    return float(tds_rate), float(turb_rate), float(rec_rate)


def evaluate(alpha=0.75, horizon_s_list=(10, 30, 60)):
    # load bundle
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    FEATURES = bundle["features"]
    TARGETS = bundle["targets"]  # not used here but kept for reference

    # load + preprocess full dataset
    df = load_and_preprocess(DATA_PATH, dt=10)

    # build engineered columns once
    df, _, _ = build_features(df)

    final_errors = []
    horizon_errors = {h: [] for h in horizon_s_list}

    for rid in df["run_id"].unique():
        df_run = df[df["run_id"] == rid].sort_values("time_s").reset_index(drop=True)

        # twin state starts at measurement
        tds = float(df_run.loc[0, "tds"])
        turb = float(df_run.loc[0, "turbidity"])
        rec = float(df_run.loc[0, "recovery"])

        for k in range(1, len(df_run)):
            dt = float(df_run.loc[k, "dt"])
            if dt <= 0 or np.isnan(dt):
                dt = 10.0

            # state row = previous measurement row, but overwrite with twin state
            state_row = df_run.loc[k - 1].copy()
            state_row["tds"] = tds
            state_row["turbidity"] = turb
            state_row["recovery"] = rec

            # rebuild features for this single row
            one = pd.DataFrame([state_row])
            one_feat, _, _ = build_features(one)
            row_feat = one_feat.iloc[0]

            # predict rates
            tds_rate, turb_rate, rec_rate = forecast_step(model, FEATURES, row_feat)

            # one-step prediction
            pred_tds = tds + tds_rate * dt
            pred_turb = turb + turb_rate * dt
            pred_rec = rec + rec_rate * dt

            # measurement at time k
            meas_tds = float(df_run.loc[k, "tds"])
            meas_turb = float(df_run.loc[k, "turbidity"])
            meas_rec = float(df_run.loc[k, "recovery"])

            # assimilation (blend predicted + measured)
            tds = alpha * pred_tds + (1 - alpha) * meas_tds
            turb = alpha * pred_turb + (1 - alpha) * meas_turb
            rec = alpha * pred_rec + (1 - alpha) * meas_rec

            # safety clamps (optional)
            tds = clamp(tds, 0, 20000)
            turb = clamp(turb, 0, 3000)
            rec = clamp(rec, -2, 2)

            # horizon evaluation (approx using 1-step pred)
            for H in horizon_s_list:
                steps_ahead = int(H / dt)
                idx_future = k + steps_ahead
                if idx_future < len(df_run):
                    true_future = float(df_run.loc[idx_future, "tds"])
                    horizon_errors[H].append(abs(pred_tds - true_future))

        true_final = df_run.iloc[-1]
        final_errors.append([
            abs(tds - float(true_final["tds"])),
            abs(turb - float(true_final["turbidity"])),
            abs(rec - float(true_final["recovery"])),
        ])

    final_errors = np.array(final_errors)

    print("\n===== DIGITAL TWIN ERROR (ASSIMILATED) =====")
    print("Final TDS MAE:", final_errors[:, 0].mean())
    print("Final Turbidity MAE:", final_errors[:, 1].mean())
    print("Final Recovery MAE:", final_errors[:, 2].mean())

    print("\n===== FORECAST HORIZON MAE (TDS) =====")
    for H in horizon_s_list:
        if len(horizon_errors[H]) > 0:
            print(f"{H}s ahead MAE:", float(np.mean(horizon_errors[H])))
        else:
            print(f"{H}s ahead MAE: N/A")


if __name__ == "__main__":
    evaluate(alpha=0.75, horizon_s_list=(10, 30, 60))
