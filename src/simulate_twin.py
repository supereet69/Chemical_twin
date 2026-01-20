# simulate_twin.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from config import DATA_PATH, MODEL_PATH
from features import add_features, FEATURE_COLS


def simulate_run(model, init_row: pd.Series, n_steps: int) -> pd.DataFrame:
    current = init_row.copy()
    history = [current.copy()]

    for _ in range(n_steps):
        X_curr = pd.DataFrame([current[FEATURE_COLS]], columns=FEATURE_COLS)
        tds_delta, turb_delta = model.predict(X_curr)[0]


        dt = current.get("dt_s", 1.0)
        if pd.isna(dt) or dt <= 0:
            dt = 1.0

        # apply delta prediction
        tds_next = float(current["tds"] + tds_delta)
        turb_next = float(current["turbidity"] + turb_delta)

        # safety clamps (avoid unrealistic jumps)
        max_tds_jump = 250
        max_turb_jump = 150

        tds_next = np.clip(tds_next, current["tds"] - max_tds_jump, current["tds"] + max_tds_jump)
        turb_next = np.clip(turb_next, current["turbidity"] - max_turb_jump, current["turbidity"] + max_turb_jump)

        # no negative values
        tds_next = max(tds_next, 0)
        turb_next = max(turb_next, 0)

        next_row = current.copy()
        next_row["tds"] = tds_next
        next_row["turbidity"] = turb_next
        next_row["t_rel_s"] = current["t_rel_s"] + dt

        # update rates based on simulated values
        next_row["tds_rate"] = (tds_next - current["tds"]) / dt
        next_row["turb_rate"] = (turb_next - current["turbidity"]) / dt

        history.append(next_row)
        current = next_row

    return pd.DataFrame(history)


def main():
    df_raw = pd.read_csv(DATA_PATH)
    df_feat = add_features(df_raw)

    run_id_demo = 4
    run_df = df_feat[df_feat["run_id"] == run_id_demo].reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    init_row = run_df.iloc[0]
    n_steps = len(run_df) - 1

    sim_df = simulate_run(model, init_row, n_steps)

    t_exp = run_df["t_rel_s"].values
    t_sim = sim_df["t_rel_s"].values

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_exp, run_df["tds"], label="Experimental TDS")
    plt.plot(t_sim, sim_df["tds"], "--", label="Simulated TDS")
    plt.xlabel("Time (s)")
    plt.ylabel("TDS")
    plt.legend()
    plt.title("TDS vs Time")

    plt.subplot(1, 2, 2)
    plt.plot(t_exp, run_df["turbidity"], label="Experimental Turbidity")
    plt.plot(t_sim, sim_df["turbidity"], "--", label="Simulated Turbidity")
    plt.xlabel("Time (s)")
    plt.ylabel("Turbidity")
    plt.legend()
    plt.title("Turbidity vs Time")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
