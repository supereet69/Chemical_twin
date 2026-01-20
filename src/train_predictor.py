import os
import joblib
import numpy as np

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import load_and_preprocess
from features import build_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "chemical_rate_model.pkl")


def train():
    df = load_and_preprocess(DATA_PATH, dt=10)
    df, FEATURES, TARGETS = build_features(df)

    runs = df["run_id"].unique()
    np.random.shuffle(runs)

    split = int(0.75 * len(runs))
    train_runs = runs[:split]
    test_runs = runs[split:]

    train_df = df[df["run_id"].isin(train_runs)]
    test_df = df[df["run_id"].isin(test_runs)]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGETS]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGETS]

    base = XGBRegressor(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
    )

    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\n===== RATE MODEL VALIDATION (GROUP SPLIT) =====")
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], pred[:, i]))
        print(f"{t}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    bundle = {
        "model": model,
        "features": FEATURES,
        "targets": TARGETS
    }

    joblib.dump(bundle, MODEL_PATH)
    print("\nSaved bundle:", MODEL_PATH)

if __name__ == "__main__":
    train()
