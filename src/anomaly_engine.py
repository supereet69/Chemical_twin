import joblib
import numpy as np
import pandas as pd


class AnomalyEngine:
    """
    Plug-and-play anomaly engine.
    Supports:
    - rule-based anomaly (current)
    - ML anomaly model (friend's model) later
    """

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path

        if model_path is not None:
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None

    def is_ready(self):
        return self.model is not None

    def predict_anomaly_score(self, state_row: dict) -> float:
        """
        Returns anomaly score in [0, 1] (recommended standard).
        If model not available -> return 0.0
        """
        if self.model is None:
            return 0.0

        # Convert dict -> dataframe
        df = pd.DataFrame([state_row])

        # ---- IMPORTANT ----
        # Your friend must tell which columns the anomaly model expects.
        # We'll handle both cases:
        # 1) model expects raw columns directly
        # 2) model expects engineered columns
        # For now we try raw columns only.
        try:
            score = self.model.predict_proba(df)[:, 1][0]  # if classifier
            return float(score)
        except Exception:
            try:
                score = self.model.predict(df)[0]  # if regression score
                return float(score)
            except Exception:
                return 0.0

    def apply_effect(self, rates, anomaly_score: float):
        """
        Convert anomaly score into effect on rates.
        rates = (tds_rate, turb_rate, rec_rate)

        You can tune this later.
        """
        tds_rate, turb_rate, rec_rate = rates

        if anomaly_score < 0.5:
            return rates

        # amplify rates as anomaly impact (simple demo)
        factor = 1.0 + 1.5 * (anomaly_score - 0.5)  # max ~1.75
        return (
            tds_rate * factor,
            turb_rate * factor,
            rec_rate
        )
