"""Machine-learning load emulator: gradient-boosted trees mapping 10-min SCADA features to fatigue DELs."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

STD = [f"{k}_{s}" for k in ("wind", "P", "wg", "pitch") for s in ("mean", "std", "min", "max")] + ["TI"]
TRANS = ["drop", "drop_rate", "pitch_rate_max", "overspeed", "P_end_start"]
TARGETS = {"DEL_Mtb": 4, "DEL_Mflap": 10, "DEL_Qlss": 4}
LABEL = {"DEL_Mtb": "Tower-base fore-aft", "DEL_Mflap": "Blade-root flap", "DEL_Qlss": "Low-speed shaft torque"}


class Emulator:
    def __init__(self, features, seed=0):
        self.features, self.seed, self.models = list(features), seed, {}

    def fit(self, df):
        for t in TARGETS:
            m = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, random_state=self.seed)
            m.fit(df[self.features].values, np.log(df[t].values))  # log target: errors relative, DEL > 0
            self.models[t] = m
        return self

    def predict(self, df):
        return pd.DataFrame({t: np.exp(m.predict(df[self.features].values).astype(float)) for t, m in self.models.items()},
                            index=df.index)


def scores(df, pred):
    """Median and 90th-percentile absolute error (%) and the ratio of estimated to true fatigue damage
    (sum of DEL^m, which is what matters for component life)."""
    rows = {}
    for t, m in TARGETS.items():
        y, p = df[t].values.astype(float), pred[t].values.astype(float)
        e = np.abs(p / y - 1) * 100
        ref = y.max()  # normalise before ^m to avoid overflow
        rows[t] = dict(median_err_pct=np.median(e), p90_err_pct=np.percentile(e, 90),
                       damage_ratio=((p / ref) ** m).sum() / ((y / ref) ** m).sum())
    return pd.DataFrame(rows).T
