"""Step 6: check on real measurements from the Björkö 45 kW research turbine (Chalmers).

Data: Fogelström et al., Björkö Wind Turbine Version 1 (45 kW) high-frequency SHM data, Zenodo,
doi:10.5281/zenodo.8230330 (CC BY 4.0). The 20 Hz file was condensed to 10-min records with
tools/prepare_bjorko.py: SCADA statistics, the logged frequency-control signals and damage-equivalent
loads of the calibrated tower-base moments (rotated to fore-aft and side-side with the yaw angle) and the
shaft-torque strain gauge. During the measurement campaign the turbine took part in frequency control
tests (FCR-N, FCR-D) and manual power shedding, logged as Pwaste, the power the controller removes.

Questions:
1. Do the measured loads change when power is shed, compared with normal records in the same wind?
2. Does an emulator trained only on normal operation mis-estimate those records, and does adding them
   (with transient and set-point features) help? Leave-one-day-out cross-validation, so no day is in
   both training and test.
3. Does the SCADA-only detection rule of step 5 transfer to this turbine?
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

HERE = Path(__file__).parent
OUT = HERE / "results"
T = {"DEL_TFA": 4, "DEL_TSS": 4, "DEL_RST2": 4}
STD = [f"{k}_{s}" for k in ("WS30", "XTurbSpeed1", "PAB1", "P") for s in ("mean", "std", "min", "max")] + ["TI30"]
EXT = STD + ["P_drop_rel", "P_ramp_max", "pitch_rate_max", "avail_ratio", "Pwaste_mean", "Pwaste_max"]
AMB = ["WS30_mean", "WS30_std"]


def load():
    d = pd.read_csv(HERE / "data" / "bjorko_10min.csv.gz", parse_dates=["time_utc"])
    r = d[(d.frac_running > 0.99) & (d.n >= 0.95 * 12000) & (d.WS30_mean > 0.5)].copy()
    r["shed"] = r.Pwaste_mean > 500                     # more than about 1% of rated power shed on average
    r["day"] = r.time_utc.dt.date.astype(str)
    r["TI30"] = r.WS30_std / r.WS30_mean
    r["avail_ratio"] = (r.P_mean / r.MaxPwrEst_mean).clip(0, 2)
    return r.reset_index(drop=True)


def model():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, min_child_weight=3,
                        random_state=0)


def lodo(r, feats, train_mask, targets):
    """Leave-one-day-out predictions of log DEL."""
    pred = pd.DataFrame(index=r.index, columns=targets, dtype=float)
    for day in r.day.unique():
        te = (r.day == day).values
        tr = ~te & train_mask
        for t in targets:
            m = model().fit(r.loc[tr, feats], np.log(r.loc[tr, t]))
            pred.loc[te, t] = np.exp(m.predict(r.loc[te, feats]).astype(float))
    return pred


def scores(y, p):
    rows = {}
    for t, m in T.items():
        e = np.abs(p[t] / y[t] - 1) * 100
        ref = y[t].max()
        rows[t] = dict(median_err_pct=e.median(), p90_err_pct=e.quantile(0.9),
                       damage_ratio=((p[t] / ref) ** m).sum() / ((y[t] / ref) ** m).sum())
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    r = load()
    print(f"{len(r)} running 10-min records on {r.day.nunique()} days; {int(r.shed.sum())} with power shedding "
          f"on {r[r.shed].day.nunique()} days")

    # 1. load change: measured DEL over the DEL expected from ambient wind alone (model trained on normal records)
    exp = lodo(r, AMB, ~r.shed.values, list(T))
    for t in T:
        r[f"{t}_rel"] = r[t] / exp[t]
    eff = r.groupby("shed")[[f"{t}_rel" for t in T]].median()
    print("\nmeasured / expected DEL (median):\n", eff.round(2).to_string())

    # 2. emulator A (normal operation only) vs C (all records, transient and set-point features)
    pA = lodo(r, STD, ~r.shed.values, list(T))
    pC = lodo(r, EXT, np.ones(len(r), bool), list(T))
    out = []
    for name, p in (("A: normal operation only", pA), ("C: + shedding records and features", pC)):
        for sub, mask in (("shedding", r.shed.values), ("normal", ~r.shed.values)):
            s = scores(r[mask], p[mask]); s["model"], s["subset"] = name, sub
            out.append(s.reset_index().rename(columns={"index": "target"}))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT / "bjorko_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", res.round(2).to_string(index=False))
    r.assign(**{f"predA_{t}": pA[t] for t in T}, **{f"predC_{t}": pC[t] for t in T}).to_csv(
        OUT / "bjorko_predictions.csv", index=False)

    # 3. does the step-5 idea (pitch activity below rated wind flags a reduction) transfer?
    det = (r.PAB1_std > 0.3) & (r.WS30_mean < 9)
    tp, fp, fn = int((det & r.shed).sum()), int((det & ~r.shed).sum()), int((~det & r.shed).sum())
    print(f"\npitch-activity rule: {tp} found, {fn} missed, {fp} false alarms "
          f"(precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f})")
    pd.Series(dict(tp=tp, fp=fp, fn=fn)).to_csv(OUT / "bjorko_detection.csv")
