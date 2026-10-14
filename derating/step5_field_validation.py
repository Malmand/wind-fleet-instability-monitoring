"""Step 5: validation against 'field' data from a 0-series prototype.

Real 0-series measurements are confidential, so a stand-in prototype is simulated with the same model but
with parameters the emulator does not know: a softer, better-damped tower, 3% lower rotor efficiency,
retuned pitch gains, a nacelle anemometer that reads 4% low, and strain-gauge calibration errors. One week
of 10-min records is generated with realistic grid-service de-ratings, the events are identified from
the SCADA signals alone, and measured fatigue loads are compared with the emulator estimates.
"""
import pickle
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import pandas as pd
import database as db
import mlle
import turbine as tb

OUT = Path(__file__).parent / "results"
PROTO = dict(f_twr=0.305, zeta_twr=0.015, cp_scale=0.97, gain_scale=0.85)
GAUGE = {"DEL_Mtb": 1.02, "DEL_Mflap": 0.97, "DEL_Qlss": 1.01}   # calibration error of the load channels
ANEMO = 0.96                                                    # nacelle anemometer reads low behind the rotor


def field_week(n=1008, seed=7):
    rng = np.random.default_rng(seed)
    # mean wind: slowly varying (AR(1), 10-min steps) with a Weibull-like marginal (A = 9.5 m/s, k = 2.1)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.985 * z[i - 1] + np.sqrt(1 - 0.985 ** 2) * rng.normal()
    u = ndtr(z)
    V = np.clip(9.5 * (-np.log(1 - np.clip(u, 1e-4, 1 - 1e-4))) ** (1 / 2.1), 3.5, 25)
    shear = np.clip(0.2 + 0.08 * np.sin(np.arange(n) * 2 * np.pi / 144) + 0.03 * rng.normal(size=n), 0.05, 0.35)
    rows = []
    for i in range(n):
        ti = db.ti_ntm(np.array([V[i]]), rng)[0]
        r = dict(kind="normal", V=V[i], TI_in=ti, shear=shear[i], p0=1.0, p1=1.0, rate=0.0, t0=1e9, service="none")
        if rng.random() < 0.12 and available_power(V[i]) > 0.15:
            svc = rng.choice(["curtailment order", "aFRR activation", "fast frequency reserve"], p=[0.3, 0.5, 0.2])
            rate = {"curtailment order": 0.002, "aFRR activation": rng.uniform(0.003, 0.01),
                    "fast frequency reserve": rng.uniform(0.05, 0.2)}[svc]
            p0 = 1.0 if rng.random() < 0.8 else rng.uniform(0.6, 0.9)
            # orders are set relative to the power the turbine can deliver at that moment
            p1 = min(p0, available_power(V[i])) * rng.uniform(0.2, 0.8)
            r.update(kind="derate", service=svc, p0=p0, p1=p1, rate=rate,
                     t0=rng.uniform(60, 450))
        rows.append(r)
    c = pd.DataFrame(rows)
    for k, v in PROTO.items():
        c[k] = v
    return c


def available_power(V):
    """Steady available power (fraction of rated) of the design turbine at mean wind V."""
    return np.minimum(1.0, 0.5 * tb.RHO * tb.A * V ** 3 * tb.CP_MAX * tb.ETA / tb.P_RATED)


def identify_events(Pw, pitch, wg):
    """De-rating events from 1 Hz SCADA alone. A commanded reduction cuts generator torque first, so the
    power falls while the rotor holds or gains speed and the blades pitch towards feather. A turbulent
    lull lowers power together with rotor speed, and a gust above rated raises pitch at constant power.
    Rule: within one (or three) minutes the 10-s mean power falls by at least max(5% of rated, 25% of its level),
    the rotor does not slow by more than 1%, pitch or speed rises (by 0.5 deg or 2%), and the power stays
    down for as long again afterwards."""
    Ps = db.smooth(Pw / tb.P_RATED, 10)
    bs = db.smooth(pitch, 10)
    ws = db.smooth(wg, 10)
    n = Ps.shape[1]
    flag = np.zeros(Ps.shape[0], bool); start = np.full(Ps.shape[0], np.nan)
    for w in (60, 180):                       # fast orders, and slow ramps such as curtailment orders
        for t in range(10, n - 2 * w - 10, 2):
            thr = np.maximum(0.05, 0.25 * Ps[:, t])
            dP = Ps[:, t + w] - Ps[:, t]
            dw = ws[:, t + w] / ws[:, t] - 1
            db_ = bs[:, t + w] - bs[:, t]
            after = Ps[:, t + w:t + 2 * w].mean(1)
            hit = (dP <= -thr) & (dw >= -0.01) & ((db_ >= 0.5) | (dw >= 0.02)) & (after <= Ps[:, t] - thr)
            new = hit & ~flag
            start[new] = t
            flag |= hit
    return flag, start


if __name__ == "__main__":
    cases = field_week()
    out, series = [], []
    for b0 in range(0, len(cases), 200):
        c = cases.iloc[b0:b0 + 200]
        prm = tb.default_params(shear=c.shear.values, **{k: c[k].values for k in PROTO})
        r = tb.simulate(c.V.values, c.TI_in.values, c.p0.values * tb.P_RATED, c.p1.values * tb.P_RATED,
                        c.rate.values * tb.P_RATED, c.t0.values, prm=prm, seed=90000 + b0)
        r["wind"] = r["wind"] * ANEMO
        std, tr = db.scada_features(r)
        tgt = pd.DataFrame({f"DEL_{k}": [db.del_1hz(x, m) * GAUGE[f'DEL_{k}'] for x in r[k]]
                            for k, m in db.M_SLOPE.items()})
        flag, start = identify_events(r["P"], r["pitch"], r["wg"])
        ev = pd.DataFrame(dict(detected=flag, t_detect=start))
        out.append(pd.concat([c.reset_index(drop=True), std, tr, tgt, ev], axis=1))
        if b0 == 0:
            series.append({k: v[:60] for k, v in r.items()})
    f = pd.concat(out, ignore_index=True)
    f["P_avail"] = available_power(f.V)
    # a commanded reduction that actually cuts output: below the available power by at least 5% of rated
    f["effective"] = (f.kind == "derate") & (np.minimum(f.p0, f.P_avail) - f.p1 >= 0.05)
    models = pickle.load(open(OUT / "emulators.pkl", "rb"))
    names = list(models)
    for tag, name in (("A", names[0]), ("C", names[2])):
        p = models[name].predict(f)
        for t in mlle.TARGETS:
            f[f"pred{tag}_{t}"] = p[t]
    f.to_csv(OUT / "field_validation.csv", index=False)
    pickle.dump(series[0], open(OUT / "field_example_series.pkl", "wb"))

    tp = (f.detected & f.effective).sum(); fp = (f.detected & ~f.effective).sum(); fn = (~f.detected & f.effective).sum()
    print(f"records {len(f)}, commanded de-ratings {int((f.kind == 'derate').sum())}, effective {int(f.effective.sum())}")
    print(f"detection: {tp} found, {fn} missed, {fp} false alarms "
          f"(precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f})")
    rows = []
    for subset, mask in (("all records", np.ones(len(f), bool)), ("detected de-rating records", f.detected.values),
                         ("other records", ~f.detected.values)):
        for tag in "AC":
            s = mlle.scores(f[mask], f.loc[mask, [f"pred{tag}_{t}" for t in mlle.TARGETS]].set_axis(list(mlle.TARGETS), axis=1))
            s["model"], s["subset"] = tag, subset
            rows.append(s.reset_index().rename(columns={"index": "target"}))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "field_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
