"""Step 4: train the load emulator with and without de-rating transients, and test sensitivity."""
from pathlib import Path
import numpy as np
import pandas as pd
import database as db
import mlle
import turbine as tb

OUT = Path(__file__).parent / "results"


def split(df, seed=0):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), bool)
    for kind, frac in (("normal", 0.2), ("derate", 0.3)):
        idx = np.where(df.kind == kind)[0]
        test[rng.choice(idx, int(frac * idx.size), replace=False)] = True
    return df[~test], df[test]


def sweep_cases():
    """Controlled sweeps around a reference de-rating (full power to 30% at 5%/s), common random seeds."""
    rows = []
    base = dict(TI_in=None, shear=0.2, p0=1.0, p1=0.3, rate=0.05, t0=200.0)
    for V in (8.0, 12.0, 18.0):
        ti = 0.14 * (0.75 * V + 5.6) / V
        for seed in range(6):
            rows.append(dict(base, V=V, TI_in=ti, sweep="none", x=np.nan, seed=seed, kind="normal", rate=0.0, t0=1e9, p1=1.0))
            for r in np.geomspace(0.002, 0.2, 9):
                rows.append(dict(base, V=V, TI_in=ti, sweep="rate", x=r, seed=seed, kind="derate", rate=r))
            for p1 in np.linspace(0.1, 0.8, 8):
                rows.append(dict(base, V=V, TI_in=ti, sweep="p1", x=p1, seed=seed, kind="derate", p1=p1))
            for p0 in np.linspace(0.5, 1.0, 6):
                rows.append(dict(base, V=V, TI_in=ti, sweep="p0", x=p0, seed=seed, kind="derate", p0=p0, p1=min(0.3, p0 - 0.1)))
    return pd.DataFrame(rows)


def run_sweeps():
    """Each batch shares the same turbulence seed, so differences come from the transient only."""
    c = sweep_cases()
    out = []
    for (V, seed), g in c.groupby(["V", "seed"]):
        d, _ = db.run_cases(g, seed=int(5000 + seed), batch=len(g))   # same seed across the batch rows
        out.append(d)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "sim_database.csv")
    tr, te = split(df)
    trN = tr[tr.kind == "normal"]
    models = {
        "A: normal operation only": mlle.Emulator(mlle.STD).fit(trN),
        "B: + de-rating simulations": mlle.Emulator(mlle.STD).fit(tr),
        "C: + de-rating simulations and transient features": mlle.Emulator(mlle.STD + mlle.TRANS).fit(tr),
    }
    rows = []
    preds = {}
    for name, m in models.items():
        p = m.predict(te)
        preds[name] = p
        for kind in ("normal", "derate"):
            s = mlle.scores(te[te.kind == kind], p[te.kind == kind])
            s["model"], s["test_set"] = name, kind
            rows.append(s.reset_index().rename(columns={"index": "target"}))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "emulator_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    te.assign(**{f"pred_A_{t}": preds[list(models)[0]][t] for t in mlle.TARGETS},
              **{f"pred_C_{t}": preds[list(models)[2]][t] for t in mlle.TARGETS}).to_csv(OUT / "emulator_test_predictions.csv", index=False)

    sw = run_sweeps()
    pC = models[list(models)[2]].predict(sw)
    pA = models[list(models)[0]].predict(sw)
    for t in mlle.TARGETS:
        sw[f"predC_{t}"] = pC[t]; sw[f"predA_{t}"] = pA[t]
    sw.to_csv(OUT / "sensitivity_sweeps.csv", index=False)
    # store the final model for the field validation step
    import pickle
    pickle.dump(models, open(OUT / "emulators.pkl", "wb"))
