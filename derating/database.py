"""Simulation database for the load emulator: case sampling, 10-min SCADA features and fatigue targets."""
import numpy as np
import pandas as pd
import rainflow
import turbine as tb

M_SLOPE = {"Mtb": 4, "Mflap": 10, "Qlss": 4}       # Wohler exponents: steel tower, composite blade, steel shaft
N_EQ = 600                                          # 1 Hz equivalent over 10 min
P = tb.P_RATED  # W


def del_1hz(x, m):
    d = sum(n * rng ** m for rng, mean, n, i0, i1 in rainflow.extract_cycles(x))
    return (d / N_EQ) ** (1 / m)


def ti_ntm(V, rng, scale=(0.6, 1.3)):
    """IEC normal turbulence model (class B, I_ref = 0.14) scaled by a random site factor."""
    return 0.14 * (0.75 * V + 5.6) / V * rng.uniform(*scale, V.size)


def sample_cases(n_normal, n_derate, seed):
    rng = np.random.default_rng(seed)
    rows = []
    V = rng.uniform(4, 25, n_normal)
    for v, ti, sh in zip(V, ti_ntm(V, rng), rng.uniform(0.1, 0.3, n_normal)):
        rows.append(dict(kind="normal", V=v, TI_in=ti, shear=sh, p0=1.0, p1=1.0, rate=0.0, t0=1e9))
    V = rng.uniform(4, 25, n_derate)
    for v, ti, sh in zip(V, ti_ntm(V, rng), rng.uniform(0.1, 0.3, n_derate)):
        p0 = 1.0 if rng.random() < 0.6 else rng.uniform(0.5, 0.9)         # unconstrained or already curtailed
        p1 = rng.uniform(0.1, p0 - 0.1)
        rate = 10 ** rng.uniform(np.log10(0.002), np.log10(0.2))          # 0.2 %/s to 20 %/s of rated
        rows.append(dict(kind="derate", V=v, TI_in=ti, shear=sh, p0=p0, p1=p1, rate=rate, t0=rng.uniform(60, 480)))
    return pd.DataFrame(rows)


def scada_features(r):
    """10-min statistics a load emulator sees in standard operation, plus transient descriptors from the
    1 Hz power, pitch and speed signals (no access to the operator set-point is assumed)."""
    f = {}
    for k, sig in (("wind", r["wind"]), ("P", r["P"] / 1e3), ("wg", r["wg"]), ("pitch", r["pitch"])):
        f[f"{k}_mean"] = sig.mean(1); f[f"{k}_std"] = sig.std(1)
        f[f"{k}_min"] = sig.min(1); f[f"{k}_max"] = sig.max(1)
    f["TI"] = f["wind_std"] / f["wind_mean"]
    std = pd.DataFrame(f)
    tr = transient_features(r["P"], r["pitch"], r["wg"])
    return std, tr


def smooth(x, w):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda s: np.convolve(s, k, mode="same"), 1, x)


def transient_features(Pw, pitch, wg):
    """Descriptors of a power reduction from 1 Hz signals: size and steepest rate of the drop in
    10-s mean power, the pitch rate at that moment, and rotor overspeed relative to the record mean."""
    Ps = smooth(Pw / P, 10)[:, 5:-5]
    ps = smooth(pitch, 5)[:, 5:-5]
    dP5 = Ps[:, 5:] - Ps[:, :-5]                      # change over 5 s
    run_max = np.maximum.accumulate(Ps, axis=1)
    drop = (run_max - Ps).max(1)                      # largest drop from a previous maximum
    rate = np.clip(-dP5.min(1) / 5, 0, None)          # steepest decrease (fraction of rated per s)
    pr = (ps[:, 5:] - ps[:, :-5]).max(1) / 5          # steepest pitch increase (deg/s)
    over = (wg.max(1) - np.median(wg, 1)) / tb.W_RATED_G * 30 / np.pi * 100  # % of rated speed (wg in rpm)
    return pd.DataFrame(dict(drop=drop, drop_rate=rate, pitch_rate_max=pr, overspeed=over,
                             P_end_start=(Ps[:, -30:].mean(1) - Ps[:, :30].mean(1))))


def run_cases(cases, prm_cols=(), seed=0, batch=200, keep_series=False):
    """Simulate every case. Columns named in prm_cols (e.g. f_twr, cp_scale) override model parameters
    per case, as for the perturbed field turbines."""
    out, series = [], []
    for b0 in range(0, len(cases), batch):
        c = cases.iloc[b0:b0 + batch]
        prm = tb.default_params(shear=c.shear.values, **{k: c[k].values for k in prm_cols})
        r = tb.simulate(c.V.values, c.TI_in.values, c.p0.values * P, c.p1.values * P, c.rate.values * P,
                        c.t0.values, prm=prm, seed=seed + b0)
        std, tr = scada_features(r)
        tgt = pd.DataFrame({f"DEL_{k}": [del_1hz(x, m) for x in r[k]] for k, m in M_SLOPE.items()})
        tgt["Mtb_range"] = np.ptp(r["Mtb"], axis=1)          # extreme-load indicator (kNm)
        out.append(pd.concat([c.reset_index(drop=True), std, tr, tgt], axis=1))
        if keep_series:
            series.append(r)
    return pd.concat(out, ignore_index=True), series
