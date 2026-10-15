"""Synthetic wind-turbine fleet: 10-min SCADA statistics and 5 Hz blade-root edgewise moments.

No fleet data is public at this level of detail, so a simple physical model plays the fleet:

- Two wind farms of six 4 MW turbines on two platforms. Platform B has a blade with lower
  structural damping in the first edgewise mode (about 1.0 Hz).
- Normal operation: gravity loading at the rotor frequency (1P) plus the turbulent response of
  the edgewise mode (a damped resonator driven by white noise scaled with wind speed and turbulence).
- Edgewise instability: while a turbine is idling in high wind with a yaw misalignment between
  20 and 50 degrees, aerodynamic damping of the edgewise mode becomes negative. If total damping
  (structural + aerodynamic) is negative, the oscillation grows exponentially and saturates in a
  limit cycle (logistic envelope).
- High-frequency signals are regenerated on demand from a seed per turbine and time stamp, which
  plays the role of the measurement database.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

FS, REC_S = 5.0, 600                      # Hz, s per record
F_EDGE = 1.0                              # Hz, first edgewise mode
PLATFORMS = {"SG-A": dict(zeta_s=0.006, zeta_aero_min=-0.003),
             "SG-B": dict(zeta_s=0.003, zeta_aero_min=-0.009)}
FARMS = {"Farm North": "SG-A", "Farm West": "SG-B"}
A_LIM, A0 = 4500.0, 40.0                  # kNm, limit-cycle and initial amplitude


def turbines():
    rows = []
    for farm, plat in FARMS.items():
        for k in range(6):
            rows.append(dict(turbine=f"{farm.split()[1][0]}{k + 1:02d}", farm=farm, platform=plat))
    return pd.DataFrame(rows)


def power_curve(v):
    p = np.clip((v - 3) / (12 - 3), 0, 1) ** 3 * 4000  # cut-in 3, rated 12 m/s, 4000 kW
    return np.where((v < 3) | (v > 25), 0.0, p)


def scada(days=14, seed=5):
    """10-min statistics for every turbine, with storms and curtailment periods."""
    rng = np.random.default_rng(seed)
    n = days * 144
    t = pd.date_range("2026-09-01", periods=n, freq="10min")
    out = []
    for farm in FARMS:
        # farm wind: AR(1) around a slowly varying mean with two storms
        base = 9 + 3 * np.sin(np.arange(n) / 144 * 2 * np.pi / 5)
        for c in (int(n * 0.25), int(n * 0.55), int(n * 0.85)):              # three storms
            base += 12 * np.exp(-0.5 * ((np.arange(n) - c) / 40) ** 2)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.97 * x[i - 1] + rng.normal(0, 0.6)  # AR(1) gusts, 10-min steps
        v_farm = np.clip(base + x, 0, None)
        curtail = np.zeros(n, bool)
        for c in rng.choice(n - 36, 8, replace=False):                 # grid curtailment: rotor idling
            curtail[c:c + int(rng.integers(12, 36))] = True
        for tb in turbines().query("farm == @farm").itertuples():
            v = np.clip(v_farm * rng.uniform(0.93, 1.07) + rng.normal(0, 0.4, n), 0, None)  # turbine-to-turbine spread
            state = np.where(v > 25, "idle", np.where(curtail, "idle", np.where(v < 3, "idle", "production")))
            maint = rng.random(n) < 0.004
            state = np.where(maint, "stop", state)
            # idling rotors drift off the wind
            yaw = np.where(state == "production", rng.normal(0, 4, n), rng.uniform(-70, 70, n))
            ti = np.clip(rng.normal(0.11, 0.02, n), 0.05, 0.25)
            out.append(pd.DataFrame(dict(ts=t, turbine=tb.turbine, farm=farm, platform=tb.platform,
                                         wind_mean=v, ti=ti, yaw_misalign=yaw, state=state,
                                         power_kw=np.where(state == "production", power_curve(v), 0.0),
                                         rotor_rpm=np.where(state == "production", np.clip(6 + v, 6, 12.5), 0.3))))
    return pd.concat(out, ignore_index=True)


def zeta_total(platform, state, wind, yaw):
    """Total damping ratio of the edgewise mode."""
    p = PLATFORMS[platform]
    if state == "production":
        za = 0.01                                              # rotating blades: positive aero damping
    else:
        shape = np.exp(-0.5 * ((abs(yaw) - 35) / 8) ** 2)       # worst near 35 deg misalignment
        # negative only above ~14 m/s
        za = 0.002 + (p["zeta_aero_min"] - 0.002) * shape * np.clip((wind - 14) / 8, 0, 1)
    return p["zeta_s"] + za


def hf_signal(row, seed):
    """5 Hz edgewise blade-root moment (kNm) for one 10-min record, plus the true instability flag."""
    rng = np.random.default_rng(seed)
    n = int(FS * REC_S)
    t = np.arange(n) / FS
    w = 2 * np.pi * F_EDGE
    z = zeta_total(row.platform, row.state, row.wind_mean, row.yaw_misalign)
    # turbulent response of a linear resonator (discretised 2nd-order system)
    r = np.exp(-z * w / FS) if z > 0 else np.exp(-0.002 * w / FS)  # pole radius; growth handled below
    a = [1, -2 * r * np.cos(w / FS), r * r]
    drive = rng.normal(0, 1, n) * (row.wind_mean ** 2) * row.ti * 14.0
    m = lfilter([1], a, drive) * (1 - r)  # (1 - r) normalises the resonant peak
    if row.state == "production":                              # gravity at 1P
        m += 2500 * np.sin(2 * np.pi * row.rotor_rpm / 60 * t + rng.uniform(0, 2 * np.pi))
    unstable = z < 0
    if unstable:
        sigma = -z * w
        env = A_LIM / (1 + (A_LIM / A0 - 1) * np.exp(-sigma * t))  # logistic: growth then limit cycle
        m += env * np.sin(w * np.sqrt(1 - z * z) * t + rng.uniform(0, 2 * np.pi))
    return m, unstable, z
