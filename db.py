"""SQLite measurement database and SQL query functions.

Tables
  scada(ts, turbine, farm, platform, wind_mean, ti, yaw_misalign, state, power_kw, rotor_rpm)
  results(ts, turbine, qc_ok, qc_issues, spikes_removed, event, damage, del_kNm,
          start_s, duration_s, peak_kNm, freq_hz, zeta_est)
The high-frequency signals are fetched with get_signal(), which plays the role of the
measurement store (regenerated deterministically, with injected sensor faults).
"""
import os
import sqlite3
import zlib

import numpy as np
import pandas as pd

import fleet_sim

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "fleet.sqlite")


def connect():
    return sqlite3.connect(DB)


def records(con, start=None, end=None, farm=None, state=None, min_wind=None):
    """Parameterised query of 10-min records."""
    sql = "SELECT * FROM scada WHERE 1=1"  # so every filter can start with AND
    args = []
    for col, op, val in [("ts", ">=", start), ("ts", "<", end), ("farm", "=", farm), ("state", "=", state),
                         ("wind_mean", ">=", min_wind)]:
        if val is not None:
            sql += f" AND {col} {op} ?"
            args.append(str(val) if col == "ts" else val)
    return pd.read_sql_query(sql + " ORDER BY turbine, ts", con, params=args)


def events_by_week(con):
    """Weekly event counts, durations and peak amplitudes per farm and turbine."""
    return pd.read_sql_query("""
        SELECT strftime('%Y-W%W', r.ts) AS week, s.farm, r.turbine,
               COUNT(*) AS events, ROUND(SUM(r.duration_s) / 3600.0, 2) AS event_hours,
               ROUND(MAX(r.peak_kNm)) AS max_peak_kNm
        FROM results r JOIN scada s ON s.ts = r.ts AND s.turbine = r.turbine
        WHERE r.event = 1
        GROUP BY week, s.farm, r.turbine ORDER BY week, s.farm, r.turbine""", con)


def availability(con):
    """Share of expected 10-min records present and passing quality checks, per turbine."""
    return pd.read_sql_query("""
        SELECT s.turbine, s.farm, COUNT(*) AS expected,
               SUM(CASE WHEN r.qc_ok = 1 THEN 1 ELSE 0 END) AS usable
        FROM scada s LEFT JOIN results r ON s.ts = r.ts AND s.turbine = r.turbine
        GROUP BY s.turbine, s.farm ORDER BY s.turbine""", con)


def get_signal(row):
    """Edgewise moment for one record, with sensor faults injected at fixed rates."""
    seed = zlib.crc32(f"{row.turbine}{row.ts}".encode())  # same record -> same signal every call
    x, unstable, z = fleet_sim.hf_signal(row, seed)
    rng = np.random.default_rng(seed + 1)
    u = rng.random()
    if u < 0.01:                                   # flat-lined sensor
        x[:] = x[0]
    elif u < 0.02:                                 # lost samples
        x = x[: int(len(x) * 0.6)]
    if rng.random() < 0.05:                        # a few spikes
        k = rng.choice(len(x), 3, replace=False)
        x[k] += rng.choice([-1, 1], 3) * 9000
    return x, unstable, z
