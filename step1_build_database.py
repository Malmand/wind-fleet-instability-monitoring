"""Step 1: build the fleet measurement database (two farms, twelve turbines, 14 days of 10-min SCADA).

Usage:  python step1_build_database.py
"""
import os

import numpy as np

import db
import fleet_sim


def main():
    s = fleet_sim.scada()
    rng = np.random.default_rng(1)
    drop = rng.random(len(s)) < 0.015                         # records never received
    s = s[~drop].copy()
    bad = rng.random(len(s)) < 0.002                          # wind sensor faults
    s.loc[bad, "wind_mean"] = -999.0
    os.makedirs(os.path.dirname(db.DB), exist_ok=True)
    if os.path.exists(db.DB):
        os.remove(db.DB)
    with db.connect() as con:
        s.assign(ts=s.ts.astype(str)).to_sql("scada", con, index=False)
        con.execute("CREATE INDEX ix_scada ON scada(turbine, ts)")
    full = fleet_sim.scada()
    print(f"{len(s):,} records stored of {len(full):,} expected; {int(bad.sum())} with wind-sensor faults")


if __name__ == "__main__":
    main()
