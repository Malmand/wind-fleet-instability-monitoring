"""Part 4: the conditions behind edgewise instability, on real SCADA from a UK wind farm.

Part 1 runs on a simulated fleet, because blade-load data from a fleet is not public. The conditions that
make a parked blade unstable (rotor stopped, strong wind, rotor well off the wind) only need SCADA,
so they can be checked on real data. Data: Penmanshiel, five Senvion MM82 (2 MW, 82 m rotor),
second half of 2016 (Plumley, Zenodo, doi:10.5281/zenodo.8253010, CC BY 4.0), condensed by
tools/prepare_penmanshiel.py.

This measures exposure, not events: there are no blade loads in this dataset.
"""
import os

import numpy as np
import pandas as pd

import plotstyle as ps
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
WIND_MIN = 14.0            # m/s, same limit as the simulated damping map of Part 1
MIS_BAND = (20.0, 50.0)    # deg, misalignment band where the parked blade is least damped (Part 1 map)
PARKED_RPM, PARKED_KW = 1.0, 10.0   # rotor below 1 rpm and no power export = parked or idling


def load():
    d = pd.read_csv(os.path.join(HERE, "data", "penmanshiel_10min.csv.gz"), parse_dates=["ts"])
    s = pd.read_csv(os.path.join(HERE, "data", "penmanshiel_stops.csv.gz"), parse_dates=["start", "end"])
    # the turbines were being commissioned until mid-August: start on the first day all five report
    # at least 90 % of their records
    daily = d.assign(ok=d.wind.notna()).groupby([d.ts.dt.floor("D"), "turbine"]).ok.mean().unstack()
    start = daily[(daily >= 0.9).all(axis=1)].index.min()
    d = d[d.ts >= start].copy()
    # the vane measures the wind relative to the nacelle; in this export the wind direction is
    # nacelle position + vane, so the vane is the yaw misalignment
    d["mis"] = (d.vane + 180) % 360 - 180
    d["parked"] = (d.rotor_rpm < PARKED_RPM) & (d.power_kw < PARKED_KW)
    d["exposed"] = d.parked & (d.wind > WIND_MIN)
    d["risk"] = d.exposed & d.mis.abs().between(*MIS_BAND)
    return d, s, start


def stop_reason(d, s):
    """Stop message active during each exposed record ('no stop logged' if none)."""
    out = pd.Series("no stop logged", index=d.index)
    for i, r in d.iterrows():
        m = s[(s.turbine == r.turbine) & (s.start <= r.ts + pd.Timedelta("10min")) & (s.end >= r.ts)]
        if len(m):
            out[i] = m.message.iloc[0]
    return out


def episodes(d):
    """Join risk-zone records of one turbine into episodes (gaps of up to 1 hour allowed)."""
    rows = []
    for t, g in d[d.risk].groupby("turbine"):
        new = g.ts.diff() > pd.Timedelta("1h")
        for _, e in g.groupby(new.cumsum()):
            # a frozen nacelle while the wind veers is the sign of a stopped yaw system
            span = d[(d.turbine == t) & d.ts.between(e.ts.min(), e.ts.max())]
            rows.append(dict(turbine=t, start=e.ts.min(), hours=len(e) / 6,
                             wind_max=e.wind.max(), mis_mean=e.mis.abs().mean(),
                             nacelle_range_deg=span.nacelle.max() - span.nacelle.min(),
                             wind_dir_range_deg=span.wind_dir.max() - span.wind_dir.min(),
                             reason=e.reason.mode().iloc[0]))
    return pd.DataFrame(rows).sort_values("hours", ascending=False)


def figure(d, ep, path):
    fig, ax = ps.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a = ax[0]
    a.add_patch(Rectangle((WIND_MIN, MIS_BAND[0]), 12, MIS_BAND[1] - MIS_BAND[0], fc="#F0E442", alpha=0.35, ec="none"))
    a.text(WIND_MIN + 0.4, MIS_BAND[1] - 2, f"risk zone:\n>{WIND_MIN:.0f} m/s, {MIS_BAND[0]:.0f}-{MIS_BAND[1]:.0f}°",
           fontsize=6.5, va="top")
    cols = [ps.C[k] for k in ("blue", "orange", "green", "purple", "sky")]
    pk = d[d.parked & d.wind.notna()]
    for i, (t, g) in enumerate(pk.groupby("turbine")):
        a.scatter(g.wind, g.mis.abs(), s=4, lw=0, alpha=0.7, color=cols[i], marker="osD^v"[i], label=f"WT{t}")
    a.set(xlabel="Wind speed (m/s)", ylabel="Yaw misalignment (°)", xlim=(0, 26), ylim=(0, 180), yticks=range(0, 181, 30))
    a.legend(frameon=False, loc="upper right", markerscale=2, fontsize=6)
    ps.panel(a, "A", x=-0.2)

    # B: the longest risk episode, with some context either side
    top = ep.iloc[0]
    b = ax[1]
    e = d[(d.turbine == top.turbine) & d.ts.between(top.start - pd.Timedelta("6h"), top.start + pd.Timedelta("20h"))]
    h = (e.ts - e.ts.iloc[0]).dt.total_seconds() / 3600
    b.fill_between(h, 0, 26, where=e.parked.values, color="#999999", alpha=0.15, lw=0, step="mid")
    b.plot(h, e.wind, color=ps.C["blue"], lw=1, label="Wind speed")
    b.axhline(WIND_MIN, color="grey", lw=0.6, ls=":")
    b.set(ylabel="Wind speed (m/s)", ylim=(0, 26),
          xlabel=f"Hours from {e.ts.iloc[0]:%d %b %Y, %H:%M} (UTC), WT{top.turbine}; grey = parked")
    c = b.twinx()
    c.spines["right"].set_visible(True)
    c.plot(h, e.wind_dir, color=ps.C["vermillion"], lw=1, ls="--", label="Wind direction")
    c.plot(h, e.nacelle, color=ps.C["black"], lw=1, ls="-.", label="Nacelle direction")
    c.set(ylabel="Direction from north (°)", ylim=(180, 260))
    l1, l2 = b.get_legend_handles_labels(), c.get_legend_handles_labels()
    b.legend(l1[0] + l2[0], l1[1] + l2[1], frameon=False, loc="lower center", ncol=3, fontsize=6)
    ps.panel(b, "B", x=-0.12)
    ps.save(fig, path)


def main():
    d, s, start = load()
    ex = d[d.exposed].copy()
    ex["reason"] = stop_reason(ex, s)
    d = d.join(ex.reason)
    ep = episodes(d)

    days = (d.ts.max() - start).total_seconds() / 86400
    kpi = d.groupby("turbine").agg(records=("ts", "size"), missing_pct=("wind", lambda x: 100 * x.isna().mean()),
                                   parked_pct=("parked", lambda x: 100 * x.mean()),
                                   exposed_h=("exposed", lambda x: x.sum() / 6), risk_h=("risk", lambda x: x.sum() / 6))
    kpi["exposed_unlogged_h"] = ex[ex.reason == "no stop logged"].groupby("turbine").size().reindex(kpi.index, fill_value=0) / 6
    kpi.round(2).to_csv(os.path.join(HERE, "results", "penmanshiel_exposure.csv"))
    ep.round(2).to_csv(os.path.join(HERE, "results", "penmanshiel_risk_episodes.csv"), index=False)
    figure(d, ep, os.path.join(HERE, "figures", "fig7_penmanshiel_exposure.png"))

    print(f"Period {start:%d %b %Y} - {d.ts.max():%d %b %Y} ({days:.0f} days, {d.turbine.nunique()} turbines, "
          f"{days * 24 * d.turbine.nunique():,.0f} turbine-hours)")
    print(kpi.round(1).to_string())
    print(f"\nParked above {WIND_MIN:.0f} m/s: {kpi.exposed_h.sum():.1f} h; in the risk zone: {kpi.risk_h.sum():.1f} h")
    print("\nWhy turbines were parked in strong wind (10-min records):")
    print(ex.reason.value_counts().head(8).to_string())
    print("\nLongest risk-zone episodes:")
    print(ep.head(5).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
