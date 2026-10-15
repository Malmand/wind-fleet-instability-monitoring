"""
Step 2: run the monitoring tool over the fleet, store results, check detection, compute fatigue
impact and KPIs, and write the weekly summary, Power BI-ready tables and dashboard.

Usage:  python step2_monitor_fleet.py        (about 1 min)
"""
import os
import time

import numpy as np
import pandas as pd

import db
import plotstyle as ps
from fleet_sim import FS
from monitoring import THRESHOLD, analyse_record

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    t0 = time.time()
    with db.connect() as con:
        recs = db.records(con)
        rows, truth = [], []
        for r in recs.itertuples():
            x, unstable, z = db.get_signal(r)
            out = analyse_record(r._asdict(), x, FS)
            rows.append(dict(ts=r.ts, turbine=r.turbine, **out))
            truth.append(dict(ts=r.ts, turbine=r.turbine, unstable=unstable, zeta_true=z))
        res = pd.DataFrame(rows)
        res.to_sql("results", con, index=False, if_exists="replace")
        weekly = db.events_by_week(con)
        avail = db.availability(con)
    res = res.merge(pd.DataFrame(truth), on=["ts", "turbine"]).merge(recs, on=["ts", "turbine"])
    ok = res[res.qc_ok.astype(bool)].copy()
    ok["event"] = ok["event"].astype(bool)

    # --- detection check against the known truth (an event must be reachable: envelope above threshold)
    from monitoring import envelope
    reach = []
    for r in ok[ok.unstable].itertuples():
        x, _, _ = db.get_signal(r)
        reach.append(envelope(x, FS)[1].max() > THRESHOLD)
    ok = ok.assign(reachable=False)
    ok.loc[ok.unstable, "reachable"] = reach
    tp = int((ok.event & ok.reachable).sum()); fp = int((ok.event & ~ok.reachable).sum())
    fn = int((~ok.event & ok.reachable).sum())
    det = dict(true_events=int(ok.reachable.sum()), detected=tp, false_alarms=fp, missed=fn,
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
               unstable_below_threshold=int((ok.unstable & ~ok.reachable).sum()))
    ev = ok[ok.event]
    det["zeta_est_vs_true_mae"] = float(np.nanmean(np.abs(ev.zeta_est - ev.zeta_true)))

    # --- fatigue impact
    dmg_event = ev.damage.sum()
    dmg_total = ok.damage.sum()
    hours_event = ev.duration_s.sum() / 3600
    hours_total = len(ok) / 6  # six 10-min records per hour
    fat = dict(event_share_of_damage_pct=100 * dmg_event / dmg_total,
               event_share_of_time_pct=100 * hours_event / hours_total,
               damage_rate_ratio_event_vs_normal=(dmg_event / max(hours_event, 1e-9)) /
               ((dmg_total - dmg_event) / (hours_total - hours_event)))
    # per-turbine DEL and event share (platform B only has events)
    turb = ok.groupby(["farm", "turbine", "platform"]).apply(
        lambda g: pd.Series(dict(records=len(g), events=int(g.event.sum()),
                                 event_hours=g.loc[g.event, "duration_s"].sum() / 3600,
                                 max_peak_kNm=g.peak_kNm.max() if g.event.any() else 0.0,
                                 damage_share_events_pct=100 * g.loc[g.event, "damage"].sum() / g.damage.sum(),
                                 # m = 10, 600 cycles per record
                                 del_10min_kNm=(g.damage.sum() / (600 * len(g))) ** 0.1)),
        include_groups=False).reset_index()
    expected = 14 * 144                                     # 10-min records in the period
    turb = turb.merge(avail.assign(availability_pct=100 * avail.usable / expected)[["turbine", "availability_pct"]],
                      on="turbine")
    # prioritisation: an event is high priority if it reached the limit cycle (peak above 4000 kNm)
    # or if its 10-min damage exceeds that of a full week of normal operation on the same turbine
    week_normal = ok[~ok.event].groupby("turbine").damage.mean() * 7 * 144
    ev = ev.assign(priority=np.where((ev.peak_kNm > 4000) | (ev.damage > ev.turbine.map(week_normal)), "high", "monitor"))

    # --- outputs: Power BI-ready tables, weekly summary
    os.makedirs(RES, exist_ok=True)
    ev[["ts", "turbine", "farm", "platform", "wind_mean", "yaw_misalign", "state", "duration_s", "peak_kNm",
        "freq_hz", "zeta_est", "damage", "priority"]].round(4).to_csv(os.path.join(RES, "fact_events.csv"), index=False)
    turb.round(3).to_csv(os.path.join(RES, "kpi_turbine.csv"), index=False)
    weekly.to_csv(os.path.join(RES, "kpi_week.csv"), index=False)
    pd.DataFrame([det | fat]).round(4).T.rename(columns={0: "value"}).to_csv(os.path.join(RES, "summary_metrics.csv"))
    qc = res.loc[~res.qc_ok.astype(bool), "qc_issues"].value_counts()
    lines = ["# Weekly fleet summary: edgewise instability", ""]
    for wk, g in weekly.groupby("week"):
        e = ev[pd.to_datetime(ev.ts).dt.strftime("%Y-W%W") == wk]
        lines += [f"## {wk}", "",
                  f"- {int(g.events.sum())} events on {g.turbine.nunique()} turbines "
                  f"({', '.join(sorted(g.farm.unique()))}), {g.event_hours.sum():.1f} h in total.",
                  f"- Highest amplitude {g.max_peak_kNm.max():.0f} kNm; {int((e.priority == 'high').sum())} events flagged high priority.",
                  f"- Conditions: idling, mean wind {e.wind_mean.mean():.1f} m/s, |yaw misalignment| {e.yaw_misalign.abs().mean():.0f} deg on average.",
                  ""]
    lines += ["## Data quality (whole period)", "",
              f"- 10-min records missing from the database: {int(expected * len(avail) - avail.expected.sum())}"] + \
             [f"- {k}: {v} records" for k, v in qc.items()]
    with open(os.path.join(RES, "weekly_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(pd.Series(det | fat).round(3).to_string()); print(turb.round(2).to_string(index=False)); print(qc)

    # --- dashboard
    fig, ax = ps.subplots(2, 2, figsize=(7.2, 5.6))
    t = turb.sort_values("turbine")
    ax[0, 0].bar(t.turbine, t.events, color=[ps.C["orange"] if p == "SG-B" else ps.C["sky"] for p in t.platform])
    ax[0, 0].set(ylabel="Events in 14 days (-)")
    ax[0, 0].tick_params(axis="x", labelsize=6.5)
    ax[0, 0].text(0.02, 0.92, "orange: platform SG-B, blue: SG-A", transform=ax[0, 0].transAxes, fontsize=6.5)
    ps.panel(ax[0, 0], "A")
    sc = ax[0, 1].scatter(ev.wind_mean, ev.yaw_misalign.abs(), c=ev.peak_kNm, cmap="viridis", s=12)
    ax[0, 1].set(xlabel="Mean wind speed (m/s)", ylabel="|Yaw misalignment| (deg)")
    fig.colorbar(sc, ax=ax[0, 1], label="Peak amplitude (kNm)")
    ps.panel(ax[0, 1], "B")
    b = t[t.platform == "SG-B"]
    ax[1, 0].bar(b.turbine, b.damage_share_events_pct, color=ps.C["orange"])
    ax[1, 0].set(ylabel="Edgewise damage from events (%)")
    ps.panel(ax[1, 0], "C")
    ax[1, 1].plot(ev.zeta_true * 100, ev.zeta_est * 100, "o", ms=3, mfc="none", color=ps.C["blue"])
    lim = [min(ev.zeta_true.min(), np.nanmin(ev.zeta_est)) * 100 * 1.1, 0]
    ax[1, 1].plot(lim, lim, ":", color="grey", lw=0.8)
    ax[1, 1].set(xlabel="True total damping (%)", ylabel="Damping estimated from growth (%)")
    ps.panel(ax[1, 1], "D")
    ps.save(fig, os.path.join(HERE, "figures", "fig1_fleet_dashboard.png"))
    # example event: signal, band-passed component, envelope and threshold
    from monitoring import envelope
    r = ev.sort_values("damage").iloc[len(ev) // 2]
    row = ok[(ok.ts == r.ts) & (ok.turbine == r.turbine)].iloc[0]
    x, _, _ = db.get_signal(row)
    xb, env = envelope(x, FS)
    t = np.arange(len(x)) / FS
    fig, ax = ps.subplots(figsize=(7.2, 2.6))
    ax.plot(t, x / 1e3, color=ps.C["sky"], lw=0.5, label="Measured edgewise moment")
    edge = slice(int(5 * FS), len(t) - int(5 * FS))            # drop filter edge effects
    ax.plot(t[edge], env[edge] / 1e3, color=ps.C["black"], lw=1.2, label="Envelope of the 0.7\u20131.3 Hz band")
    ax.axhline(THRESHOLD / 1e3, color=ps.C["vermillion"], ls=":", lw=1, label="Detection threshold")
    ax.set(xlabel="Time in 10-min record (s)", ylabel="Moment (MNm)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title(f"{r.turbine}, idling at {r.wind_mean:.1f} m/s, yaw misalignment {r.yaw_misalign:.0f} deg; "
                 f"estimated total damping {100 * r.zeta_est:.2f}%", fontsize=7, loc="right", y=1.12)
    ps.save(fig, os.path.join(HERE, "figures", "fig2_example_event.png"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
