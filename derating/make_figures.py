"""Figures for the de-rating extension (figures/fig3-fig5)."""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotstyle import C, panel, save, subplots  # noqa: E402
import mlle  # noqa: E402
import turbine as tb  # noqa: E402

RES = Path(__file__).parent / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
P = tb.P_RATED


def fig3():
    """One turbulent wind field, three power orders: response of power, speed, pitch and tower load."""
    V, ti = 10.0, 0.14 * (0.75 * 10 + 5.6) / 10
    runs = [("No order", C["black"], "-", 0.0, 1e9), ("0.5 %/s", C["blue"], "--", 0.005, 200.0),
            ("10 %/s", C["vermillion"], "-", 0.10, 200.0)]
    res = []
    for _, _, _, rate, t0 in runs:
        res.append(tb.simulate(np.array([V]), ti, P, 0.3 * P, rate * P, t0, seed=21))
    fig, ax = subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    t1 = np.arange(res[0]["P"].shape[1]); t10 = np.arange(res[0]["Mtb"].shape[1]) / 10
    for (lab, col, ls, _, _), r in zip(runs, res):
        ax[0, 0].plot(t1, r["P"][0] / 1e6, color=col, ls=ls, lw=1, label=lab)
        ax[0, 1].plot(t1, r["wg"][0], color=col, ls=ls, lw=1)
        ax[1, 0].plot(t1, r["pitch"][0], color=col, ls=ls, lw=1)
        ax[1, 1].plot(t10, r["Mtb"][0] / 1e3, color=col, ls=ls, lw=0.8)
    ax[0, 0].set_ylabel("Electrical power (MW)"); ax[0, 1].set_ylabel("Generator speed (rpm)")
    ax[1, 0].set_ylabel("Blade pitch (deg)"); ax[1, 1].set_ylabel("Tower-base moment (MNm)")
    for a in ax[1]:
        a.set_xlabel("Time (s)")
    for a in ax.flat:
        a.set_xlim(150, 450)
    h, lb = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.09),
               title="Order to 30% of rated at 200 s, ramp rate", title_fontsize=7)
    for a, l in zip(ax.flat, "ABCD"):
        panel(a, l, x=-0.2)
    save(fig, FIG / "fig3_derating_transient.png")


def fig4():
    te = pd.read_csv(RES / "emulator_test_predictions.csv")
    sc = pd.read_csv(RES / "emulator_scores.csv")
    sw = pd.read_csv(RES / "sensitivity_sweeps.csv")
    fig, ax = subplots(2, 2, figsize=(7.2, 6.0))
    d = te[te.kind == "derate"]
    a = ax[0, 0]
    a.scatter(d.DEL_Mtb / 1e3, d.pred_A_DEL_Mtb / 1e3, s=9, marker="o", facecolors="none", edgecolors=C["orange"], lw=0.7,
              label="A: trained on normal operation")
    a.scatter(d.DEL_Mtb / 1e3, d.pred_C_DEL_Mtb / 1e3, s=7, marker="s", color=C["blue"], lw=0, label="C: with de-rating cases")
    lim = [0, d.DEL_Mtb.max() / 1e3 * 1.15]
    a.plot(lim, lim, color="k", lw=0.6); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("Simulated tower-base DEL (MNm)"); a.set_ylabel("Emulator estimate (MNm)")
    a.legend(frameon=False, loc="upper left")
    a = ax[0, 1]
    s = sc[sc.test_set == "derate"]
    models = s.model.unique(); w = 0.26
    hatches = ["", "///", ""]
    cols = [C["orange"], C["sky"], C["blue"]]
    for k, m in enumerate(models):
        v = s[s.model == m].set_index("target").loc[list(mlle.TARGETS)]
        a.bar(np.arange(3) + (k - 1) * w, v.p90_err_pct, w, color=cols[k], hatch=hatches[k], edgecolor="k", lw=0.4,
              label=m.split(":")[0])
    a.set_xticks(range(3)); a.set_xticklabels(["Tower base", "Blade-root flap", "Shaft torque"])
    a.set_ylabel("90th-percentile DEL error (%)")
    a.legend(frameon=False, title="Emulator", title_fontsize=7)
    base = sw[sw.sweep == "none"].set_index(["V", "seed"])
    for a, t, lab in ((ax[1, 0], "DEL_Mtb", "Tower-base DEL"), (ax[1, 1], "DEL_Qlss", "Shaft-torque DEL")):
        g = sw[sw.sweep == "p1"].copy()
        ref = base.loc[list(zip(g.V, g.seed)), t].values
        g["r"] = g[t].values / ref; g["rp"] = g[f"predC_{t}"].values / ref
        for V, col, mk in ((8.0, C["green"], "o"), (12.0, C["blue"], "s"), (18.0, C["vermillion"], "^")):
            h = g[g.V == V].groupby("x")
            m, sd = h.r.mean(), h.r.std()
            a.errorbar(m.index * 100, m, yerr=sd, color=col, lw=1, capsize=2, label=f"{V:.0f} m/s simulated")
            a.plot(m.index * 100, h.rp.mean(), ls="none", marker=mk, mfc="white", mec=col, ms=4, label=f"{V:.0f} m/s emulator")
        a.axhline(1, color="k", lw=0.5, ls=":")
        a.set_xlabel("Final power level (% of rated)"); a.set_ylabel(lab + ", event / no event")
    ax[1, 1].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
    for a, l in zip(ax.flat, "ABCD"):
        panel(a, l, x=-0.2)
    save(fig, FIG / "fig4_emulator_derating.png")


def fig5():
    f = pd.read_csv(RES / "field_validation.csv")
    s = pickle.load(open(RES / "field_example_series.pkl", "rb"))
    fig, ax = subplots(1, 3, figsize=(7.4, 2.6))
    first = f.index[(f.index < 60) & f.detected & f.effective][0]
    a = ax[0]
    t = np.arange(s["P"].shape[1])
    a.plot(t, s["P"][first] / 1e6, color=C["black"], lw=0.9, label="Power")
    a2 = a.twinx()
    a2.plot(t, s["pitch"][first], color=C["blue"], lw=0.9, ls="--", label="Pitch")
    a2.spines["right"].set_visible(True); a2.set_ylabel("Blade pitch (deg)", color=C["blue"])
    a.axvline(f.t_detect[first], color=C["vermillion"], lw=0.8, ls=":")
    a.text(f.t_detect[first] + 8, a.get_ylim()[1] * 0.97, "detected", color=C["vermillion"], fontsize=6, va="top")
    a.set_xlabel("Time (s)"); a.set_ylabel("Power (MW)")
    a = ax[1]
    d = f[f.detected]
    a.scatter(d.DEL_Mtb / 1e3, d.predA_DEL_Mtb / 1e3, s=12, marker="o", facecolors="none", edgecolors=C["orange"], lw=0.8, label="A")
    a.scatter(d.DEL_Mtb / 1e3, d.predC_DEL_Mtb / 1e3, s=10, marker="s", color=C["blue"], lw=0, label="C")
    lim = [0, max(d.DEL_Mtb.max(), d.predA_DEL_Mtb.max()) / 1e3 * 1.1]
    a.plot(lim, lim, color="k", lw=0.6); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("Measured tower-base DEL (MNm)"); a.set_ylabel("Emulator estimate (MNm)")
    a.legend(frameon=False, loc="upper left", title="Emulator", title_fontsize=7)
    a = ax[2]
    sc = pd.read_csv(RES / "field_scores.csv")
    sc = sc[sc.subset == "detected de-rating records"]
    w = 0.36
    for k, (m, col, hat) in enumerate((("A", C["orange"], ""), ("C", C["blue"], ""))):
        v = sc[sc.model == m].set_index("target").loc[list(mlle.TARGETS)]
        a.bar(np.arange(3) + (k - 0.5) * w, v.damage_ratio, w, color=col, edgecolor="k", lw=0.4, label=m)
    a.axhline(1, color="k", lw=0.6, ls=":")
    a.set_xticks(range(3)); a.set_xticklabels(["Tower", "Flap", "Shaft"])
    a.set_ylabel("Estimated / measured damage")
    a.legend(frameon=False, title="Emulator", title_fontsize=7)
    for a, l in zip(ax, "ABC"):
        panel(a, l, x=-0.25)
    save(fig, FIG / "fig5_field_validation.png")


if __name__ == "__main__":
    fig3(); fig4(); fig5()
