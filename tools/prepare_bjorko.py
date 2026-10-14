"""Condense the Björkö 45 kW turbine dataset (Zenodo 10.5281/zenodo.8230330) to a small file set.

Run this on your own computer, in the folder that holds B1_CL4_20.csv:

    pip install pandas numpy rainflow
    python prepare_bjorko.py B1_CL4_20.csv

It reads the 2.8 GB file in chunks (about 30-60 min) and writes:
  bjorko_10min.csv.gz    one row per 10-min record: SCADA statistics, frequency-reserve signals,
                         transient descriptors and measured damage-equivalent loads (DEL)
  bjorko_windows.npz     20 Hz windows around the largest commanded power reductions and some
                         normal-operation windows, for time-domain checks
  bjorko_header.txt      the column names and the first rows, so the format can be checked
Together they are a few tens of MB. Upload them to the Drive folder.
"""
import sys
import csv
import numpy as np
import pandas as pd

try:
    import rainflow
except ImportError:
    sys.exit("Please run: pip install rainflow")

SRC = sys.argv[1] if len(sys.argv) > 1 else "B1_CL4_20.csv"
FS = 20                        # Hz
REC = 600                      # s per record
MIN_FILL = 0.95                # minimum fraction of samples present in a record
WANT = ["Time", "SysMode", "XTurbSpeed1", "PAB1", "PA_B1_SP", "GenTorqSP", "DCC", "DCV", "RST2",
        "TMBNS", "TMBEW", "TMBTOR", "WSN", "WS30", "WDN", "WD30", "YP", "NAX1", "NAY1", "B1EMR", "B1FMR",
        "PwrPercent", "MaxPwrEst", "WindEst", "OptRpm", "Pwaste", "Method_f", "Fgrid"]
SLOPE = {"TFA": 4, "TSS": 4, "RST2": 4, "B1EMR": 10, "B1FMR": 10}


def sniff(path):
    with open(path, "r", errors="replace") as f:
        head = [next(f) for _ in range(6)]
    delim = csv.Sniffer().sniff(head[0] + head[1], delimiters=",;\t").delimiter
    with open("bjorko_header.txt", "w") as f:
        f.write(f"delimiter: {delim!r}\n")
        f.writelines(head)
    cols = [c.strip() for c in head[0].strip().split(delim)]
    return delim, cols


def turning_points(x):
    """Keep only local extrema (fast numpy pre-step; rainflow on these gives the same cycles)."""
    d = np.diff(x)
    keep = d != 0
    x2 = np.concatenate([[x[0]], x[1:][keep]])
    d2 = np.diff(x2)
    idx = np.where(np.sign(d2[1:]) != np.sign(d2[:-1]))[0] + 1
    return np.concatenate([[x2[0]], x2[idx], [x2[-1]]])


def del_1hz(x, m):
    x = x[np.isfinite(x)]
    if x.size < 100:
        return np.nan
    d = sum(n * r ** m for r, n in rainflow.count_cycles(turning_points(x)))
    return (d / REC) ** (1 / m)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def safe_mode(x):
    x = x.dropna().round()
    return float(x.mode().iloc[0]) if len(x) else np.nan


def record_stats(t0, g):
    r = {"t_start": t0, "n": len(g)}
    if "SysMode" in g:
        sm = g["SysMode"].round()
        r["frac_running"] = float((sm == 12).mean())           # mode 12 = Running
        r["sysmode"] = safe_mode(g["SysMode"])
    if {"DCC", "DCV"} <= set(g.columns):
        g = g.assign(P=g["DCC"] * g["DCV"])                    # DC-link power (W)
    # tower fore-aft / side-side from the north-south and east-west moments and the yaw position
    if {"TMBNS", "TMBEW", "YP"} <= set(g.columns):
        yaw = np.radians(np.mod(g["YP"].values, 360))
        g = g.assign(TFA=g["TMBNS"].values * np.cos(yaw) + g["TMBEW"].values * np.sin(yaw),
                     TSS=-g["TMBNS"].values * np.sin(yaw) + g["TMBEW"].values * np.cos(yaw))
    for c in ["WSN", "WS30", "XTurbSpeed1", "PAB1", "P", "GenTorqSP", "RST2", "PwrPercent",
              "MaxPwrEst", "WindEst", "Pwaste", "Fgrid", "TFA", "TSS", "WDN"]:
        if c in g:
            v = g[c].values.astype(float)
            r[f"{c}_mean"] = np.nanmean(v); r[f"{c}_std"] = np.nanstd(v)
            r[f"{c}_min"] = np.nanmin(v); r[f"{c}_max"] = np.nanmax(v)
    if "Method_f" in g:
        r["method_f"] = safe_mode(g["Method_f"])
    for c, m in SLOPE.items():
        if c in g:
            r[f"DEL_{c}"] = del_1hz(g[c].values.astype(float), m)
    # transient descriptors from 1 Hz means, as a SCADA system would see them
    if "P" in g and "PAB1" in g:
        sec = (g["Time"].values - t0).astype(int)
        p1 = pd.Series(g["P"].values).groupby(sec).mean().values
        b1 = pd.Series(g["PAB1"].values).groupby(sec).mean().values
        if p1.size > 60:
            pmax = max(np.nanmax(p1), 1.0)
            ps = smooth(p1, 10)[5:-5] / pmax
            r["P_drop_rel"] = float(np.nanmax(np.maximum.accumulate(ps) - ps))
            r["P_ramp_max"] = float(np.nanmax(-(ps[5:] - ps[:-5]) / 5)) if ps.size > 10 else np.nan
            bs = smooth(b1, 5)[5:-5]
            r["pitch_rate_max"] = float(np.nanmax((bs[5:] - bs[:-5]) / 5)) if bs.size > 10 else np.nan
    if "PwrPercent" in g:
        pp = g["PwrPercent"].values.astype(float)
        pp = np.where(pp >= 10, pp - 10, pp)                    # 10 is added while the FFR sequence is active
        r["ffr_active_frac"] = float((g["PwrPercent"].values >= 10).mean())
        r["PwrPct_min"] = float(np.nanmin(pp)); r["PwrPct_drop"] = float(np.nanmax(pp) - np.nanmin(pp))
    return r


def to_seconds(ch):
    """Time as LabVIEW seconds (since 1904); if the file holds date strings, convert them."""
    t = pd.to_numeric(ch["Time"], errors="coerce")
    if t.notna().mean() < 0.5:
        t = (pd.to_datetime(ch["Time"], errors="coerce", utc=True) - pd.Timestamp("1904-01-01", tz="UTC")).dt.total_seconds()
    out = ch.drop(columns="Time").apply(pd.to_numeric, errors="coerce")
    out.insert(0, "Time", t)
    return out.dropna(subset=["Time"])


def main():
    delim, cols = sniff(SRC)
    use = [c for c in WANT if c in cols]
    missing = [c for c in WANT if c not in cols]
    print("columns used:", use)
    if missing:
        print("not found (fine if optional):", missing)
    if "Time" not in use:
        sys.exit("No 'Time' column found; please send bjorko_header.txt")
    rows, carry = [], None
    cand = []                      # (score, t0, kind) for window extraction in a second pass
    reader = pd.read_csv(SRC, sep=delim, usecols=use, chunksize=2_000_000, low_memory=False)
    for k, ch in enumerate(reader):
        ch = to_seconds(ch)
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        rec = np.floor(ch["Time"].values / REC) * REC
        last = rec.max()
        carry = ch[rec == last]
        ch = ch[rec < last]; rec = rec[rec < last]
        for t0, g in ch.groupby(rec):
            if len(g) < MIN_FILL * REC * FS:
                continue
            try:
                r = record_stats(t0, g)
            except Exception as e:                      # keep going; report the first few problems
                if len(rows) < 5:
                    print(f"record {t0}: {e!r}")
                continue
            rows.append(r)
            if r.get("PwrPct_drop", 0) > 0.15 or r.get("Pwaste_max", 0) > 2000:
                cand.append((r.get("PwrPct_drop", 0) + r.get("Pwaste_max", 0) / 45e3, t0, "reduction"))
        print(f"chunk {k}: {len(rows)} records so far", flush=True)
    if carry is not None and len(carry) >= MIN_FILL * REC * FS:          # last record of the file
        rows.append(record_stats(np.floor(carry["Time"].iloc[0] / REC) * REC, carry))
    df = pd.DataFrame(rows)
    # LabVIEW time: seconds since 1904-01-01 UTC
    df.insert(1, "time_utc", pd.to_datetime(df.t_start - 2082844800, unit="s", utc=True))
    df.to_csv("bjorko_10min.csv.gz", index=False, float_format="%.6g")
    print(f"wrote bjorko_10min.csv.gz with {len(df)} records")

    # second pass: 20 Hz windows of the 30 strongest reductions and 30 random running records
    cand = sorted(cand, reverse=True)[:30]
    run = df[(df.get("frac_running", 1) > 0.99)]
    rng = np.random.default_rng(0)
    normal = run.t_start.values[rng.choice(len(run), min(30, len(run)), replace=False)] if len(run) else []
    targets = {t0: "reduction" for _, t0, _ in cand}
    targets.update({t0: "normal" for t0 in normal if t0 not in targets})
    keep = ["Time", "SysMode", "XTurbSpeed1", "PAB1", "GenTorqSP", "DCC", "DCV", "RST2", "TMBNS", "TMBEW",
            "YP", "WSN", "PwrPercent", "Pwaste", "MaxPwrEst", "Fgrid"]
    keep = [c for c in keep if c in use]
    out = {t0: [] for t0 in targets}
    for ch in pd.read_csv(SRC, sep=delim, usecols=keep, chunksize=2_000_000, low_memory=False):
        ch = to_seconds(ch)
        rec = np.floor(ch["Time"].values / REC) * REC
        for t0 in np.intersect1d(np.unique(rec), list(targets)):
            out[t0].append(ch[rec == t0])
    arrays = {"columns": np.array(keep)}
    for i, (t0, parts) in enumerate(out.items()):
        if parts:
            arrays[f"{targets[t0]}_{int(t0)}"] = pd.concat(parts).sort_values("Time").values.astype(np.float32)
    np.savez_compressed("bjorko_windows.npz", **arrays)
    print(f"wrote bjorko_windows.npz with {len(arrays) - 1} windows")


if __name__ == "__main__":
    main()
