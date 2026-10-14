"""Fleet monitoring tool: data-quality checks, edgewise instability detection, event features and
fatigue damage for one 10-min record.

Interface (the unit that would be integrated into a digital-twin environment):
    analyse_record(scada_row: dict-like, signal: np.ndarray, fs: float) -> dict
Inputs are one row of 10-min statistics and the edgewise blade-root moment (kNm); the output is a
flat dictionary of quality flags, event features and damage, ready for a database table.
"""
import numpy as np
import rainflow
from scipy.signal import butter, hilbert, sosfiltfilt

BAND = (0.7, 1.3)          # Hz, around the first edgewise mode
THRESHOLD = 800.0          # kNm, envelope amplitude for an event
MIN_DURATION = 30.0        # s above threshold
M_SLOPE = 10               # S-N slope for glass-fibre blades
N_EQ = 600                 # equivalent cycles for a 1 Hz damage-equivalent load over 10 min


def quality(row, x, fs):
    """Returns a list of data-quality issues for the record (empty = usable)."""
    issues = []
    if not np.isfinite(x).all() or len(x) < 0.9 * 600 * fs:
        issues.append("missing samples")
    if np.nanstd(x[: int(60 * fs)]) < 1e-3 or np.nanstd(x[-int(60 * fs):]) < 1e-3:
        issues.append("flat-lined sensor")
    if not (0 <= row["wind_mean"] <= 50):
        issues.append("wind out of range")
    return issues


def despike(x, k=8.0):
    """Replace isolated samples further than k robust standard deviations from a rolling median."""
    from scipy.ndimage import median_filter
    med = median_filter(x, size=9, mode="nearest")
    dev = np.abs(x - med)
    mad = np.median(dev) * 1.4826 + 1e-9
    y = x.copy()
    bad = dev > k * mad + 500
    y[bad] = med[bad]
    return y, int(bad.sum())


def envelope(x, fs):
    sos = butter(4, BAND, btype="band", fs=fs, output="sos")
    xb = sosfiltfilt(sos, x)
    env = np.abs(hilbert(xb))
    win = int(5 * fs)
    return xb, np.convolve(env, np.ones(win) / win, mode="same")


def detect(x, fs):
    """Edgewise instability event: band-passed envelope above THRESHOLD for at least MIN_DURATION."""
    xb, env = envelope(x, fs)
    above = env > THRESHOLD
    if not above.any():
        return None
    # longest continuous run above threshold
    d = np.diff(np.r_[0, above.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    k = np.argmax(ends - starts)
    s, e = starts[k], ends[k]
    if (e - s) / fs < MIN_DURATION:
        return None
    seg = xb[s:e]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / fs)
    f_dom = freqs[np.argmax(spec * ((freqs > BAND[0]) & (freqs < BAND[1])))]
    # growth rate: fit a logistic envelope (exponential growth saturating in a limit cycle) up to the peak
    peak = env.max()
    zeta = np.nan
    i_pk = int(np.argmax(env))
    t = np.arange(i_pk + 1) / fs
    if i_pk > 10 * fs:
        from scipy.optimize import curve_fit
        logistic = lambda t, L, sig, t0, c: L / (1 + np.exp(-sig * (t - t0))) + c
        try:
            p, _ = curve_fit(logistic, t, env[: i_pk + 1], p0=[peak, 0.05, t[np.argmax(env > 0.5 * peak)], 0.0],
                             bounds=([0, 1e-4, 0, -peak], [2 * peak, 2.0, t[-1] * 2, peak]), maxfev=5000)
            zeta = -p[1] / (2 * np.pi * f_dom)
        except RuntimeError:
            pass
    return dict(start_s=s / fs, duration_s=(e - s) / fs, peak_kNm=float(peak), freq_hz=float(f_dom),
                zeta_est=float(zeta))


def damage(x):
    """Miner damage sum (sum of n * range^m) and the 1 Hz damage-equivalent load over 10 min."""
    d = sum(n * r ** M_SLOPE for r, mean, n, i0, i1 in rainflow.extract_cycles(x))
    return d, (d / N_EQ) ** (1 / M_SLOPE)


def analyse_record(row, x, fs):
    issues = quality(row, x, fs)
    out = dict(qc_ok=not issues, qc_issues="; ".join(issues), spikes_removed=0)
    if issues:
        return out
    x, out["spikes_removed"] = despike(np.asarray(x, float))
    ev = detect(x, fs)
    d, del_ = damage(x)
    out.update(event=ev is not None, damage=d, del_kNm=del_)
    if ev:
        out.update(ev)
    return out
