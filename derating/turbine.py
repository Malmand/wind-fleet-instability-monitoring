"""Reduced-order aero-servo-elastic model of a 5 MW onshore turbine, vectorised over many runs.

Degrees of freedom: rotor speed (rigid drivetrain) and the first tower fore-aft mode. Aerodynamics come
from the steady BEM tables (bem.py) evaluated with the wind relative to the moving tower top, so the
aerodynamic damping of the tower is captured. The controller is a baseline variable-speed,
pitch-regulated design in the spirit of the NREL 5 MW controller (torque law with region 2.5, gain-
scheduled PI pitch on filtered generator speed), extended with a power reference that the operator can
lower: generator torque is capped at P_ref / omega and the pitch loop then holds rated speed.

Loads: tower-base fore-aft bending moment (modal), blade-root flap moment of blade 1 (quasi-steady from
the BEM flap coefficient, with wind shear seen as the blade rotates) and low-speed shaft torque.

This is not OpenFAST: no blade or drivetrain flexibility, no dynamic inflow or dynamic stall, a
rotor-effective wind instead of a full turbulence box. It is meant to capture the controller-driven load
response to power transients, not certified load levels.
"""
import numpy as np
from pathlib import Path

T = np.load(Path(__file__).parent / "aero_tables.npz")
TSR, PITCH, CP, CT, CMF = T["tsr"], T["pitch"], T["Cp"], T["Ct"], T["Cmf"]

R, RHO, A, H_HUB, H_TWR = 63.0, 1.225, np.pi * 63.0 ** 2, 90.0, 87.6
P_RATED, ETA, N_GEAR = 5.0e6, 0.944, 97.0
W_RATED_G = 1173.7 * np.pi / 30            # rated generator speed (rad/s)
W_MIN_G = 670.0 * np.pi / 30
J_ROT = 38.76e6 + N_GEAR ** 2 * 534.1      # drivetrain inertia on the low-speed side (kg m2)
J_GEN = 534.1


def pitch_schedule(wn=0.6, zeta=0.7):
    """Gain-scheduled PI pitch gains by pole placement (Jonkman et al., 2009, sec. 7.4; ROSCO-style
    look-up table): the pitch sensitivity of aerodynamic power of this rotor is computed at the steady
    above-rated operating points and the gains are scheduled on blade pitch."""
    wr = W_RATED_G / N_GEAR
    th, sens = [], []
    for V in np.arange(11.0, 25.51, 0.5):
        lam = np.array([wr * R / V])
        target = P_RATED / ETA / (0.5 * RHO * A * V ** 3)
        b = PITCH[PITCH >= 0]
        cp = lookup(CP, np.full_like(b, lam[0]), b)
        ok = np.where(cp >= target)[0]
        if not ok.size:
            continue
        t0 = max(b[ok[-1]], 0.5)
        d = 0.5 * RHO * A * V ** 3 * (lookup(CP, lam, np.array([t0 + 0.5])) - lookup(CP, lam, np.array([t0 - 0.5])))
        th.append(t0); sens.append(d[0] / np.radians(1.0))
    th, sens = np.array(th), np.array(sens)
    o = np.argsort(th); th, sens = th[o], sens[o]
    kp = 2 * J_ROT * wr * zeta * wn / (N_GEAR * -sens)
    ki = J_ROT * wr * wn ** 2 / (N_GEAR * -sens)
    return th, kp, ki


def default_params(**kw):
    p = dict(f_twr=0.324, zeta_twr=0.01, m_twr=4.37e5, cp_scale=1.0, gain_scale=1.0, shear=0.2,
             pitch_rate=8.0)
    p.update(kw)
    return p


i_opt, j_opt = np.unravel_index(np.argmax(np.where(PITCH[None, :] == 0.0, CP, -1)), CP.shape)
TSR_OPT, CP_MAX = TSR[i_opt], CP[i_opt, j_opt]
K_OPT = 0.5 * RHO * A * R ** 3 * CP_MAX / TSR_OPT ** 3 / N_GEAR ** 3   # generator-side K (N m s2)
# region 2.5: linear torque from synchronous speed (10% slip) to rated torque at 99% rated speed
TG_RATED = P_RATED / (ETA * W_RATED_G)
W_SYNC = W_RATED_G / 1.10
SLOPE25 = TG_RATED / (0.99 * W_RATED_G - W_SYNC)
W_25 = (SLOPE25 - np.sqrt(SLOPE25 ** 2 - 4 * K_OPT * SLOPE25 * W_SYNC)) / (2 * K_OPT)


PI_GAINS = None


def lookup(tab, lam, beta):
    """Bilinear interpolation in a (tsr, pitch) table, vectorised."""
    x = np.clip((lam - TSR[0]) / (TSR[1] - TSR[0]), 0, TSR.size - 1.001)
    y = np.clip((beta - PITCH[0]) / (PITCH[1] - PITCH[0]), 0, PITCH.size - 1.001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    return (tab[i, j] * (1 - fx) * (1 - fy) + tab[i + 1, j] * fx * (1 - fy)
            + tab[i, j + 1] * (1 - fx) * fy + tab[i + 1, j + 1] * fx * fy)


def torque_law(wg, p_ref, beta=0.0):
    """Generator torque: optimal-mode gain, linear region 2.5, constant power above rated (as in the
    NREL 5 MW baseline: region 3 once the speed reaches 99% of rated or the blades are pitched > 1 deg),
    then capped by the operator power reference."""
    tg = np.where(wg < W_25, K_OPT * wg ** 2, np.minimum(SLOPE25 * (wg - W_SYNC), TG_RATED))
    tg = np.where((wg >= 0.99 * W_RATED_G) | (beta >= 1.0), P_RATED / (ETA * wg), tg)
    return np.clip(np.minimum(tg, p_ref / (ETA * wg)), 0, None)


def _init():
    global PI_GAINS
    PI_GAINS = pitch_schedule()


def steady_state(V, p_ref, prm):
    """Steady rotor speed and pitch for mean wind V and power reference p_ref (vectorised search)."""
    V = np.atleast_1d(V).astype(float); p_ref = np.broadcast_to(p_ref, V.shape).astype(float)
    wr_grid = np.linspace(W_MIN_G / N_GEAR, W_RATED_G / N_GEAR, 400)
    wr = np.empty_like(V); beta = np.zeros_like(V)
    cps = np.broadcast_to(np.asarray(prm["cp_scale"], float), V.shape)
    for k in range(V.size):
        # region 2: find speed where aero torque balances generator torque at zero pitch
        lam = wr_grid * R / V[k]
        qa = 0.5 * RHO * A * V[k] ** 3 * cps[k] * lookup(CP, lam, 0 * lam) / wr_grid
        qg = N_GEAR * torque_law(wr_grid * N_GEAR, p_ref[k])
        d = qa - qg
        idx = np.where(np.diff(np.sign(d)) < 0)[0]
        if d[-1] < 0 and idx.size and wr_grid[idx[-1]] * R / V[k] > 5.0:
            wr[k] = wr_grid[idx[-1]]
            continue
        # speed limited at rated: pitch to match the power
        wr[k] = W_RATED_G / N_GEAR
        lam = wr[k] * R / V[k]
        b = PITCH[PITCH >= 0]
        cp = cps[k] * lookup(CP, np.full_like(b, lam), b)
        target = min(p_ref[k], P_RATED) / ETA / (0.5 * RHO * A * V[k] ** 3)
        ok = np.where(cp >= target)[0]               # feathering branch: largest pitch reaching the target
        beta[k] = b[ok[-1]] if ok.size else 0.0
    return wr, beta


def kaimal_wind(V, ti, n, dt, rng):
    """Rotor-effective longitudinal wind: Kaimal spectrum (L = 340.2 m) with rotor averaging."""
    f = np.fft.rfftfreq(n, dt)[1:]
    L = 340.2
    sig = ti * V
    S = sig[:, None] ** 2 * 4 * L / V[:, None] / (1 + 6 * f[None, :] * L / V[:, None]) ** (5 / 3)
    S = S / (1 + (f[None, :] * 2 * R / V[:, None]) ** 2)                    # rotor averaging
    amp = np.sqrt(S * f[0] / 2) * n
    ph = rng.uniform(0, 2 * np.pi, amp.shape)
    spec = np.concatenate([np.zeros((V.size, 1)), amp * np.exp(1j * ph)], axis=1)
    u = np.fft.irfft(spec, n=n, axis=1)
    return V[:, None] + u


def simulate(V, ti, p0, p1, rate, t0, prm=None, seed=0, t_sim=600.0, t_spin=60.0, dt=0.05):
    """Simulate a batch of runs. p0, p1 in W; rate in W/s (ramp from p0 to p1 starting at t0, s).

    The ramp starts from the electrical power at t0 (or p0 if lower), as a power-reference ramp would.
    Returns dict of 1 Hz SCADA-like signals and 10 Hz load signals (spin-up removed)."""
    prm = prm or default_params()
    V = np.atleast_1d(np.asarray(V, float)); nb = V.size
    ti, p0, p1, rate, t0 = [np.broadcast_to(np.asarray(a, float), V.shape) for a in (ti, p0, p1, rate, t0)]
    rng = np.random.default_rng(seed)
    n = int(round((t_sim + t_spin) / dt))
    U = kaimal_wind(V, ti, n, dt, rng)
    t = np.arange(n) * dt - t_spin
    wr, beta = steady_state(V, p0, prm)
    wg_f = wr * N_GEAR
    k_t = prm["m_twr"] * (2 * np.pi * prm["f_twr"]) ** 2
    c_t = 2 * prm["zeta_twr"] * np.sqrt(k_t * prm["m_twr"])
    lam0 = wr * R / V
    x = 0.5 * RHO * A * V ** 2 * lookup(CT, lam0, beta) / k_t
    xd = np.zeros(nb); psi = np.zeros(nb)
    th_s, kp_s, ki_s = PI_GAINS
    integ = np.radians(beta)                                  # integral term, stored in pitch units (rad)
    a_f = np.exp(-2 * np.pi * 0.25 * dt)                      # 0.25 Hz speed filter
    shear_up = (1 + 0.7 * R / H_HUB) ** prm["shear"]; shear_dn = (1 - 0.7 * R / H_HUB) ** prm["shear"]
    s_amp = (shear_up ** 2 - shear_dn ** 2) / 2              # 1P amplitude of the flap moment (relative)
    every_scada, every_load = int(round(1 / dt)), int(round(0.1 / dt))
    keep = t >= 0
    out = {k: [] for k in ("P", "wg", "pitch", "wind", "pref")}
    ld = {k: [] for k in ("Mtb", "Mflap", "Qlss")}
    p_start = np.full(nb, np.nan)          # the ramp starts from the power being produced when the order arrives
    p_el = np.zeros(nb)
    for s in range(n):
        ts = t[s]
        arm = (ts >= t0) & np.isnan(p_start)
        p_start[arm] = np.minimum(p0, p_el)[arm]
        p_ref = np.where(ts < t0, p0, np.maximum(p1, np.nan_to_num(p_start) - rate * (ts - t0)))
        vrel = U[:, s] - xd
        wg = wr * N_GEAR
        lam = wr * R / np.maximum(vrel, 1.0)
        q = 0.5 * RHO * A * vrel ** 2
        cp = prm["cp_scale"] * lookup(CP, lam, beta)
        thrust = q * lookup(CT, lam, beta)
        q_aero = q * vrel * cp / wr
        tg = torque_law(wg, p_ref, beta)
        p_el = ETA * tg * wg
        wr_dot = (q_aero - N_GEAR * tg) / J_ROT
        # pitch PI on filtered generator speed, gain scheduled
        wg_f = a_f * wg_f + (1 - a_f) * wg
        err = wg_f - W_RATED_G
        kp = prm["gain_scale"] * np.interp(beta, th_s, kp_s)
        ki = prm["gain_scale"] * np.interp(beta, th_s, ki_s)
        integ = np.clip(integ + ki * err * dt, 0.0, np.radians(90))
        cmd = np.degrees(kp * err + integ)
        cmd = np.clip(cmd, 0.0, 90.0)
        beta = beta + np.clip(cmd - beta, -prm["pitch_rate"] * dt, prm["pitch_rate"] * dt)
        # tower
        xdd = (thrust - c_t * xd - k_t * x) / prm["m_twr"]
        xd = xd + xdd * dt; x = x + xd * dt
        wr = np.maximum(wr + wr_dot * dt, 0.1)
        psi = psi + wr * dt
        if keep[s]:
            if s % every_load == 0:
                ld["Mtb"].append(k_t * x * H_TWR / 1e3)                                       # kNm
                mf = q * R * lookup(CMF, lam, beta) * (1 + s_amp * np.cos(psi)) / 1e3   # kNm
                ld["Mflap"].append(mf)
                ld["Qlss"].append((N_GEAR * tg + J_GEN * N_GEAR ** 2 * wr_dot) / 1e3)         # kNm
            if s % every_scada == 0:
                out["P"].append(ETA * tg * wg); out["wg"].append(wg * 30 / np.pi)
                out["pitch"].append(beta.copy()); out["wind"].append(U[:, s]); out["pref"].append(p_ref)
    res = {k: np.array(v).T for k, v in out.items()}
    res.update({k: np.array(v).T for k, v in ld.items()})
    return res


_init()
