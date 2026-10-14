"""Blade element momentum (BEM) rotor model and steady aerodynamic tables.

Geometry follows the published NREL 5 MW reference blade (Jonkman et al., 2009): chord and twist at
17 stations, 63 m tip radius, 1.5 m hub radius. The aerofoil polars are generic (linear lift with a
smooth stall to flat-plate behaviour, thickness-dependent drag), not the original DU/NACA tables,
so absolute performance is approximate.

The steady tables Cp, Ct and the blade-root flap coefficient Cmf over tip-speed ratio and pitch are
what the time-domain model uses at each step, a standard reduced-order approach.
"""
import numpy as np

R, R_HUB, B, RHO = 63.0, 1.5, 3, 1.225
r = np.array([2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25, 36.35, 40.45,
              44.55, 48.65, 52.75, 56.17, 58.90, 61.63])
dr = np.array([2.73, 2.73, 2.73, 4.10, 4.10, 4.10, 4.10, 4.10, 4.10, 4.10, 4.10,
               4.10, 4.10, 4.10, 2.73, 2.73, 2.73])
twist = np.radians([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795, 6.544,
                    5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748, 3.502,
                  3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
# relative thickness (%) of the aerofoil family at each station: cylinders at the root, DU 40-21, NACA 64-618
thick = np.array([100, 100, 100, 40, 35, 35, 30, 25, 25, 21, 21, 18, 18, 18, 18, 18, 18])


def polar(alpha, t):
    """Generic lift and drag coefficients. alpha in rad (array), t relative thickness (%)."""
    if t >= 100:                                  # cylinder
        return 0.0 * alpha, 0.5 + 0.0 * alpha
    a0 = np.radians(-4.2)       # zero-lift angle
    a_st = np.radians(10.5 - 0.08 * (t - 18))     # stall angle
    slope = 2 * np.pi * (0.95 - 0.004 * (t - 18))
    cl_lin = slope * (alpha - a0)
    cl_max = slope * (a_st - a0)
    # soft post-stall: lift decays from cl_max towards flat-plate behaviour (Viterna-like)
    excess = np.maximum(np.abs(alpha) - a_st, 0)
    w2 = np.exp(-excess / np.radians(8.0))
    cl_post = np.sign(alpha) * (w2 * cl_max + (1 - w2) * 1.05 * np.abs(np.sin(2 * alpha)))
    w = 1 / (1 + np.exp(-(np.abs(alpha) - a_st) / np.radians(0.8)))  # blend through stall
    cl = (1 - w) * cl_lin + w * cl_post
    cd0 = 0.006 + 0.0006 * (t - 18)
    cd = (1 - w) * (cd0 + 0.005 * (alpha - a0) ** 2 / np.radians(10) ** 2) + w * (0.02 + 1.3 * np.sin(alpha) ** 2)
    return cl, cd


def solve(tsr, pitch_deg, n_iter=200):
    """Steady BEM with Prandtl tip/hub loss and Glauert correction (Buhl). Returns Cp, Ct, Cmf."""
    lam_r = tsr * r / R
    sigma = B * chord / (2 * np.pi * r)
    pitch = np.radians(pitch_deg)
    a = np.full_like(r, 0.3)
    ap = np.zeros_like(r)
    for _ in range(n_iter):
        phi = np.arctan2(1 - a, lam_r * (1 + ap))
        alpha = phi - twist - pitch
        cl = np.empty_like(r); cd = np.empty_like(r)
        for i, t in enumerate(thick):
            cl[i], cd[i] = polar(alpha[i], t)
        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)
        sphi = np.maximum(np.abs(np.sin(phi)), 1e-4)
        F = (2 / np.pi) * np.arccos(np.exp(-B / 2 * (R - r) / (r * sphi))) * \
            (2 / np.pi) * np.arccos(np.exp(-B / 2 * (r - R_HUB) / (R_HUB * sphi)))
        F = np.maximum(F, 1e-4)
        k = sigma * cn / (4 * F * sphi ** 2)
        a_new = k / (1 + k)
        # Buhl correction for heavily loaded elements
        CT_loc = sigma * (1 - a) ** 2 * cn / sphi ** 2
        hi = CT_loc > 0.96 * F
        if hi.any():
            Fh = F[hi]; C = CT_loc[hi]
            a_new[hi] = (18 * Fh - 20 - 3 * np.sqrt(np.maximum(C * (50 - 36 * Fh) + 12 * Fh * (3 * Fh - 4), 0))) / (36 * Fh - 50)
        ap_new = 1 / (4 * F * sphi * np.cos(phi) / (sigma * ct) - 1) if np.all(ct != 0) else ap
        ap_new = np.clip(np.nan_to_num(ap_new), -0.5, 0.5)
        a = 0.7 * a + 0.3 * np.clip(a_new, -0.5, 0.95)
        ap = 0.7 * ap + 0.3 * ap_new
    W2 = ((1 - a) ** 2 + (lam_r * (1 + ap)) ** 2)            # (W / V)^2
    dFn = 0.5 * W2 * chord * cn * dr                           # per unit rho V^2, per blade
    dFt = 0.5 * W2 * chord * ct * dr
    A = np.pi * R ** 2
    Ct_ = B * dFn.sum() / (0.5 * A)
    Cq = B * (dFt * r).sum() / (0.5 * A * R)
    Cmf = (dFn * (r - R_HUB)).sum() / (0.5 * A * R)            # one blade, root flap moment
    return tsr * Cq, Ct_, Cmf


def build_tables(path):
    tsr = np.arange(1.0, 16.01, 0.25)
    pitch = np.arange(-2.0, 30.01, 0.5)
    Cp = np.zeros((tsr.size, pitch.size)); Ct = np.zeros_like(Cp); Cm = np.zeros_like(Cp)
    for i, l in enumerate(tsr):
        for j, p in enumerate(pitch):
            Cp[i, j], Ct[i, j], Cm[i, j] = solve(l, p)
    np.savez(path, tsr=tsr, pitch=pitch, Cp=Cp, Ct=Ct, Cmf=Cm)
    return tsr, pitch, Cp, Ct, Cm


if __name__ == "__main__":
    for l in [6, 7, 7.5, 8, 9]:
        print(l, [round(v, 3) for v in solve(l, 0)])
