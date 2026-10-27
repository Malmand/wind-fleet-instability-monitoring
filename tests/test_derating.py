"""Checks for the de-rating extension: rotor model, controller, power orders and event identification."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "derating"))
import turbine as tb  # noqa: E402
import step5_field_validation as fv  # noqa: E402


def test_rotor_and_power_curve():
    assert 0.44 < tb.CP_MAX < 0.50 and 7.0 < tb.TSR_OPT < 8.5          # NREL 5 MW: 0.48 at 7.55
    r = tb.simulate(np.array([8.0, 16.0]), 1e-4, tb.P_RATED, tb.P_RATED, 0, 1e9, t_sim=120)
    assert abs(r["P"][1].mean() / tb.P_RATED - 1) < 0.01                # rated power above rated wind
    assert abs(r["wg"][1].mean() / 1173.7 - 1) < 0.01                   # rated speed held by pitch
    assert r["P"][0].mean() < 0.5 * tb.P_RATED and r["pitch"][0].max() < 0.1


def test_power_order_is_followed():
    r = tb.simulate(np.array([14.0]), 1e-4, tb.P_RATED, 0.4 * tb.P_RATED, 0.05 * tb.P_RATED, 100, t_sim=300)
    assert abs(r["P"][0][200:].mean() / tb.P_RATED - 0.4) < 0.01
    assert r["pitch"][0][200:].mean() > r["pitch"][0][:90].mean() + 2  # pitches towards feather


def test_event_identification():
    V = np.array([9.0, 9.0, 15.0, 15.0])
    ti = 0.14 * (0.75 * V + 5.6) / V
    p1 = np.array([1.0, 0.2, 1.0, 0.4]) * tb.P_RATED
    r = tb.simulate(V, ti, tb.P_RATED, p1, 0.05 * tb.P_RATED, 200, seed=3)
    flag, _ = fv.identify_events(r["P"], r["pitch"], r["wg"])
    assert list(flag) == [False, True, False, True]


def test_minimum_rotor_speed():
    r = tb.simulate(np.array([4.0, 5.0]), 0.15, tb.P_RATED, tb.P_RATED, 0, 1e9, t_sim=300, seed=2)
    assert (r["wg"].min(1) > 0.97 * 670.0).all()                         # region 1.5 holds the speed above ~6.9 rpm
    wr, _ = tb.steady_state(np.array([4.0]), tb.P_RATED, tb.default_params())
    assert wr[0] * 30 / np.pi < 0.95 * tb.W_RATED_G / tb.N_GEAR * 30 / np.pi  # light-wind start is not at rated speed


def test_common_wind():
    r = tb.simulate(np.array([10.0, 10.0]), 0.12, tb.P_RATED, tb.P_RATED, 0, 1e9, t_sim=120, seed=4, common_wind=True)
    assert np.allclose(r["wind"][0], r["wind"][1])                       # paired runs share the same gusts


if __name__ == "__main__":
    for f in (test_rotor_and_power_curve, test_power_order_is_followed, test_event_identification,
              test_minimum_rotor_speed, test_common_wind):
        f(); print("ok", f.__name__)
