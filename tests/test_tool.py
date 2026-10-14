"""Consistency tests: reference implementations vs the integrated tool.

Run:  python -m pytest -q tests     (or: python tests/test_tool.py)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fleet_sim  # noqa: E402
import monitoring  # noqa: E402
import twin_adapter  # noqa: E402


def reference_rainflow_damage(x, m):
    """Plain three-point rainflow counting (ASTM E1049), written independently as a reference."""
    # reversals
    d = np.diff(x)
    keep = np.r_[True, (d[1:] * d[:-1]) < 0, True]
    rev = list(x[keep])
    stack, dmg = [], 0.0
    for p in rev:
        stack.append(p)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:
                dmg += 0.5 * Y ** m
                stack.pop(0)
            else:
                dmg += 1.0 * Y ** m
                last = stack.pop()
                stack.pop(); stack.pop()
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        dmg += 0.5 * abs(b - a) ** m
    return dmg


def test_rainflow_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.cumsum(rng.normal(size=500))
        tool, _ = monitoring.damage(x)
        ref = reference_rainflow_damage(x, monitoring.M_SLOPE)
        assert abs(tool - ref) / ref < 1e-9


def test_damping_estimate_on_known_instability():
    row = type("R", (), dict(platform="SG-B", state="idle", wind_mean=22.0, yaw_misalign=35.0, ti=0.1))()
    x, unstable, z = fleet_sim.hf_signal(row, seed=3)
    assert unstable
    ev = monitoring.detect(x, fleet_sim.FS)
    assert ev is not None
    assert abs(ev["zeta_est"] - z) < 0.15 * abs(z)


def test_adapter_reproduces_direct_call():
    row = dict(turbine="W01", wind_mean=22.0, yaw_misalign=35.0, state="idle", ti=0.1, platform="SG-B")
    r = type("R", (), row)()
    x, _, _ = fleet_sim.hf_signal(r, seed=7)
    direct = monitoring.analyse_record(row, x, fleet_sim.FS)
    via = json.loads(twin_adapter.run(json.dumps({"scada": row, "signal": x.tolist(), "fs": fleet_sim.FS})))
    for k, v in direct.items():
        if isinstance(v, float):
            assert abs(via[k] - v) <= 1e-9 * max(1.0, abs(v))
        else:
            assert via[k] == v
    assert set(via) == set(twin_adapter.OUTPUT_SCHEMA)


def test_quality_checks_flag_bad_records():
    row = dict(wind_mean=10.0)
    assert "flat-lined sensor" in monitoring.quality(row, np.zeros(3000), 5.0)
    assert "missing samples" in monitoring.quality(row, np.ones(1000), 5.0)
    assert "wind out of range" in monitoring.quality(dict(wind_mean=-999.0), np.random.randn(3000), 5.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
