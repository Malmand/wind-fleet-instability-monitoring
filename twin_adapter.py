"""Adapter for running the monitoring tool inside a digital-twin service.

A digital-twin platform typically calls tools with serialised inputs and expects serialised,
schema-stable outputs. This adapter wraps monitoring.analyse_record():
    input  JSON: {"scada": {...10-min statistics...}, "signal": [...], "fs": 5.0}
    output JSON: every key of OUTPUT_SCHEMA, always present (null when not applicable)
"""
import json
import math

import numpy as np

from monitoring import analyse_record

OUTPUT_SCHEMA = {"qc_ok": bool, "qc_issues": str, "spikes_removed": int, "event": bool, "damage": float,
                 "del_kNm": float, "start_s": float, "duration_s": float, "peak_kNm": float,
                 "freq_hz": float, "zeta_est": float}


def run(payload: str) -> str:
    req = json.loads(payload)
    out = analyse_record(req["scada"], np.asarray(req["signal"], float), float(req["fs"]))
    clean = {}
    for k, typ in OUTPUT_SCHEMA.items():
        v = out.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            clean[k] = None
        else:
            clean[k] = typ(v)
    return json.dumps(clean)
