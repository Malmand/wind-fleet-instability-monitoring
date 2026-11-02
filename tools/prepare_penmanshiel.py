"""Condense the open Penmanshiel SCADA export (Zenodo, doi:10.5281/zenodo.8253010, CC BY 4.0) into the two
small files step 7 reads.

Input is the Zenodo archive Penmanshiel_SCADA_2016_WT11-15_3107.zip (56 MB, five Senvion MM82 turbines).
Only the signals needed for the exposure analysis are kept, which brings 300 MB of CSV down to about 2 MB.

    python tools/prepare_penmanshiel.py path/to/Penmanshiel_SCADA_2016_WT11-15_3107.zip
"""
import io
import os
import re
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data")

# Greenbyte column names -> short names used in step 7
KEEP = {"# Date and time": "ts", "Wind speed (m/s)": "wind", "Wind direction (°)": "wind_dir",
        "Nacelle position (°)": "nacelle", "Vane position 1+2 (°)": "vane",
        "Rotor speed (RPM)": "rotor_rpm", "Power (kW)": "power_kw"}


def main(path):
    scada, stops = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            m = re.search(r"Penmanshiel_(\d+)_", name)
            if not m:
                continue
            turbine = int(m.group(1))
            raw = io.BytesIO(z.read(name))
            # both file types start with 9 comment lines from the Greenbyte export
            if name.startswith("Turbine_Data"):
                d = pd.read_csv(raw, skiprows=9, usecols=list(KEEP)).rename(columns=KEEP)
                d["turbine"] = turbine
                scada.append(d)
            elif name.startswith("Status"):
                s = pd.read_csv(raw, skiprows=9)
                s = s[s.Status == "Stop"]   # warnings and informational messages are not stops
                stops.append(pd.DataFrame({"turbine": turbine, "start": s["Timestamp start"],
                                           "end": s["Timestamp end"], "message": s["Message"]}))
    scada = pd.concat(scada, ignore_index=True).round(3)
    stops = pd.concat(stops, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    scada.to_csv(os.path.join(OUT, "penmanshiel_10min.csv.gz"), index=False)
    stops.to_csv(os.path.join(OUT, "penmanshiel_stops.csv.gz"), index=False)
    print(f"{len(scada):,} 10-min records and {len(stops):,} stops from {scada.turbine.nunique()} turbines")


if __name__ == "__main__":
    main(sys.argv[1])
