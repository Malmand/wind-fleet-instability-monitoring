# Wind Fleet Monitoring: Edgewise Instability Detection, Fatigue Impact and KPIs

A reproducible fleet-analysis workflow for one type of aeroelastic event, edgewise blade instability of idling turbines:

| Step | What it does |
|---|---|
| 1. Measurement database | 10-min SCADA statistics for 12 turbines on two platforms (14 days) in SQLite; 5 Hz blade-root edgewise moments served per record |
| 2. Data quality | Missing records, flat-lined sensors, lost samples, out-of-range wind and spikes are flagged or repaired before analysis |
| 3. Detection and features | Band-pass and envelope of the first edgewise mode; events with duration, peak amplitude, frequency, wind, yaw and state; total damping estimated from the growth of the oscillation |
| 4. Fatigue impact | Rainflow counting and Miner's rule (S-N slope 10 for glass-fibre blades); damage from events vs normal operation; prioritisation thresholds |
| 5. KPIs and reporting | Weekly summary, per-turbine KPIs and Power BI-ready tables, dashboard figure |
| 6. Integration | A JSON adapter with a fixed output schema for running the tool in a digital-twin service, and consistency tests against reference implementations |

```bash
pip install -r requirements.txt
python step1_build_database.py    # SQLite database with 23,846 10-min records
python step2_monitor_fleet.py     # about 2 min
python tests/test_tool.py         # or: python -m pytest -q tests
```

## The fleet model

Fleet data at this level of detail is not public, so a simple physical model plays the fleet (`fleet_sim.py`):

- **Fleet.** Two farms of six 4 MW turbines on two platforms. Platform SG-B has a blade with lower structural damping in the first edgewise mode (1.0 Hz).
- **Normal loading.** Gravity at the rotor frequency, plus the turbulent response of the edgewise mode.
- **Instability.** When a turbine idles in high wind (storms above cut-out, or grid curtailment) with a yaw misalignment around 20–50°, the aerodynamic damping of the edgewise mode turns negative. If the total damping is negative, the oscillation grows exponentially until it saturates in a limit cycle.
- **Sensor faults.** 1.5% of records missing, 1% flat-lined, 1% with lost samples, 5% with spikes, and some wind-sensor faults.

Because the true state of each record is known, detection, damping estimates and fatigue results can all be checked.

## Results (14 days, 12 turbines)

**Detection.** Of 54 instability events large enough to reach the 800 kNm threshold, 53 were detected, with no false alarms. A further 9 records were unstable but grew too slowly to cross the threshold within 10 minutes. The damping estimated from the growth of the envelope (logistic fit) matches the true total damping within 0.02 percentage points on average.

**Fatigue.** On platform SG-B, instabilities cover only **0.17% of the time but cause 62% of the edgewise fatigue damage**. An event 10-min record accumulates damage about 970 times faster than normal operation, because the high S-N slope weighs large cycles so heavily. Per turbine the event share ranges from 35% to 92%. Events are flagged high priority when they reach the limit cycle (peak above 4,000 kNm) or when one record does more damage than a week of normal operation.

**KPIs** (`results/kpi_turbine.csv`, `results/kpi_week.csv`, `results/fact_events.csv`): events and event hours per turbine and week, peak amplitudes, damage share, damage-equivalent load and data availability (95.6–96.8% of expected records usable). The tables are flat and keyed by turbine and time, ready for Power BI. `results/weekly_summary.md` is the generated weekly overview.

![Fleet dashboard](figures/fig1_fleet_dashboard.png)

*A: events per turbine (none on platform SG-A). B: event conditions and peak amplitude. C: share of edgewise fatigue damage caused by events. D: total damping estimated from the measured growth vs the true value.*

![Example event](figures/fig2_example_event.png)

## Integration and consistency checks

`twin_adapter.run()` takes a JSON payload (10-min statistics, signal, sampling rate) and returns JSON with a fixed schema, so the tool can be called by a digital-twin service without Python objects crossing the interface. `tests/test_tool.py` checks:

- rainflow damage against an independent ASTM E1049 implementation (equal to 1e-9);
- the damping estimate on a known instability (within 15%);
- that the adapter reproduces the direct call exactly, with every schema key present;
- that the quality checks catch flat-lined, truncated and out-of-range records.

## SQL

`db.py` holds parameterised query functions: records by period, farm, state and minimum wind, weekly event counts joined with SCADA, and availability per turbine. The same queries would translate directly to KQL on a time-series store.

## Limitations

- The fleet and signals are synthetic: one edgewise mode, a simple aerodynamic-damping map, no aeroelastic solver. Real events also include vortex-induced vibrations and stall-induced vibrations in operation.
- Fatigue is evaluated on the blade-root edgewise moment only, with a single S-N slope and relative damage units, not certified load cases.
- Detection uses a fixed threshold; on real data it would be tuned per platform, and machine learning could classify event types.

## Repository layout

```
fleet_sim.py              synthetic fleet: SCADA, edgewise dynamics, instability model
db.py                     SQLite database and SQL query functions; signal access with sensor faults
monitoring.py             quality checks, despiking, detection, features, rainflow fatigue
twin_adapter.py           JSON interface for digital-twin integration
step1_build_database.py   builds the database
step2_monitor_fleet.py    fleet run, detection check, fatigue, KPIs, weekly summary, figures
tests/test_tool.py        consistency tests
```
