# Wind Fleet Monitoring: Edgewise Instability Detection, Fatigue Impact and KPIs

Four parts:

1. **Fleet monitoring** of one type of aeroelastic event, edgewise blade instability of idling turbines (steps 1-2 below).
2. **Load emulator for power de-rating transients** (`derating/`, steps 3-5): a reduced-order aero-servo-elastic turbine model, a simulation database, a machine-learning load emulator extended to commanded power reductions, and validation against field-like prototype data. See [Part 2](#part-2-load-emulator-for-power-de-rating-transients).
3. **Check on real measurements** (step 6): measured tower and shaft loads of a research turbine during frequency-control power shedding. See [Part 3](#part-3-check-on-real-measurements-bjorko-45-kw-turbine).
4. **Instability conditions on real SCADA** (step 7): how often parked turbines of a real UK wind farm sat in strong wind and off the wind, and why. See [Part 4](#part-4-instability-conditions-on-real-scada-penmanshiel).

## Part 1: fleet monitoring

A reproducible fleet-analysis workflow for edgewise blade instability of idling turbines:

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
python step7_penmanshiel_exposure.py  # Part 4, real SCADA (condensed files included in data/)

cd derating                       # Part 2
python step3_derating_database.py # 2,500 10-min simulations, about 1 min
python step4_train_emulator.py    # emulators A, B, C and sensitivity sweeps, about 2 min
python step5_field_validation.py  # one week of prototype data, event identification, comparison
python step6_bjorko_validation.py # real measurements (condensed file included in derating/data)
python make_figures.py
```

### The fleet model

Fleet data at this level of detail is not public, so a simple physical model plays the fleet (`fleet_sim.py`):

- **Fleet.** Two farms of six 4 MW turbines on two platforms. Platform SG-B has a blade with lower structural damping in the first edgewise mode (1.0 Hz).
- **Normal loading.** Gravity at the rotor frequency, plus the turbulent response of the edgewise mode.
- **Instability.** When a turbine idles in high wind (storms above cut-out, or grid curtailment) with a yaw misalignment around 20–50°, the aerodynamic damping of the edgewise mode turns negative. If the total damping is negative, the oscillation grows exponentially until it saturates in a limit cycle.
- **Sensor faults.** 1.5% of records missing, 1% flat-lined, 1% with lost samples, 5% with spikes, and some wind-sensor faults.

Because the true state of each record is known, detection, damping estimates and fatigue results can all be checked.

### Results (14 days, 12 turbines)

**Detection.** Of 54 instability events large enough to reach the 800 kNm threshold, 53 were detected, with no false alarms. A further 9 records were unstable but grew too slowly to cross the threshold within 10 minutes. The damping estimated from the growth of the envelope (logistic fit) matches the true total damping within about 0.01 percentage points on average (largest error 0.04 points).

**Fatigue.** Across the fleet, instabilities (time with the envelope above the threshold) cover only **0.17% of the time but cause 61% of the edgewise fatigue damage**; on platform SG-B alone, the only platform with events, they cover 0.33% of the time and cause 76% of its damage. During an instability, damage accumulates about 950 times faster per hour than in normal operation (about 690 times per 10-min record), because the high S-N slope weighs large cycles so heavily. Per turbine the event share ranges from 35% to 92%. Events are flagged high priority when they reach the limit cycle (peak above 4,000 kNm) or when one record does more damage than a week of normal operation.

**KPIs** (`results/kpi_turbine.csv`, `results/kpi_week.csv`, `results/fact_events.csv`): events and event hours per turbine and week, peak amplitudes, damage share, damage-equivalent load and data availability (95.6–96.8% of expected records usable). The tables are flat and keyed by turbine and time, ready for Power BI. `results/weekly_summary.md` is the generated weekly overview.

![Fleet dashboard](figures/fig1_fleet_dashboard.png)

*A: events per turbine (none on platform SG-A). B: event conditions and peak amplitude. C: share of edgewise fatigue damage caused by events. D: total damping estimated from the measured growth vs the true value.*

![Example event](figures/fig2_example_event.png)

### Integration and consistency checks

`twin_adapter.run()` takes a JSON payload (10-min statistics, signal, sampling rate) and returns JSON with a fixed schema, so the tool can be called by a digital-twin service without Python objects crossing the interface. `tests/test_tool.py` checks:

- rainflow damage against an independent ASTM E1049 implementation (equal to 1e-9);
- the damping estimate on a known instability (within 15%);
- that the adapter reproduces the direct call exactly, with every schema key present;
- that the quality checks catch flat-lined, truncated and out-of-range records.

### SQL

`db.py` holds parameterised query functions: records by period, farm, state and minimum wind, weekly event counts joined with SCADA, and availability per turbine. The same queries would translate directly to KQL on a time-series store.

### Limitations

- The fleet and signals are synthetic: one edgewise mode, a simple aerodynamic-damping map, no aeroelastic solver. Real events also include vortex-induced vibrations and stall-induced vibrations in operation.
- Fatigue is evaluated on the blade-root edgewise moment only, with a single S-N slope and relative damage units, not certified load cases.
- Detection uses a fixed threshold; on real data it would be tuned per platform, and machine learning could classify event types.

### Repository layout

```
fleet_sim.py              synthetic fleet: SCADA, edgewise dynamics, instability model
db.py                     SQLite database and SQL query functions; signal access with sensor faults
monitoring.py             quality checks, despiking, detection, features, rainflow fatigue
twin_adapter.py           JSON interface for digital-twin integration
step1_build_database.py   builds the database
step2_monitor_fleet.py    fleet run, detection check, fatigue, KPIs, weekly summary, figures
step7_penmanshiel_exposure.py  Part 4: instability conditions on real SCADA
tools/prepare_penmanshiel.py   condenses the Zenodo SCADA export into data/penmanshiel_*.csv.gz
tests/test_tool.py        consistency tests
```

## Part 2: load emulator for power de-rating transients

Wind turbines take part more and more in balancing and ancillary-service markets, so operators command power reductions (de-ratings) far more often than before. A load emulator trained only on normal operation does not know what such a transient does to the structure. This part extends a machine-learning load emulator so it can estimate fatigue loads during de-rating events, and checks it against measurements from a prototype.

| Step | What it does |
|---|---|
| Rotor aerodynamics | Blade element momentum (Prandtl tip and hub losses, Glauert/Buhl correction) on the NREL 5 MW blade geometry; Cp, Ct and blade-root flap tables over tip-speed ratio and pitch (`bem.py`) |
| Aero-servo-elastic model | Rotor speed and first tower fore-aft mode, aerodynamics evaluated with the wind relative to the moving tower top (so aerodynamic damping is captured), turbulent rotor-effective wind (Kaimal); baseline controller with a minimum-speed region 1.5 (670-871 rpm generator speed, as in the NREL 5 MW controller), region 2.5, constant power above rated and a gain-scheduled PI pitch loop tuned by pole placement to this rotor; an operator power reference that ramps down from the current output (`turbine.py`) |
| Simulation database | 1,000 normal-operation and 1,500 de-rating cases sampled over wind speed (4-25 m/s), turbulence (IEC NTM), shear, initial and final power and ramp rate (0.2 to 20 % of rated per second); 10-min SCADA statistics and transient descriptors as inputs, 1 Hz damage-equivalent loads (DEL) of tower base (m = 4), blade-root flap (m = 10) and shaft torque (m = 4) as targets (`database.py`, step 3) |
| Emulator | Gradient-boosted trees on log DEL (`mlle.py`). **A**: trained on normal operation only. **B**: with de-rating simulations. **C**: with de-rating simulations and transient features (power drop, steepest ramp, pitch rate, overspeed) |
| Sensitivity | Controlled sweeps of ramp rate and initial and final power at 8, 12 and 18 m/s with identical gusts in every run of a batch, so each order is compared with a no-order run in the same wind; simulated vs emulated DEL ratio (step 4) |
| Field validation | One week (1,008 records) from a stand-in prototype the emulator has never seen: softer, better-damped tower, 3% lower rotor efficiency, retuned pitch gains, an anemometer reading 4% low and strain-gauge calibration errors; grid-service orders (curtailment, aFRR, fast frequency reserve); events identified from SCADA alone; measured vs estimated DEL and fatigue damage (step 5) |

### Results

**A de-rating changes the load picture, and in different directions.** Below rated wind (8 m/s), a reduction pushes the rotor to rated speed and the pitch loop takes over, so the tower is regulated by pitch and loses some aerodynamic damping: tower-base DEL rises by up to about 26% (largest for a reduction to 30-40% of rated). At 12 and 18 m/s the lower thrust reduces tower DEL by about 15%. The shaft is the opposite: at 18 m/s the torque step itself is one large cycle, so a deep reduction (to 10% of rated) raises shaft DEL by about 78%, while at 8 m/s, where there is little torque to shed, shaft DEL falls by about 45%. Within 10-min records, the ramp rate (0.2-20 %/s) changed DELs and the peak tower moment by only a few percent: the depth of the reduction and the wind speed matter far more.

**The emulator needs to have seen transients.** On held-out de-rating simulations:

| Emulator | Tower base, p90 error | Blade flap, p90 error | Shaft, p90 error |
|---|---|---|---|
| A: normal operation only | 28% | 20% | 46% |
| B: + de-rating simulations | 11% | 16% | 11% |
| C: + transient features | 12% | 16% | 11% |

Most of the gain comes from including de-rating simulations in training; the transient features add little once they are there. In the sweeps, emulator C reproduces the simulated sensitivity to final power level and wind within a median 2-6% (blade flap 2%, shaft 3%, tower 6%).

**Field validation.** Of 49 effective de-ratings in the prototype week (orders that cut the prototype's available power by at least 5% of rated), 45 were identified from power, pitch and rotor speed alone (recall 0.92). The 4 missed events start late in a record (after 6 minutes) and would be caught in the following one on continuous data. The 2 other flagged records were also real orders, cutting 4.5-5% of rated, just below the threshold; no record without an order was flagged. On the identified event records:

| | Tower base | Blade flap | Shaft |
|---|---|---|---|
| Median DEL error, emulator A | 37% | 11% | 19% |
| Median DEL error, emulator C | 5% | 7% | 7% |
| Estimated / measured damage, A | 2.37 | 1.29 | 1.47 |
| Estimated / measured damage, C | 0.87 | 0.91 | 0.91 |

Emulator A overestimates tower damage during de-ratings by more than a factor of two, because it reads the pitched, high-speed state as a high-wind condition. Emulator C stays within about 13% on damage, with the remaining bias coming from the prototype-model differences (the stand-in for real model error).

![De-rating transient](figures/fig3_derating_transient.png)

*Same turbulent wind at 10 m/s, order to 30% of rated at 200 s. A: power; B: generator speed; C: pitch; D: tower-base fore-aft moment.*

![Emulator](figures/fig4_emulator_derating.png)

*A: emulator estimates on held-out de-rating simulations. B: 90th-percentile error by emulator. C, D: DEL ratio (event / no event) vs final power level, simulation (lines, mean and SD over 6 seeds) and emulator C (markers).*

![Field validation](figures/fig5_field_validation.png)

*A: an identified event in the prototype data. B: measured vs estimated tower DEL on event records. C: estimated over measured fatigue damage on event records.*

### Limitations and next steps

- **Not OpenFAST.** Two structural degrees of freedom, quasi-steady BEM tables (no dynamic inflow or dynamic stall), rigid blades and drivetrain, rotor-effective wind instead of a full turbulence box, generic aerofoil polars (Cp max 0.46 vs 0.48 for the NREL 5 MW). The next step would be to generate the database with OpenFAST or the in-house tool and keep the same pipeline.
- **The prototype is simulated.** It is the same model with changed parameters, so it tests robustness to model error, not to physics the model lacks. Part 3 adds a check on real measurements.
- **Blade flap with m = 10** is dominated by a few high-wind records, and wind shear is not visible in nacelle SCADA, so flap damage on the simulated test set is underestimated (ratio 0.58). Tail-weighted training or a shear estimate (met mast, lidar or blade-load asymmetry) would help.
- **Restorations** (ramping back up) and repeated orders in one record are not yet included.

### Files (`derating/`)

```
bem.py                      BEM rotor model and steady Cp, Ct, flap tables
turbine.py                  reduced-order aero-servo-elastic model and controller, vectorised
database.py                 case sampling, SCADA features, transient descriptors, DEL targets
mlle.py                     load emulator (gradient-boosted trees) and scoring
step3_derating_database.py  simulation database
step4_train_emulator.py     emulators A, B, C and sensitivity sweeps
step5_field_validation.py   prototype week, event identification, measured vs estimated loads
make_figures.py             figures 3-5
../tests/test_derating.py   power curve, order tracking and event identification checks
```

## Part 3: check on real measurements (Björkö 45 kW turbine)

Simulated "field" data can only test robustness to model error. To test the method on physics the model does not contain, I used the open high-frequency dataset of the Chalmers research turbine on Björkö, Sweden (45 kW, 30 m wooden tower, 20 Hz, CC BY 4.0; Fogelström et al., Zenodo, [doi:10.5281/zenodo.8230330](https://doi.org/10.5281/zenodo.8230330)). During the campaign the turbine ran frequency-control tests (FCR-N, FCR-D) and manual power shedding, and the controller logged the power it removed (`Pwaste`), next to calibrated tower-base moments and a shaft-torque strain gauge.

`tools/prepare_bjorko.py` condenses the 2.8 GB file into 10-min records (SCADA statistics, logged control signals, transient descriptors and 1 Hz DELs of tower fore-aft and side-side moments, rotated with the yaw angle, and shaft torque). It ran on a GitHub Actions runner (`tools/bjorko_github_workflow.yml`); the condensed file is `derating/data/bjorko_10min.csv.gz`. After filtering, 265 running records (44 hours on 34 days) remain, 39 of them with power shedding on 7 days. All model tests use leave-one-day-out cross-validation.

**Results**

- **Power shedding raised tower fore-aft fatigue.** Compared with the DEL expected from the mast wind speed and turbulence alone, shedding records had a median tower fore-aft DEL 19% higher (normal records: 0%). The simulations of Part 2 predicted the same direction below rated wind, where a reduction hands control of rotor speed to the pitch loop (about +26%). Side-side and shaft DELs did not change in the median.
- **Emulator trained on normal operation only (A) vs with shedding records and set-point features (C)**, on the shedding records:

| | Tower FA | Tower SS | Shaft torque |
|---|---|---|---|
| Median DEL error, A / C | 11% / 9% | 6% / 7% | 20% / 12% |
| 90th-percentile error, A / C | 25% / 28% | 16% / 16% | 83% / 44% |
| Estimated / measured damage, A / C | 0.34 / 0.41 | 0.98 / 0.91 | 0.71 / 1.00 |

  The clearest gain is on the shaft, where emulator A overestimated the low-load shedding records of one day by up to 130%. The tower fore-aft damage is underestimated by both, because one shedding record (8.7 m/s mast wind, constant 5 kW shed) has a DEL nearly twice as high as any other record; with 39 shedding records on 7 days, such cases cannot be learned.
- **The SCADA-only detection rule of Part 2 does not transfer.** This turbine regulates rotor speed with pitch already at moderate wind, so pitch activity below rated is common in normal operation: the rule found 27 of 39 shedding records but raised 64 false alarms. On a real platform the operator set-point is logged and should be used directly; a SCADA-only detector has to be tuned to the controller of each platform.

![Real measurements](figures/fig6_bjorko_measurements.png)

*A: tower fore-aft DEL relative to the value expected from ambient wind, normal vs power-shed records (median and interquartile range). B: shaft-torque DEL on shedding records, measured vs estimated. C: 90th-percentile DEL error on shedding records.*

**Limits.** A 45 kW turbine with a wooden tower is far from a multi-megawatt platform, the shedding tests cover 7 days, and the blade-root strain gauges in this dataset are uncalibrated, so blade loads are not used.

## Part 4: instability conditions on real SCADA (Penmanshiel)

Part 1 runs on a simulated fleet because blade-load data from a fleet is not public. The conditions that make a parked blade lose edgewise damping only need SCADA, though: rotor stopped, strong wind, rotor well off the wind. So these can be checked on real data. I used the open SCADA and status logs of five Senvion MM82 turbines (2 MW, 82 m rotor) at the Penmanshiel wind farm in Scotland (Plumley, Zenodo, [doi:10.5281/zenodo.8253010](https://doi.org/10.5281/zenodo.8253010), CC BY 4.0). `tools/prepare_penmanshiel.py` condenses the 2016 export for turbines 11-15 into `data/penmanshiel_10min.csv.gz` (seven signals) and `data/penmanshiel_stops.csv.gz` (stop log).

**Method.** The analysis starts on the first day all five turbines report at least 90% of their records (21 August 2016, after commissioning) and runs to the end of the year: 133 days, about 16,000 turbine-hours. A record counts as parked when the rotor turns below 1 rpm with no power export. In this export the wind direction is the nacelle position plus the vane reading, so the vane gives the yaw misalignment directly. The risk zone uses the limits of the Part 1 damping map: parked, wind above 14 m/s, misalignment 20-50°. Each parked high-wind record is matched with the stop active at the time; consecutive risk-zone records form episodes, and each episode reports whether the nacelle moved.

**Results**

- **Parked in strong wind:** 56 hours in total (0.35% of turbine-hours). The logged reasons were mostly high-wind cut-out, safety-chain faults and manual stops; 10 records were stops for tower oscillation.
- **In the risk zone:** 12 hours, and 9.3 of them on one turbine. WT14 was stopped by a safety-chain fault from 26 to 31 August and its nacelle did not move for the whole stop. When a front came through on 30 August, the wind veered away from the frozen nacelle and the parked rotor sat 20-38° off the wind in winds up to 20 m/s. On the other four turbines the yaw kept tracking while parked, and the risk zone added up to less than an hour each.
- **Data quality:** 1-5% of records missing per turbine. WT15 was stopped for about 21 hours above 14 m/s, most of it in the storm of 21-22 November, with no stop in its status log, which a monitoring tool should flag rather than trust.

So the exposure is not spread evenly over the fleet: it comes from faults that leave a turbine parked without yaw control. A simple SCADA rule (parked, strong wind, misaligned, nacelle not moving) finds these periods; on a platform with known standstill instabilities it would trigger a yaw intervention or a blade inspection.

![Penmanshiel exposure](figures/fig7_penmanshiel_exposure.png)

*A: yaw misalignment against wind speed for every parked 10-min record, by turbine, with the risk zone shaded. B: the longest risk episode (WT14, 30-31 August 2016): the nacelle stays at 241° while the wind direction moves; grey shading marks the parked rotor, the dotted line 14 m/s.*

**Limits.** This dataset has no blade loads, so it measures exposure, not events: nothing shows that these blades vibrated, and edgewise standstill instabilities mainly affect much longer blades than the 40 m blades of the MM82. The risk-zone limits come from the simulated damping map of Part 1, not from this turbine type. Wind speed and direction are measured on top of the nacelle, behind the stopped rotor, so they are approximate when the rotor is far off the wind. The period covers the first five months after commissioning, which may have more faults than a mature farm.

Outputs: `results/penmanshiel_exposure.csv` (per turbine: records, missing data, parked time, exposure hours, risk hours, unlogged stops) and `results/penmanshiel_risk_episodes.csv`.
