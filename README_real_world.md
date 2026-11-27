# Real-world continuous-context benchmarks

Twenty-four test problems (seven families) in `Benchmark/Real_World.py`, with the same interface
as the synthetic problems in `Benchmark/Test_Function.py` (the interface of
[WDRBO](https://github.com/frmicheli/WDRBO) and [sbokde](https://github.com/lamda-bbo/sbokde)).
They are selected in `main.py` with `--benchmark`, run with every optimiser in
`edrbo/optimizers.py` (including `EDRBO`), and write the same result files,
`Result/<optimizer>_<benchmark>_<seed>_<running_rounds>_<dim>_<mu>_<sigma>_<beta>_results.pkl`, so
`stochastic_Y`, `cumulative_regret` etc. are produced exactly as for the synthetic problems.

## Install

```bash
pip install -r requirements_real_world.txt    # requirements.txt + floris, Box2D, pyyaml, pandas
```

FLORIS and Box2D are imported lazily, so `Real_World` imports even if they are missing; only
the classes that need them fail (`Wake_Steering*` and `Robot_Push*`). The synthetic benchmarks
need none of the extra packages. The plant, glucose, building and crop problems need only
numpy, scipy and pandas at run time. BOPTEST, WOFOST and the plant GP are shipped as fitted
numpy surrogates or tables, and the glucose simulator is a numpy port; the packages that
produced them are needed only to regenerate the data (see *Regenerating the data*).

## Run

```bash
python main.py --optimizer EDRBO --benchmark Wake_Steering_Shift --init_size 5 \
               --running_rounds 200 --repeat 10 --start_seed 100 --device cpu
python main_multiple_runs.py --suite real_world         # all optimisers × 16 problems
python main_multiple_runs.py --suite real_world_extra   # all optimisers × the 8 further variants
python main_multiple_runs.py --benchmarks Plant_Leaf Plant_Leaf_Shift --optimizers EDRBO UCB --device cpu
```

`--suite real_world_extra` runs `Building_Heating_2C*`, `Crop_Management_2C*`,
`Glucose_Bolus_2Meals*` and `Robot_Push_Success*`; `--suite real_world` runs all the others, and
`--suite all` the synthetic and real-world problems together. Without `--suite`,
`main_multiple_runs.py` runs the synthetic problems, as before.

## The problems

All decisions and contexts are in [0, 1]; the last `contexts_dim` coordinates are the context.
**Regret** is always measured under **P_eval**. The optimiser only sees contexts drawn from
**P_nominal**. These are equal for the problems without `_Shift` (the usual data-driven setting)
and differ for the `*_Shift` problems. Distances are in the normalised context coordinates.

| `--benchmark` | d_x | d_c | max_x E_eval f | W1 / W2 (P_nom, P_eval) | regret of the P_nom-optimal x under P_eval |
|---|---|---|---|---|---|
| `Wind_Commitment` | 2 | 1 | 0.11476 | – | – |
| `Wind_Commitment_Shift` | 2 | 1 | 0.09338 | 0.088 / 0.106 | 0.0104 (11 % of optimum) |
| `Wake_Steering` | 2 | 1 | 0.25840 MW | – | – |
| `Wake_Steering_Shift` | 2 | 1 | 0.26944 MW | 0.067 / 0.075 | 0.342 MW (wakes steered the wrong way) |
| `Wake_Steering_2C` | 2 | 2 | 0.29224 MW | – | – |
| `Wake_Steering_2C_Shift` | 2 | 2 | 0.29306 MW | 0.111 / 0.122 | 0.334 MW |
| `Robot_Push` | 3 | 2 | 2.71999 | – | – |
| `Robot_Push_Shift` | 3 | 2 | 2.71999 | 0.139 / 0.181 | 0.679 |
| `Robot_Push_Success` | 3 | 2 | 0.40486 | – | – |
| `Robot_Push_Success_Shift` | 3 | 2 | 0.40486 | 0.139 / 0.181 | 0.240 (59 % of optimum) |
| `Plant_Leaf` | 1 | 1 | 131.36 | – | – |
| `Plant_Leaf_Shift` | 1 | 1 | 128.80 | 0.115 / 0.123 | 25.95 (20 %) |
| `Glucose_Bolus` | 1 | 1 | −2.5288 | – | – |
| `Glucose_Bolus_Shift` | 1 | 1 | −1.6290 | 0.179 / 0.179 | 0.966 (59 %; over-dosing) |
| `Glucose_Bolus_2Meals` | 2 | 2 | −1.1740 | – | – |
| `Glucose_Bolus_2Meals_Shift` | 2 | 2 | −1.9730 | 0.214 / 0.229 | 0.815 (41 %) |
| `Building_Heating` | 2 | 1 | −1.5632 €/day | – | – |
| `Building_Heating_Shift` | 2 | 1 | −1.7028 €/day | 0.148 / 0.168 | 0.083 €/day (4.9 %) |
| `Building_Heating_2C` | 2 | 2 | −1.5636 €/day | – | – |
| `Building_Heating_2C_Shift` | 2 | 2 | −1.7247 €/day | 0.278 / 0.307 | 0.217 €/day (12.6 %) |
| `Crop_Management` | 2 | 1 | 6121.8 €/ha | – | – |
| `Crop_Management_Shift` | 2 | 1 | 6155.5 €/ha | 0.309 / 0.313 | 109.5 €/ha (1.8 %) |
| `Crop_Management_2C` | 2 | 2 | 6058.9 €/ha | – | – |
| `Crop_Management_2C_Shift` | 2 | 2 | 5222.3 €/ha | 0.295 / 0.298 | 148.3 €/ha (2.8 %) |

(`python real_world_tools/describe_benchmarks.py` recomputes this table.)

### Wind-power commitment (`Wind_Commitment*`): real data
- **Context c:** the realised hourly capacity factor of a wind farm.
  - The bundled data is 8,760 hours of 2010 weather at one German site: the 80 m wind speed
    from windpowerlib's example data.
  - It is converted with the Enercon E-82/2300 power curve, smoothed into a farm curve with the
    Staffell–Pfenninger rule σ(v) = 0.6 + 0.2 v (see `scripts/make_wind_cf_windpowerlib.py`).
- **Decision:** the day-ahead committed energy q = x1 and the reserve bought ahead,
  R = 0.5·x2 (both as fractions of installed capacity).
- **Profit:** per MW per hour, in units of the day-ahead price:
  `f = q + 0.2 (c − q)^+ − 0.2 R − 1.5 min((q − c)^+, R) − 4 ((q − c)^+ − R)^+`.
  - Surplus is sold at 0.2.
  - Shortfall is covered by activated reserve at 1.5, and the rest is penalised at 4.
  - The optimum is a pair of quantiles: F(q*) = 0.4615 and F(q* − R*) = 0.08.
- **Exact computation:** E f has a closed form through the empirical CDF, so
  `max_stochastic` is computed exactly at construction.
- **Shift:** observed in the windiest months (Nov, Dec, Mar; mean CF 0.28); evaluated in the
  calmest months (Jun, Jul; mean CF 0.19).
- **Other data:** you can use real country-level series from Open Power System Data with
  `scripts/download_opsd_wind.py` (see *Custom variants*).

### Robust wake steering (`Wake_Steering*`): FLORIS
- **Model:** FLORIS (NREL's engineering wake model, Gauss-curl-hybrid) with three NREL-5MW
  turbines in a row, 6 rotor diameters apart.
- **Decision:** the yaw offsets of the two upstream turbines in [−25°, 25°]. This is the entry
  of the yaw look-up table for the measured wind direction 270° (full wake overlap).
- **Context:** the *true* wind direction in [255°, 285°]. It differs from the measured one
  because of direction variability and vane misalignment.
  - It follows a truncated N(270° + bias, sd²).
  - The nominal model is N(270°, 3²), the default of FLORIS' `UncertainFlorisModel`.
- **2-D version (`_2C`):** adds the wind speed in [3.5, 14.5] m/s, independent of the direction
  error. It follows the real site wind rose shipped with FLORIS: speeds observed with
  directions in 257.5°–282.5°, uniform within the 1 m/s bins. The 1-D version uses 8 m/s.
- **Objective:** the farm power gain from yawing, `P(yaw, c) − P(0, c)`, in MW.
  - It has the same maximiser and regret as the farm power, because the baseline doesn't
    depend on x.
  - Pass `objective='power'` for raw power.
- **Expectations:** Gauss–Legendre quadrature on 64 nodes (1-D) or 32 × 66 nodes (2-D). The
  error relative to a much finer rule is below 5·10⁻⁴ MW.
- **Shift:** observed from N(270°, 3²); evaluated under N(268°, 4²), i.e. a −2° vane
  misalignment and more variability than assumed.
  - The optimal yaw flips sign: about (+23°, +20°) nominally versus (−21°, −18°) under P_eval.
  - In `_2C_Shift` the speed distribution also shifts, from the site-wide rose (all directions)
    to the waked sector.

### Robot pushing (`Robot_Push*`): Box2D
- **Simulator:** a headless Python 3 port of Zi Wang's 3-D push simulator (MES/EBO papers, MIT
  licence) in `Benchmark/push_sim.py`. It is deterministic, as in the original but without the
  velocity noise.
- **Decision:** the hand start (r_x, r_y) ∈ [−5, 5]² and the push duration in [1, 30].
- **Context:** the goal position in [−5, 5]², drawn from the mixture
  0.6·N((2.5, 2), 0.7²I) + 0.4·N((−2, 3), 1.2²I), clamped to the table (the existing problems
  clamp the same way).
- **Rewards:**
  - `Robot_Push`: the classic `5 − ‖e(x) − g‖`. Its optimum equals the Weiszfeld
    geometric-median bound, so `max_stochastic` is exact.
  - `Robot_Push_Success`: `exp(−‖e − g‖²/2)`.
  - With the distance reward, the Wasserstein worst case of every x is just E f − ε, so
    robustness cannot change the decision. Use the success reward when you want robustness to
    matter.
- **Cost:** the object's end position doesn't depend on the goal, so each x needs one
  simulation (about 2 ms, cached).
- **Shift:** goal-frequency shift, with mixture weights 0.3/0.7 observed and 0.6/0.4 evaluated.

### Plant leaf area (`Plant_Leaf*`): GP fitted to real lab data
- **Model:** the posterior mean of the heteroscedastic sparse GP of Tay et al. (ICML 2022). It was
  fitted to real measurements of maximum leaf area (Temasek Life Sciences Laboratory). This is
  the "Plant" benchmark of the DRBO literature.
  - The model comes from the authors' repository and was converted to numpy
    (`scripts/export_plant_gp.py`). The numpy version agrees with gpflow to within 1e-11.
- **Decision:** the nutrient-solution pH, 2.5 + 4x.
- **Context:** the NH₃ concentration, 30,000·c µM (Tay et al.'s ranges).
- **Objective:** maximum leaf area.
- **Context distribution:** NH₃ ~ truncated N(0.12, 0.06²). Below about 0.15 an acidic pH is
  best; above it, pH 5–6. E f therefore has two near-equal maxima: pH 3.5 (128.2) and pH 5.2
  (131.4).
- **Shift:** observed from N(0.08, 0.06²), where pH ≈ 3.3 is optimal; evaluated under
  N(0.20, 0.10²), where pH ≈ 5.3 is optimal.

### Insulin bolus dosing (`Glucose_Bolus*`): UVA/Padova model (simglucose)
- **Simulator:** `Benchmark/glucose_sim.py` is a vectorised numpy port of simglucose's patient
  model (all 30 virtual patients).
  - It matches simglucose to within 0.05 mg/dL; check with
    `python real_world_tools/check_glucose_port.py`, which needs `pip install simglucose`.
  - The default patient is `adult#001`.
- **`Glucose_Bolus`:**
  - **Decision:** the bolus dose, 0–25 U, given 15 min before the meal.
  - **Context:** the meal's carbohydrates, 10–150 g, unknown when dosing.
  - **Objective:** −(mean Kovatchev blood-glucose risk index) from the bolus to 6 h after the
    meal, as in simglucose. Hypoglycaemia is penalised much more than hyperglycaemia.
  - **P:** the mixture of simglucose's breakfast, lunch and dinner sizes, N(45, 10²),
    N(70, 10²) and N(80, 10²).
  - **Shift:** a dose learned on lunch-size meals, evaluated on breakfast-size meals
    (over-dosing, hence hypoglycaemia risk).
- **`Glucose_Bolus_2Meals`:**
  - **Decision:** the breakfast and lunch doses (d_x = 2).
  - **Context:** both meals' carbohydrates (d_c = 2). Breakfast is at t = 0 and lunch at
    t = 300 min, and the doses interact through the insulin still active at lunch.
  - **P:** breakfast N(45, 10²) × lunch N(70, 10²).
  - **Shift:** larger, more variable meals, N(60, 15²) × N(95, 20²).
- **Why dose only:** decisions on bolus timing, a temporary basal rate or a delayed second
  bolus were tried and dropped. In this model they are always optimal at a bound (earliest or
  front-loaded), because every virtual patient's basal glucose is at least 133 mg/dL.
- **Cost:** each `evaluate_stochastic` takes about 0.25 s (1 meal) or 0.7 s (2 meals).
- **Use:** in-silico benchmark only, not medical advice.

### Building heating schedule (`Building_Heating*`): BOPTEST surrogate
- **Test case:** BOPTEST `bestest_hydronic` (IBPSA Project 1). It is a 48 m² single-zone
  dwelling with a gas boiler and a radiator, in Brussels weather, occupied before 07:00 and after
  20:00.
- **Decision:**
  - the occupied heating setpoint, 20.5–22.5 °C;
  - the pre-heating lead time before the 20:00 occupancy, 0–4 h. Between occupied periods the
    setback is 15 °C.
- **Context:** the daily mean outdoor temperature, −6 to 14 °C. `_2C` adds the daily global
  horizontal irradiation, 0–4 kWh/m².
- **Objective:** −(energy cost + 2 €/Kh × thermal discomfort) per day.
  - Cost uses the test case's prices: gas 0.0498 €/kWh, electricity 0.2535 €/kWh.
  - Discomfort uses BOPTEST's comfort band.
- **Surrogate:** the means of two exact GPs (Matérn-5/2 with a linear mean), one for cost and
  one for discomfort. They were fitted to 2,500 simulations of the test-case FMU, run here with
  FMPy (`scripts/generate_boptest_data.py`).
  - On held-out *days*, the cost error is 0.34 €/day with one context and 0.22 €/day with two,
    against a data spread of 0.52. The discomfort error is 0.33 / 0.17 Kh against a spread of
    1.43.
  - The remaining cost error is genuine weather variation that the daily summaries don't
    capture. It acts as context misspecification, as the report warned.
- **P:** the empirical distribution of the 142 real November–March days of the test-case
  weather year.
- **Shift:** tuned on March, evaluated in January–February. The optimal pre-heating grows from
  about 1.15 h to 1.8 h.
- **Platform:** the FMU itself is Linux-only and needs `libgfortran.so.4`. You only need it to
  regenerate the surrogate.

### Potato crop management (`Crop_Management*`): WOFOST 8.1 surrogate
- **Model:** WOFOST 8.1 in PCSE (`Wofost81_NWLP_CWB_CNB`: water- and nitrogen-limited) for
  potato (Fontane), planted 20 April and harvested 20 September, on a sandy soil (EC1) with
  real 2006 Wageningen weather.
- **Decision:**
  - total mineral N, 0–250 kg/ha: 60 % at planting and 40 % 40 days later;
  - total irrigation, 0–40 cm, in 10 weekly doses from 1 June.
- **Context:** a rainfall multiplier, 0.2–1.8. `_2C` adds a temperature offset, −2 to +4 °C,
  with evapotranspiration recomputed by PCSE's Penman routine.
- **Objective:** profit in €/ha = 0.5 €/kg × tuber dry matter − 2 €/kg × N − 40 €/cm ×
  irrigation.
  - The optimum is about 115 kg N/ha and 26 cm of irrigation.
  - The N response is nearly linear above about 100 kg/ha, so with N prices below about
    1.5 €/kg the optimal N moves to the bound.
- **Surrogate:** an exact GP (Matérn-5/2 with a linear mean) fitted to 2,500 WOFOST runs
  (`scripts/generate_wofost_data.py`). The held-out error is 39 kg/ha (1-D) and 95 kg/ha (2-D),
  against a spread of 3,100–3,800 kg/ha.
- **P:**
  - Rainfall multiplier: log-normal, fitted to the real year-to-year variation of
    planting-to-harvest rainfall at Wageningen (2004–2008 and 2013–2015;
    `Benchmark/data/wageningen_season_rain.csv`).
  - Temperature offset: N(0, 0.59²), the year-to-year s.d. of the seasonal mean temperature.
- **Shift:** severe drought (rain × 0.5) for 1-D, and drier plus warmer (rain × 0.75,
  +1.5 °C) for 2-D. Irrigation buffers rainfall, so the 1-D shift costs only about 2 %.

## Conventions (same as `Continuous_Vendor`)
- **Sign:** `main.py` passes `negate = not --minimization`, i.e. `negate=True` by default. With
  `negate=True` the class returns the natural objective (profit, MW gain, push reward), to be
  maximised, and `max_stochastic = max_x E_eval f`. With `--minimization` it returns −f and
  `max_stochastic = −min_x E_eval f`.
- **Return values:** `evaluate_true(X)` returns `(f, contexts)` with shapes `(n,)` and
  `(n, d_c)`. It samples contexts with the torch RNG, so `main.py`'s seeding makes runs
  reproducible.
- **`mu` / `sigma`:** short strings that only label the result files (e.g. `N270-3toN268-4`).
- **Precomputed optima:** `max_stochastic` for the default configurations is stored in
  `PRECOMPUTED` in `Real_World.py`.
  - An instance with non-default arguments computes it at construction: about 15 s for 1-D
    FLORIS and robot pushing, about 2 min for 2-D FLORIS.
  - `python real_world_tools/compute_max_stochastic.py --problem <Name> --write` stores new values.
- **Extra methods:** `f_xc(X, C)`, `context_distribution(which)` and `expected_value(X, which)`
  expose the objective and the discrete distributions (`which='nominal'|'eval'`) for your own
  analysis.
- **Observation noise:** as in the original classes, `noise_std` is not supported, because
  botorch would try to add noise to the `(f, contexts)` tuple. The only randomness of an
  observation is the context.

## Robust (worst-case) evaluation

`real_world_tools/robust_eval.py` reads the decisions `X` from the `Result/*_results.pkl` files
written by `main.py`. For every iterate it computes:
- the expected value under P_eval and the expected regret;
- for each radius ε, the Wasserstein worst case
  `inf_{W_p(Q, P_ctr) ≤ ε} E_Q f(x_t, ·)` and the robust regret
  `max_x WC_ε − WC_ε(x_t)`.

The worst case is exact for distributions supported on a fine context grid. It is computed
through the 1-D concave dual of the transport LP and checked against HiGHS
(`python real_world_tools/robust_eval.py --test`).

```bash
python real_world_tools/robust_eval.py --problem Wake_Steering_Shift --optimizers UCB SBOKDE WassersteinUCB EDRBO \
       --seeds 100 101 102 103 104 105 106 107 108 109 --rounds 200 --radii 0.02 0.05 0.1 --p 2 --stride 5
```

It writes `Result/robust_<problem>_<center>_p<p>.csv` and prints the mean ± s.e. of final
simple and cumulative regret.
- `--center nominal` centres the ball at the data-generating distribution instead of P_eval.
- **Runtime:** the robust optimum max_x WC_ε is searched once per problem (about 1–6 min) and
  cached in `Result/robust_opt_<problem>_<center>_p<p>.json`. After that, the worst cases cost
  about 0.1–0.5 s per iterate (most for 2-D FLORIS). `--stride` subsamples iterates.
- For `Glucose_Bolus_2Meals*` every atom–target pair is a simulation. Use `--grid 21 --stride 10`
  there.

## Custom variants

`main.py` instantiates classes by name with only `negate`, so to use other settings define a
subclass, e.g. at the end of `Real_World.py`, and add its name to `__all__` there (`main.py`
imports the classes and the `--benchmark` choices from `__all__`):

```python
class Wind_Commitment_OPSD_DE(Wind_Commitment):     # nominal 2016, regret on 2019
    def __init__(self, noise_std=None, negate=False, bounds=None):
        super().__init__(noise_std, negate, bounds,
                         data_file='Benchmark/data/opsd_DE_onshore_2016.csv',
                         eval_data_file='Benchmark/data/opsd_DE_onshore_2019.csv')

class Wake_Steering_Bias(Wake_Steering):            # +2 deg bias, sd 5 deg
    def __init__(self, noise_std=None, negate=False, bounds=None):
        super().__init__(noise_std, negate, bounds, wd_nominal=(270, 3), wd_eval=(272, 5))
```

To create the OPSD files, run
`python scripts/download_opsd_wind.py --region DE --kind onshore --years 2016 --out Benchmark/data/opsd_DE_onshore_2016.csv`
and the same with `--years 2019`. The script downloads OPSD's 60-min single-index file
(~125 MB). `--list` shows the available wind columns; `--csv` uses a file you downloaded by
hand.

## Files

```
Benchmark/Real_World.py                     benchmark classes
Benchmark/push_sim.py                       Box2D push simulator (port, MIT)
Benchmark/data/wind_cf_windpowerlib_2010.csv  hourly capacity factors (derived, see script)
Benchmark/data/floris_wind_rose.csv         site wind rose (FLORIS example data, BSD-3)
main.py, main_multiple_runs.py              run the benchmarks (--benchmark / --suite real_world)
requirements_real_world.txt                 requirements.txt + the packages of the real-world benchmarks
real_world_tools/robust_eval.py             expected / worst-case evaluation of finished runs
real_world_tools/compute_max_stochastic.py  (re)compute max_stochastic
real_world_tools/describe_benchmarks.py     dimensions, optima, shift sizes
real_world_tools/check_glucose_port.py      compares the glucose port with simglucose
scripts/make_wind_cf_windpowerlib.py        rebuilds the capacity-factor file
scripts/download_opsd_wind.py               country-level wind capacity factors from OPSD
Benchmark/glucose_sim.py                    vectorised UVA/Padova patient model (simglucose port, MIT)
Benchmark/data/plant_leaf_NH3pH_svgp.npz    plant leaf-area GP (Tay et al., MIT)
Benchmark/data/simglucose_vpatient_params.csv  virtual-patient parameters (simglucose, MIT)
Benchmark/data/boptest_hydronic_*.csv/.npz  BOPTEST simulations, daily weather contexts, fitted GPs
Benchmark/data/wofost_potato_*.csv/.npz     WOFOST simulations and fitted GP
Benchmark/data/wageningen_season_rain.csv   seasonal rainfall/temperature per year (P of the crop context)
scripts/export_plant_gp.py                  gpflow pickles -> numpy
scripts/generate_boptest_data.py            FMU simulations + GP fit (+ daily weather table)
scripts/generate_wofost_data.py             WOFOST simulations + GP fit (+ rainfall table)
```

## Regenerating the data

The shipped files were produced as follows (about 15 min each for BOPTEST and WOFOST on 2 CPUs):

```bash
# plant GP (needs gpflow, tensorflow, tensorflow-probability, tf-keras)
python scripts/export_plant_gp.py --edrbo efficient-distributionally-robust-BO --bocu unified-framework-BOCU
# BOPTEST (Linux; pip install fmpy; libgfortran.so.4, e.g. from Ubuntu's libgfortran4 package)
python scripts/generate_boptest_data.py --weather project1-boptest/testcases/bestest_hydronic/models/Resources/weather.csv --out Benchmark/data/boptest_hydronic_days.csv
python scripts/generate_boptest_data.py --fmu project1-boptest/testcases/bestest_hydronic/models/wrapped.fmu --n 2500 --out Benchmark/data/boptest_hydronic_sims.csv
python scripts/generate_boptest_data.py --fit Benchmark/data/boptest_hydronic_sims.csv
# WOFOST (pip install pcse; clone ajwdewit/pcse_notebooks and the wofost81 branch of ajwdewit/WOFOST_crop_parameters)
python scripts/generate_wofost_data.py --notebooks pcse_notebooks --rain_table Benchmark/data/wageningen_season_rain.csv
python scripts/generate_wofost_data.py --notebooks pcse_notebooks --crop WOFOST_crop_parameters --n 2500 --out Benchmark/data/wofost_potato_sims.csv
python scripts/generate_wofost_data.py --fit Benchmark/data/wofost_potato_sims.csv
# then refresh the stored optima
python real_world_tools/compute_max_stochastic.py --write
```

## Sources

- **FLORIS** (NREL), BSD-3: <https://github.com/NREL/floris>. Wind rose from
  `examples/inputs/wind_rose.csv`.
- **windpowerlib** (oemof), MIT: <https://github.com/wind-python/windpowerlib>. Weather from
  `example/weather.csv`; power curve from `tests/oedb/power_curves.csv`.
- **Staffell, I. and Pfenninger, S. (2016):** Using bias-corrected reanalysis to simulate current
  and future wind power output. *Energy* 114, 1224–1239.
- **Wang, Z. and Jegelka, S. (2017):** Max-value Entropy Search for Efficient Bayesian
  Optimization, ICML. Robot-pushing code: <https://github.com/zi-w/Max-value-Entropy-Search>
  (MIT).
- **Tay, S. S., Foo, C. S., Urano, D., Leong, R. and Low, B. K. H. (2022):** Efficient
  distributionally robust Bayesian optimization with worst-case sensitivity, ICML. Plant GP:
  <https://github.com/sebtsh/efficient-distributionally-robust-BO> (MIT).
- **simglucose** (Jinyu Xie), MIT: <https://github.com/jxx123/simglucose>. Implements the
  UVA/Padova T1D simulator (Dalla Man et al. 2007/2014).
- **BOPTEST** (IBPSA Project 1): <https://github.com/ibpsa/project1-boptest>, test case
  `bestest_hydronic`. Blum, D. et al. (2021), Building optimization testing framework (BOPTEST)
  for simulation-based benchmarking of control strategies in buildings, *J. Building
  Performance Simulation* 14(5).
- **PCSE / WOFOST** (Wageningen University & Research):
  - PCSE: <https://github.com/ajwdewit/pcse>;
  - crop parameters (EUPL): <https://github.com/ajwdewit/WOFOST_crop_parameters>;
  - weather and soil: <https://github.com/ajwdewit/pcse_notebooks> (MIT).
- **Open Power System Data (2020):** Data Package Time series, version 2020-10-06,
  <https://doi.org/10.25832/time_series/2020-10-06> (optional, downloaded by the script).
