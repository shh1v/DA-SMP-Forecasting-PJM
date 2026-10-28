# PJM day-ahead SMP forecaster

Command-line tool that forecasts the **hourly PJM day-ahead (DA) system energy price (SMP, $/MWh)**
(`system_energy_price_da`; LMP = SMP + congestion + losses, only the SMP part is forecast).
The main model is LightGBM with piecewise-linear trees (`linear_tree=True`) predicting the SMP directly.

__RESULTS__

## 1. Install

Python 3.11. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps        # makes `python -m pjmda` available (or: $env:PYTHONPATH="src")
```

Dependencies: pandas, numpy, pyarrow, lightgbm (>= 4.0), scikit-learn, shap, matplotlib, pyyaml.

Drop the raw CSVs into `data/raw/` (paths and globs are set in `config.yaml`). Every tunable value lives
in `config.yaml`: paths, hub codes, half-life, window lengths, floors, LightGBM parameters, cut-off time,
publication lags, clip bounds, `train.start`, `train.end`.

## 2. Commands

All commands are `python -m pjmda <cmd>` (add `--config path.yaml` before the command to use another config).

| Command | What it does |
|---|---|
| `clean [--raw-dir DIR]` | Parse and clean every file matching each dataset glob -> `data/clean/*.parquet`. Logs the effective training start. |
| `features [--start D --end D]` | Build the 19 model features + helper variables -> `data/features/features.parquet`, `helpers.parquet`. |
| `tune` | 12-combination grid on a 2025 walk-forward (training to 2024-12-31) -> `models/params.json`. |
| `backtest --start D --end D [--train-start D] [--retrain-every N] [--models a,b] [--ablation] [--run-name R]` | Walk-forward backtest -> `outputs/backtest/<R>/`. Refuses days after `train.end`. |
| `train --until D` | Fit and save `lgbm_linear`, `gbm_const`, `lear` (`--until` may not exceed `train.end`) -> `models/`. |
| `forecast (--date D \| --start D --end D) [--raw-dir DIR] [--model P] [--retrain] [--explain] [--plot] [--out F]` | Forecast; writes the 2-column CSV; prints a per-day summary. |
| `evaluate --start D --end D [--model P] [--forecast F] [--run-name R] [--plots]` | Score the frozen model (or a forecast CSV) against supplied actuals -> `outputs/test/<R>/`. |
| `plot (--pred P \| --run R [--source backtest\|test]) (--day D \| --month YYYY-MM \| --all-months) [--models a,b]` | Forecast-vs-actual plots from any predictions parquet. |
| `explain --start D --end D [--model P]` | SHAP, permutation importance, ALE, error breakdown -> `outputs/explain/`. |
| `run-all` | `clean` -> `features` -> `backtest` (2025, run `wf2025`) -> `train --until 2025-12-31`. Does not forecast 2026. |

Examples:

```powershell
python -m pjmda clean
python -m pjmda tune
python -m pjmda run-all
python -m pjmda backtest --start 2025-07-01 --end 2025-12-31 --run-name wf2025h2 --ablation
python -m pjmda backtest --start 2025-01-01 --end 2025-12-31 --run-name frozen2025 --retrain-every 400
python -m pjmda train --until 2025-12-31
python -m pjmda forecast --date 2026-02-10 --explain --plot
python -m pjmda evaluate --start 2026-01-01 --end 2026-08-31 --plots
python -m pjmda plot --run wf2025 --day 2025-07-15
python -m pjmda plot --run wf2025 --month 2025-07 --models lgbm_linear,naive_week
python -m pjmda plot --run test2026 --source test --all-months
python -m pjmda explain --start 2025-07-01 --end 2025-12-31
```

### Interview workflow (after the 2026 files are received)

1. Copy the 2026 files into `data/raw/` (e.g. `da_prices_2026.csv` next to `da_prices_2023_2026.csv`; any file
   matching a dataset glob is picked up, rows are concatenated and de-duplicated, the most recently modified
   file wins on conflicts and a warning is logged).
2. `python -m pjmda clean`
3. `python -m pjmda forecast --start 2026-01-01 --end 2026-08-31 --out outputs/forecast_2026-01-01_2026-08-31.csv`
   (the submission CSV: exactly the columns `datetime_beginning_ept`, `da_price_forecast`, 5,831 rows because
   2026-03-08 has 23 hours).
4. `python -m pjmda evaluate --start 2026-01-01 --end 2026-08-31 --plots` (if the 2026 actuals are included).
   Add `--forecast outputs/forecast_2026-01-01_2026-08-31.csv` to score the submitted file instead of re-predicting.

Live forecast on the spot: `python -m pjmda forecast --date YYYY-MM-DD --explain --plot` (about 8 s from raw
files; `forecast` only re-cleans raw files whose modification time or size changed).
If a DA price input (D-1 or D-7 results) is missing, `forecast` and `evaluate` stop with an error that names
the missing dates and the file they were expected in. There is no fallback model.

## 3. Information cut-off and vintages (the most important rule)

The forecast for delivery day D may only use data available **before 10:30 EPT on D-1**
(`cutoff(D)`, configurable). Every clean table carries `available_at` (tz-aware EPT) and every feature is built
with the single function `vintage.asof_select(df, keys, cutoff)`: rows with `available_at < cutoff`, latest
`available_at` per key.

| Dataset | `available_at` | Notes |
|---|---|---|
| DA prices | `(opr_date - 1 day) 13:30` | Results of D-1 are posted on D-2, so usable; D itself is not. |
| Gas | `(report_date + 1 day) 00:00` | Usable for D only if `report_date <= D-2`; late revisions are excluded. |
| Outages | `posting_timestamp` | Posted at 00:00 on D-1. |
| Load, wind, solar forecasts | `forecast_timestamp` | One vintage per target date in the files, issued about 10:17 on D-1. |
| Generation by fuel (actuals) | `ts_begin + 1 h + 24 h` | No publication time in the file: conservative lag (`gen_by_fuel.publish_lag_hours`). |

Missing-vintage handling (all logged, flagged as helper columns, never fed to the model):

* If the latest vintage before the cut-off is older than `vintage.stale_after_hours`, `stale_<source>=1`.
* The load files hold exactly one vintage per region and target date, but some regions are absent on some dates
  (`MID ATLANTIC REGION` on 17 dates). Because `RTO COMBINED = MID ATLANTIC + WESTERN + SOUTHERN` holds in the data
  (relative residual below 0.2%), one missing member is rebuilt from the other three of the same vintage
  (`filled_load=1`, config `load.identity`).
* If a target hour still has no vintage at all (RTO load: 3 days, solar: 1 day, outages: 4 days in the files), the same
  hour of D-7 (then D-14) is reused through its own cut-off, `stale_<source>=1`. Outages reuse the latest earlier posting.
* Gaps of up to 2 hours inside one day's forecast are interpolated (`filled_<source>=1`).
* Gas hubs are forward-filled for at most 4 days (`gas_filled=1`).
* `g_comp` renormalises its weights over regions present; if every region is missing the forecast stops with an error.

I checked the selection against the raw files for several days (including DST days): the load used equals the raw
vintage issued before the cut-off, `smp_lag1/7` equal the raw prices of D-1/D-7, gas uses the flow date D-1 with
`report_date <= D-2`, and `coal_share_7` recomputed from the 168 generation hours available before the cut-off
matches the feature.

## 4. Data findings that shaped the cleaning

* **Coverage.** DA prices run 2023-01-01 to 2025-12-31 only. Load, wind, solar, outages and gas already contain
  2026 inputs (through 2026-08-31), so every 2026 forecast hour has its fundamentals; only the 2026 DA prices
  are held back. `train.start` is 2022-01-01, but no file starts before 2023-01-01: the **effective training start
  is 2023-01-08** (latest dataset start + 7 warm-up days for the D-7 price lag). History is not fabricated.
* **DST.** The DA/load/wind/solar files use wall-clock hour-ending labels; spring-forward days have 23 labels and
  fall-back days have **24** labels (the repeated hour is missing; on 2023-11-05 the load file also lacks label 2). Timestamps
  are mapped with the wall clock (`ts_begin` is unique, the repeated hour resolves to its first occurrence).
  Generation by fuel is UTC based, so its 25-hour days are exact.
* **Solar hour labels.** The label is `hour_ending`; the first 88 forecast days (2023-01-01 to 2023-03-29) are
  labelled 0-23, all later days 1-24. Comparing the forecast with actual solar generation shows label k aligns
  with the hour ending at k in both groups, so the default `solar.hour_convention: hour_ending` is kept. Label 0
  falls outside the day and is dropped; the missing night hour is filled by the short-gap rule. This only
  affects night-time values in early 2023.
* **Gas file.** The `description` field contains unquoted commas, so each line is split from both ends
  (`fields[0]`=flow_date, `[1]`=code, `[-1]`=report_date, `[-2]`=price, the middle is the description).
  `(null)` prices become NaN. The `21` and `04` series are different products (e.g. Algonquin 2023-01-02:
  2.965 vs 23.693); only `gas.suffix` (default `21`) is used, never mixed. The rows for flow date 2024-01-01 carry a
  `report_date` of 2024-12-31 (a year late, probably a typo): by the vintage rule they only become usable in 2025, so
  that day is forward-filled. Weekend packages (one report date, several flow dates) are handled by the same rule.
* **Generation by fuel.** 30 exact duplicate rows removed, 6 gaps of more than one hour, `Other` fuel appears
  only in 1,140 hours (treated as absent). Storage is not clipped.
* **Out-of-range gas in January 2026.** The 2026 gas files contain a winter-storm episode: composite gas
  `g_comp` peaks at 94 $/MMBtu on 2026-01-28 (training maximum 23.0 on 2025-01-21), with Transco Z6/Z5 near
  135 $/MMBtu. These days are far outside the training data. This is exactly what the linear leaves are meant
  for, and also the main extrapolation risk (see Limitations).

## 5. Features (19, one model, all always used)

HELPER variables are intermediate quantities and are never passed to the model; they are stored in
`data/features/helpers.parquet`. Only the 19 columns below reach LightGBM.

| # | Feature | Plain-English formula | Mechanism | Source |
|---|---|---|---|---|
| 1 | `g_comp` | Load-weighted regional gas price: `w_midatl*g_midatl + w_south*g_south + w_west*g_west` ($/MMBtu); weights renormalised over regions present | Gas units are usually marginal, so fuel cost sets the price level; weighting by regional load reflects where marginal units run; linear leaves let price follow gas outside the training range | [S1][S3] |
| 2 | `gas_basis_ew` | `g_midatl - g_west` | When pipelines into the East are constrained (cold snaps) eastern gas decouples from Appalachian supply and the stack steepens (Winter Storm Elliott) | [S1] |
| 3 | `nl` | `L_rto - W - S` (MW) | Wind and solar have near-zero marginal cost, so dispatchable plants only serve net load | [S2] |
| 4 | `tight` | `nl / (C_ref - out_total)` | Position on the supply curve: flat at low utilisation, convex near capacity; outages enter by reducing available capacity | [S2][S1] |
| 5 | `ren_share` | `(W + S) / L_rto` | High renewable share at equal net load means more inflexible thermal cycling and oversupply (low/negative overnight and midday prices) | [S1][S2] |
| 6 | `ramp` | `nl(h) - nl(h-1)`; HE1 uses HE24 of D-1 from the latest vintage before `cutoff(D)` | Steep ramps (post-solar evening, HE17-HE19) need fast-start, more expensive units | [S1] |
| 7 | `nl_max` | Daily maximum of `nl` | The DA market commits units for the whole day: the day's peak decides what is committed and lifts shoulder hours | [S6] |
| 8 | `s_east` | `(L_midatl + L_south) / L_rto` | East and south load must partly be served over constrained interfaces by western generation, bringing expensive local units on the margin (East Coast and Dominion data-centre load) | [S1] |
| 9 | `coal_share_7` | `sum(coal) / sum(coal + gas)` over the last 168 hours of generation actuals available before the cut-off | A larger coal share means coal units set the price more often (changes how strongly price follows gas); tracks coal retirements the gas price cannot see | [S1][S5] |
| 10 | `hour` (cat.) | Hour ending 1-24 | Daily demand cycle and solar shape | [S1] |
| 11 | `dow` (cat.) | Day of week 0-6 | Commercial/industrial load and weekend unit commitment; Monday differs from mid-week | [S6] |
| 12 | `is_holiday` (cat.) | NERC holiday (New Year, Memorial, Independence, Labor, Thanksgiving, Christmas; Sunday -> Monday) | Holiday load looks like weekend load | [S6] |
| 13 | `month` (cat.) | Month 1-12 of D | Seasonal structure not in load and gas (maintenance seasons, summer vs winter fleets); December and January are separate categories so no sin/cos is needed | [S1] |
| 14 | `ci_lag1` | Mean over the `*_RESID_AGG` zones of `abs(C_z)` of D-1, same hour, divided by `smp1_abs` | Binding constraints force out-of-merit dispatch; system-wide congestion intensity measures how much the grid distorted dispatch yesterday and such patterns persist | [S3][S1] |
| 15 | `spr_ew_lag1` | `(C_EASTERN_HUB - C_WESTERN_HUB)` of D-1 / `smp1_abs` | The classic west-to-east transfer constraint: when it binds, eastern load is served by more expensive local generation | [S1] |
| 16 | `mlf_ew_lag1` | `(M_EASTERN_HUB - M_WESTERN_HUB)` of D-1 / `smp1_abs` | Marginal losses grow with the flow (losses ~ current squared, standard physics), so the loss spread measures transfer volume even when no constraint binds (when `spr_ew` is 0): information beyond feature 15 | [S3] |
| 17 | `smp_lag1` | SMP of D-1, same hour | Prices are persistent; yesterday's price carries scarcity and bidding conditions the fundamentals do not show | [S4] |
| 18 | `smp_lag7` | SMP of D-7, same hour | Same weekday last week (the client's naive benchmark) captures the weekly pattern | [S4] |
| 19 | `dnl` | `nl(D,h) - nl_fcst(D-1,h)`, the latter built from the vintages selected at `cutoff(D-1)` | How much tomorrow's fundamentals differ from the day behind `smp_lag1`; only meaningful together with `smp_lag1` | design |

Sources: [S1] ElectronX, "PJM: Beyond Western Hub" (Apr 2026), https://docs.electronx.com/pjm-beyond-western-hub ·
[S2] Montel, "Price Fluctuation Causes in the Power Market",
https://montel.energy/resources/blog/factors-influencing-price-fluctuation-causes-in-the-power-market ·
[S3] Monitoring Analytics, 2017 State of the Market Report for PJM, Sec. 11,
https://www.monitoringanalytics.com/reports/PJM_State_of_the_Market/2017/2017-som-pjm-sec11.pdf ·
[S4] Lago, Marcjasz, De Schutter, Weron (2021), Applied Energy 293, 116983, https://arxiv.org/abs/2008.08004 ·
[S5] Monitoring Analytics, 2025 State of the Market Report for PJM (press release and Sec. 7),
https://www.monitoringanalytics.com/reports/Market_Messages/Messages/2025-som-pjm-press-release.pdf ·
[S6] general power-market practice (DA unit commitment over the whole day, weekday/holiday load patterns), standard
domain knowledge, not a citation ·
[S7] LightGBM documentation, Parameters (`linear_tree`), https://lightgbm.readthedocs.io/en/latest/Parameters.html.

Monotone constraints (split level only): +1 on `tight`, `nl`, `nl_max`, `g_comp`; -1 on `ren_share`.
`hour`, `dow`, `is_holiday`, `month` are declared categorical: with `linear_tree=True` every numerical feature in a
leaf's branch enters that leaf's regression, and a linear term in an hour or weekday index is meaningless.

### Helper variables (never model inputs)

* From load/wind/solar forecasts (hourly, day D): `L_rto, L_midatl, L_south, L_west`, `W`, `S`.
* Gas (daily): the six hub prices (`gas_<HUB>`), `g_midatl`, `g_south`, `g_west`, `gas_filled`, load shares
  `w_midatl, w_south, w_west` (daily mean regional load / daily mean RTO load).
* Outages: `out_total` (PJM RTO total outages for day D, MW).
* Generation actuals, lagged: `C_ref` = trailing 365-day maximum of `gen_total + out_total` of that hour's operating
  day (a lower bound of installed capacity, expanding maximum before 365 days exist with `cref_warmup=1`; it ignores
  net imports and only falls when the peak leaves the window).
* DA prices of D-1 (same hour): `C_x`, `M_x` (congestion and marginal-loss components), `smp1_abs = max(|SMP(D-1,h)|, 1)`.
* Metrics only: `on_peak` (HE8-HE23, Mon-Fri, not a NERC holiday), used to break down errors, not a feature.
* Flags: `stale_load/wind/solar/outages`, `filled_load/wind/solar`, `gas_filled`, `cref_warmup`.

### Dropped features (and why)

Redundant (no information beyond the kept set):

* Raw `W`, `S`, `L_rto`: fully captured by `nl`, `ren_share` and `hour`.
* `tight_plus` (squared excess tightness): a transform of `tight`; tree splits plus leaf slopes already model the convex region.
* `nl_min`, `nl_rel` (= `nl / nl_max`): `nl_rel` is a deterministic combination of two kept features; overnight oversupply is covered by `nl` and `ren_share`.
* `ramp_eve` (daily maximum evening ramp): covered by the hourly `ramp`.
* `s_dom`, `s_midatl` separately: with the western share they sum to 1; `s_east` carries the east-vs-west information.
* `f_forced`, `f_west`, `out_forced`, `out_total` as separate features: outages enter through `tight`. The forced/planned split matters for real-time surprises, not for the DA energy price once outages are posted.
* `dg1` (day-on-day gas change): `g_comp` gives the level; together with `smp_lag1` the change adds little.
* `smp_d1_mean`, `smp_d1_max`, `smp_h_7`: highly correlated with the kept lags.
* `spr_dom`, `spr_ni`, interface spreads (MISO, SOUTH, NYIS, IMO), `mlf_wh`, `mlf_eh`, `mlf_dom`, 7-day congestion means: mostly local basis (affect LMP, not SMP) or collinear with `ci_lag1` / `spr_ew_lag1` / `mlf_ew_lag1`.
* `on_peak`, `is_dst_day`: deterministic from hour, dow and holiday; kept as helper/metric flags only.

Not allowed or not available:

* Actual load, wind or solar for day D (look-ahead).
* Generation-by-fuel actuals after the cut-off minus the publication lag.
* Gas index for flow day D (published after the cut-off); late-revised gas rows.
* Real-time prices, DA-RT spreads, installed capacity, binding-constraint data: not in the provided datasets, and no other data was pulled.
* All other gas hubs, individual load zones, merchant-line nodes (NEPTUNE, LINDENVFT, HUDSONTP), mixing the `04` and `21` gas series.

## 6. Model

* **Target.** `y = SMP(D,h)` in $/MWh, predicted directly (identity target in `model.target_fn`, the only place to
  change it). Predictions are clipped to [-50, 1000] and the number of clipped hours is logged.
* **Main model `lgbm_linear`.** LightGBM with `linear_tree=True` [S7]: splits are chosen as usual, but each leaf holds
  a ridge regression (`linear_lambda`) on the numerical features of its branch instead of a constant. The forecast can
  therefore keep rising with gas and tightness beyond the training range instead of flattening at the largest training
  value. Objective `huber` (alpha 10 $/MWh): spikes are heavy-tailed and the scoring is MAE/RMSE; `l1` is not supported
  with linear trees [S7].
* **Caveats.** (i) Numerical features are standardised with mean/std fitted on the training window only, re-fitted at
  every retrain, saved with the model and applied unchanged at prediction time. (ii) NaNs stay NaN (a leaf falls back
  to its constant). (iii) Monotone constraints are enforced when choosing splits but **not** inside the leaf
  regressions, so the final response can still violate them. (iv) The first tree has constant leaves. (v) Leaf models can
  extrapolate wildly far outside the training range; clipping bounds the damage.
* **Training window.** Every fit uses only delivery days `effective start .. min(D-1, train.end)`; nothing from 2026
  is ever used to fit or tune. Sample weights `0.5 ** (age_days / 180)`. Early stopping (100 rounds) on the **last 60 days**
  of the window (time ordered, never random); then the model is refitted on the full window with the best iteration.
* **Baselines.** `naive_week` (SMP of D-7, the client's benchmark), `naive_day` (SMP of D-1), `lear` (24 per-hour
  LassoCV models with `TimeSeriesSplit` on asinh-transformed, standardised prices; inputs: same-hour prices of D-1, D-2,
  D-3, D-7, all 24 hours of D-1, `g_comp`, the group-B features and calendar dummies [S4]), `gbm_const` (the same
  LightGBM with `linear_tree=False`, isolating the effect of the linear leaves).
* **Tuning.** Only inside the training data: 2025 walk-forward with training to 2024-12-31, 12 of the 18 combinations of
  `num_leaves {7,15,31}`, `min_data_in_leaf {100,300}`, `linear_lambda {0.1,1,10}`; learning rate and half-life fixed.
  Chosen parameters are saved in `models/params.json`.
  Note: the 2025 backtests below overlap the tuning period, so they are slightly optimistic for the tuned parameters.
* **Persistence.** `models/<name>_<until>.txt` (LightGBM native, includes the leaf regressions) plus a JSON with the
  feature list, categorical features, scaler mean/std per feature, training window, config hash and git commit.

## 7. Layout

```
config.yaml              all tunable values
src/pjmda/
  cli.py                 argparse entry point (python -m pjmda <cmd>)
  config.py              load config.yaml
  clean.py               one loader/cleaner per raw dataset
  vintage.py             cut-off logic, generic as-of selection, DST-safe time helpers
  features.py            helper variables + the 19 model features (pure functions)
  model.py               scaling, LightGBM fit/predict, LEAR, naive baselines, train/forecast
  backtest.py            walk-forward loop, metrics, ablation, tuning, test evaluation
  explain.py             SHAP, permutation importance, ALE, error breakdown
  plots.py               day / month / overview plots and summary plots
data/raw  data/clean  data/features  models  outputs
```

Every stage writes parquet so stages can be re-run independently (`features` always rebuilds from the clean tables;
it takes about a second for the whole training period).

## 8. Known limitations

* **Spikes.** The Huber objective and MAE/RMSE scoring favour typical hours; scarcity spikes are systematically
  under-predicted (see the top-5% hours and cold-snap rows above and the peak hours in the month plots).
  RMSE is dominated by a handful of days (January and late-June/July 2025).
* **Leaf models can extrapolate wildly.** Linear leaves have no built-in bound; monotone constraints are only
  enforced at splits, and ridge shrinkage (`linear_lambda`) plus clipping to [-50, 1000] are the only guards.
  In the tuned model the response is gentle rather than wild (see section 9), but it has never been validated beyond
  `g_comp` = 23, and the 2026 file contains 94. The number of clipped hours is logged by every backtest, `forecast`
  and `evaluate`; more than a handful would mean the leaves extrapolate badly and must be reported.
* **`C_ref` is a proxy.** `max(generation + outages)` ignores net imports and only falls when the peak hour leaves the
  365-day window. For the first year (8,591 training hours, `cref_warmup=1`) it is an expanding maximum, which makes
  `tight` less comparable with later years. `capacity.exclude_warmup_from_training: true` drops those rows from fits.
* **2026 forecasts depend on the 2026 DA file being complete.** `smp_lag1` and `smp_lag7` (and the congestion/loss
  features of D-1) come from the supplied DA results; a missing day stops `forecast`/`evaluate`. After the first
  forecast days the lags are actuals, not forecasts: errors do not compound, but nothing is available beyond the file.
* **Assumptions to confirm with the organisers:** gas suffix `21` (not `04`), a 24-hour publication lag for
  generation by fuel actuals, solar `hour_ending`, and that the wall-clock label mapping used for DST days matches
  the data source (fall-back days have 24 labels in the files, so the repeated hour has no DA price).
* **Vintage coverage.** The files hold one forecast vintage per region and target date, so the "older vintage"
  fallback is only exercised by missing dates (RTO load 3 days, solar 1 day, outages 4 days; the single region gaps
  are rebuilt from the identity above). With several vintages per date (live use) the rule picks the latest one
  issued before the cut-off. On 2026-08-19 the RTO, MID ATLANTIC and SOUTHERN load forecasts are all missing, so that
  day uses the D-7 load shape (`stale_load=1`).
* **Tuning overlap.** Hyperparameters were chosen on the 2025 walk-forward, so the 2025 numbers are mildly optimistic
  for the tuned parameters. The 2026 test period was never used for fitting, tuning or feature selection.
* **No uncertainty bands, point forecasts only; only the SMP, not LMP.**
