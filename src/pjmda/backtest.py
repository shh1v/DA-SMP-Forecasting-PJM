"""Walk-forward backtest, metrics, ablation, tuning and test-period evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pjmda import model as M
from pjmda.clean import effective_start, read_clean
from pjmda.config import dir_path
from pjmda.features import (
    MissingInputError, build_features, check_forecastable, feature_names, lear_inputs,
    missing_price_days,
)
from pjmda.vintage import date_range, localize

log = logging.getLogger(__name__)
DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------------- data
def prepare_data(cfg: dict, tables: dict, start, end) -> pd.DataFrame:
    return M.prepare_data(cfg, tables, start, end)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------- core loop
def walk_forward(cfg: dict, data: pd.DataFrame, days, models: list[str], retrain_every: int,
                 train_start, features: list[str] | None = None, overrides: dict | None = None,
                 on_update=None) -> pd.DataFrame:
    """Expanding-window walk-forward.

    For forecast day D the models are fitted on delivery days train_start..D-1 (their DA
    prices were posted before cutoff(D)); between retrains the last model and scaler are
    reused while the features of each day are built with that day's own cut-off.
    """
    trainable = [m for m in models if m in M.MODELS]
    by_day = {d: g for d, g in data.groupby("delivery_day")}
    bundles: dict[str, dict] = {}
    last_fit, outputs, skipped = None, [], []
    n_clip = {m: 0 for m in models}
    min_days = int(cfg["model"]["min_train_days"])
    t_fit = 0.0
    for D in days:
        rows = by_day.get(D)
        if rows is None or rows["g_comp"].isna().any():
            skipped.append(D)
            continue
        if trainable and (last_fit is None or (D - last_fit).days >= retrain_every):
            win = M.window(data, cfg, train_start, D - DAY)
            if win["delivery_day"].nunique() < min_days:
                raise ValueError(f"only {win['delivery_day'].nunique()} training days before {D.date()} "
                                 f"(model.min_train_days={min_days})")
            t0 = time.time()
            for m in trainable:
                bundles[m] = M.fit_model(m, win, cfg, D, features=features if m != "lear" else None,
                                         overrides=overrides)
            t_fit += time.time() - t0
            last_fit = D
            log.info("fitted %s on %s -> %s (%d rows) for forecasts from %s",
                     trainable, win["delivery_day"].min().date(), win["delivery_day"].max().date(),
                     len(win), D.date())
        out = pd.DataFrame({"ts_begin": rows["ts_begin"].to_numpy(), "actual": rows["y_raw"].to_numpy()})
        for m in models:
            p, nc = M.predict_model(m, bundles.get(m), rows, cfg)
            out[m] = p
            n_clip[m] += nc
        outputs.append(out)
        if on_update is not None:
            on_update(pd.concat(outputs, ignore_index=True))
    if skipped:
        log.warning("%d forecast day(s) skipped (no features or g_comp undefined): %s", len(skipped),
                    [str(d.date()) for d in skipped[:10]])
    clipped = {m: n for m, n in n_clip.items() if n}
    log.info("walk-forward done: %d days, fit time %.0fs; clipped predictions: %s", len(outputs), t_fit,
             clipped or "none")
    if any(n > 5 for n in clipped.values()):
        log.warning("MORE THAN A HANDFUL of predictions were clipped %s: the leaf models are "
                    "extrapolating badly; report this", clipped)
    if not outputs:
        raise ValueError("no forecast days could be produced")
    return pd.concat(outputs, ignore_index=True)


# ---------------------------------------------------------------------- metrics
def _err(a: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    e = f - a
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def score_predictions(pred: pd.DataFrame, data: pd.DataFrame, cfg: dict,
                      models: list[str] | None = None, baseline: str | None = None) -> pd.DataFrame:
    """MAE / RMSE / skill (1 - MAE/MAE_baseline) per model for the standard breakdowns.

    Only hours where the actual and every model have a value are scored, so all models
    are compared on the same hours.
    """
    models = models or [c for c in pred.columns if c not in ("ts_begin", "actual")]
    baseline = baseline if baseline in models else cfg["backtest"]["baseline"]
    flags = data[["ts_begin", "delivery_day", "hour", "on_peak", "g_comp"]]
    p = pred.merge(flags, on="ts_begin", how="left")
    ok = p[["actual"] + models].notna().all(axis=1)
    log.info("scoring %d hours (%d dropped: actual or a model missing)", int(ok.sum()), int((~ok).sum()))
    p = p.loc[ok]
    if p.empty:
        raise ValueError("no hours with both a forecast and an actual to score")
    q_top, q_cold = float(cfg["backtest"]["top_price_quantile"]), float(cfg["backtest"]["cold_snap_quantile"])
    slices = [("overall", "all", np.ones(len(p), dtype=bool)),
              ("peak", "on_peak", p["on_peak"].eq(1).to_numpy()),
              ("peak", "off_peak", p["on_peak"].eq(0).to_numpy()),
              ("price", "top5pct_hours", (p["actual"] >= p["actual"].quantile(q_top)).to_numpy()),
              ("regime", "cold_snap_days", (p["g_comp"] >= p["g_comp"].quantile(q_cold)).to_numpy())]
    for mth, idx in p.groupby(p["delivery_day"].dt.strftime("%Y-%m")).groups.items():
        slices.append(("month", mth, p.index.isin(idx)))
    for hr in sorted(p["hour"].unique()):
        slices.append(("hour", f"HE{int(hr):02d}", p["hour"].eq(hr).to_numpy()))
    rows = []
    for kind, label, m in slices:
        if not m.any():
            continue
        a = p.loc[m, "actual"].to_numpy()
        res = {mod: _err(a, p.loc[m, mod].to_numpy()) for mod in models}
        base_mae = res[baseline][0] if baseline in res else np.nan
        for mod in models:
            mae, rmse = res[mod]
            rows.append({"breakdown": kind, "group": label, "model": mod, "n": int(m.sum()),
                         "mae": mae, "rmse": rmse, "skill_mae": 1 - mae / base_mae if base_mae else np.nan})
    return pd.DataFrame(rows)


def log_overall(metrics: pd.DataFrame, title: str) -> None:
    o = metrics[(metrics["breakdown"] == "overall")].set_index("model")[["n", "mae", "rmse", "skill_mae"]]
    log.info("%s\n%s", title, o.round(3).to_string())


# ---------------------------------------------------------------------- backtest
def _check_range(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> None:
    lo, hi = pd.Timestamp(cfg["train"]["start"]), pd.Timestamp(cfg["train"]["end"])
    if start < lo or end > hi:
        raise ValueError(f"backtest days must lie within {lo.date()}..{hi.date()} (train.start..train.end); "
                         f"got {start.date()}..{end.date()}. 2026 is only for `forecast` and `evaluate`.")


def run_dir(cfg: dict, kind: str, run: str, create: bool = False) -> Path:
    p = dir_path(cfg, "outputs_dir") / kind / run
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_meta(rdir: Path, **meta) -> None:
    (rdir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def run_backtest(cfg: dict, start, end, models: list[str] | None = None, retrain_every: int | None = None,
                 run_name: str | None = None, train_start=None, ablation: bool = False,
                 tables: dict | None = None) -> pd.DataFrame:
    """Walk-forward backtest -> outputs/backtest/<run>/ (predictions, metrics, ablation, plots)."""
    from pjmda import plots

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _check_range(cfg, start, end)
    models = models or list(cfg["backtest"]["models"])
    retrain_every = int(retrain_every or cfg["backtest"]["retrain_every_days"])
    run_name = run_name or f"{start:%Y%m%d}_{end:%Y%m%d}"
    rdir = run_dir(cfg, "backtest", run_name, create=True)
    tables = tables or read_clean(cfg)
    eff = effective_start(cfg)
    train_start = max(eff, pd.Timestamp(train_start)) if train_start else eff
    log.info("BACKTEST %s: forecast days %s..%s, training from %s, retrain every %d day(s), models %s",
             run_name, start.date(), end.date(), train_start.date(), retrain_every, models)
    data = prepare_data(cfg, tables, train_start, end)

    path = rdir / "predictions.parquet"
    pred = walk_forward(cfg, data, date_range(start, end), models, retrain_every, train_start,
                        on_update=lambda df: _write_atomic(df, path))
    _write_atomic(pred, path)

    main = cfg["backtest"]["main_model"] if cfg["backtest"]["main_model"] in models else models[0]
    metrics = score_predictions(pred, data, cfg, models)
    metrics.to_csv(rdir / "metrics.csv", index=False)
    log_overall(metrics, f"{run_name}: overall")
    write_run_meta(rdir, kind="backtest", main_model=main, models=models, start=start, end=end,
                   train_start=train_start, retrain_every=retrain_every)
    plots.summary_plots(pred, data, rdir, cfg, main)
    plots.month_plots(pred, plots.plots_dir(cfg, run_name), cfg, main)
    if ablation:
        run_ablation(cfg, data, pred, start, end, retrain_every, train_start, rdir, main)
    return pred


# --------------------------------------------------------------------- ablation
def ablation_variants(cfg: dict) -> dict[str, list[str]]:
    """Feature sets to remove: each group, and coal_share_7 alone."""
    v = {f"no_{g}": list(members) for g, members in cfg["features"]["groups"].items()}
    v["no_coal_share_7"] = ["coal_share_7"]
    return v


def run_ablation(cfg: dict, data: pd.DataFrame, full_pred: pd.DataFrame, start, end, retrain_every: int,
                 train_start, rdir: Path, main: str = "lgbm_linear") -> pd.DataFrame:
    """Re-run the main model without each feature group and report the change in MAE / RMSE."""
    log.info("ABLATION on %s..%s (main model %s)", pd.Timestamp(start).date(), pd.Timestamp(end).date(), main)
    base = feature_names(cfg)
    runs = {"full": full_pred[["ts_begin", "actual", main]].rename(columns={main: "full"})}
    for name, drop in ablation_variants(cfg).items():
        feats = [f for f in base if f not in drop]
        p = walk_forward(cfg, data, date_range(start, end), [main], retrain_every, train_start, features=feats)
        runs[name] = p[["ts_begin", main]].rename(columns={main: name})
    wide = runs["full"]
    for name, p in runs.items():
        if name != "full":
            wide = wide.merge(p, on="ts_begin", how="left")
    ok = wide.notna().all(axis=1)
    wide = wide.loc[ok]
    rows = []
    mae0, rmse0 = _err(wide["actual"].to_numpy(), wide["full"].to_numpy())
    rows.append({"variant": "full (19 features)", "removed": "", "n_features": len(base), "n_hours": len(wide),
                 "mae": mae0, "rmse": rmse0, "d_mae": 0.0, "d_rmse": 0.0, "verdict": "reference"})
    for name, drop in ablation_variants(cfg).items():
        mae, rmse = _err(wide["actual"].to_numpy(), wide[name].to_numpy())
        verdict = "removal worsens error: keep" if (mae > mae0 or rmse > rmse0) else \
            "removal does NOT worsen error: candidate for removal (kept, not removed silently)"
        rows.append({"variant": name, "removed": ",".join(drop), "n_features": len(base) - len(drop),
                     "n_hours": len(wide), "mae": mae, "rmse": rmse, "d_mae": mae - mae0,
                     "d_rmse": rmse - rmse0, "verdict": verdict})
    out = pd.DataFrame(rows)
    out.to_csv(rdir / "ablation.csv", index=False)
    log.info("ablation (change vs full model, $/MWh):\n%s",
             out[["variant", "mae", "rmse", "d_mae", "d_rmse"]].round(3).to_string(index=False))
    return out


# ----------------------------------------------------------------------- tuning
def tune(cfg: dict, tables: dict | None = None) -> dict:
    """Grid search inside the training data: 2025 walk-forward, training up to 2024-12-31."""
    t = cfg["tune"]
    tables = tables or read_clean(cfg)
    start, end = pd.Timestamp(t["start"]), pd.Timestamp(t["end"])
    _check_range(cfg, start, end)
    eff = effective_start(cfg)
    data = prepare_data(cfg, tables, eff, end)
    results = []
    for i, combo in enumerate(t["grid"], 1):
        t0 = time.time()
        p = walk_forward(cfg, data, date_range(start, end), ["lgbm_linear"], int(t["retrain_every_days"]),
                         eff, overrides=combo)
        ok = p.dropna()
        mae, rmse = _err(ok["actual"].to_numpy(), ok["lgbm_linear"].to_numpy())
        results.append({**combo, "mae": mae, "rmse": rmse})
        log.info("tune %d/%d %s -> MAE %.3f RMSE %.3f (%.0fs)", i, len(t["grid"]), combo, mae, rmse,
                 time.time() - t0)
    res = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    best = {k: (int(v) if k in ("num_leaves", "min_data_in_leaf") else float(v))
            for k, v in res.loc[0, list(t["grid"][0])].items()}
    out = {"params": best, "mae": float(res.loc[0, "mae"]), "rmse": float(res.loc[0, "rmse"]),
           "tuned_on": f"walk-forward {start.date()}..{end.date()}, training up to {t['train_until']}, "
                       f"retrain every {t['retrain_every_days']} days",
           "grid_results": res.to_dict(orient="records")}
    mdir = dir_path(cfg, "models_dir", create=True)
    (mdir / "params.json").write_text(json.dumps(out, indent=2))
    log.info("tuning done, best %s (MAE %.3f); saved models/params.json", best, out["mae"])
    return out


# ------------------------------------------------------------- test evaluation
def read_forecast_csv(path: str | Path, cfg: dict) -> pd.Series:
    """Forecast CSV -> Series of da_price_forecast indexed by tz-aware ts_begin."""
    f = pd.read_csv(path)
    ts = localize(pd.to_datetime(f["datetime_beginning_ept"]), cfg["timezone"], nonexistent="NaT")
    return pd.Series(f["da_price_forecast"].to_numpy(), index=pd.DatetimeIndex(ts)).dropna()


def evaluate(cfg: dict, start, end, model_path: str | Path | None = None, forecast_file: str | Path | None = None,
             run_name: str | None = None, plots_flag: bool = False, tables: dict | None = None) -> pd.DataFrame:
    """Score the FROZEN model (no refit, no tuning) or a forecast CSV on a range of the test period."""
    from pjmda import plots

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tables = tables or read_clean(cfg)
    days = date_range(start, end)
    run_name = run_name or f"{start:%Y%m%d}_{end:%Y%m%d}"
    rdir = run_dir(cfg, "test", run_name, create=True)

    missing = missing_price_days(days, tables["da_smp"], cfg)
    if missing:
        need = sorted({d for v in missing.values() for d in v})
        raise MissingInputError(
            f"DA price inputs (D-1 / D-7 results) are missing or incomplete for {len(need)} day(s): "
            f"{[str(d.date()) for d in need[:15]]}{' ...' if len(need) > 15 else ''}. "
            f"Expected in {cfg['paths']['raw_dir']}/{cfg['raw_files']['da_prices']} (then run `clean`).")
    feats, helpers = build_features(days, tables, cfg)
    check_forecastable(feats, cfg)
    data = M.training_frame(feats, helpers, tables["da_smp"])
    data = data.merge(lear_inputs(days, tables, cfg), on=["delivery_day", "hour"], how="left")
    data = data.sort_values("ts_begin").reset_index(drop=True)

    pred = pd.DataFrame({"ts_begin": data["ts_begin"], "actual": data["y_raw"]})
    if forecast_file:
        fc = read_forecast_csv(forecast_file, cfg)
        pred["submission"] = fc.reindex(pd.DatetimeIndex(pred["ts_begin"])).to_numpy()
        main, n_f = "submission", int(pred["submission"].notna().sum())
        log.info("scoring forecast file %s: %d of %d hours found", forecast_file, n_f, len(pred))
    else:
        path = Path(model_path) if model_path else M.default_model_path(cfg, "lgbm_linear")
        if not path.exists():
            raise FileNotFoundError(f"frozen model {path} not found: run `python -m pjmda train --until "
                                    f"{cfg['train']['end']}` first")
        bundle = M.load_lgbm(path)
        pred["lgbm_linear"], n_clip = M.predict_lgbm(bundle, data, cfg)
        main = "lgbm_linear"
        log.info("frozen model %s (trained %s); %d hours clipped", path.name, bundle["training_window"], n_clip)
    for name in ("gbm_const", "lear"):
        p = M.default_model_path(cfg, name)
        if not p.exists():
            log.warning("baseline %s skipped: %s not found (run `train`)", name, p.name)
            continue
        if name == "lear":
            pred[name], _ = M.predict_lear(M.load_lear(p), data, cfg)
        else:
            pred[name], _ = M.predict_lgbm(M.load_lgbm(p), data, cfg)
    for name, col in M.NAIVE.items():                       # from the supplied actuals (D-1 / D-7 known at cut-off)
        pred[name] = data[col].to_numpy()

    models = [c for c in pred.columns if c not in ("ts_begin", "actual")]
    no_actual = int(pred["actual"].isna().sum())
    no_fc = int(pred[main].isna().sum())
    log.info("hours in range: %d; without actual: %d; without forecast: %d", len(pred), no_actual, no_fc)
    pred.to_parquet(rdir / "predictions.parquet", index=False)
    metrics = score_predictions(pred, data, cfg, models)
    metrics.to_csv(rdir / "metrics.csv", index=False)
    log_overall(metrics, f"test {run_name}: overall")
    write_run_meta(rdir, kind="test", main_model=main, models=models, start=start, end=end,
                   model=str(model_path or ""), forecast_file=str(forecast_file or ""))
    plots.summary_plots(pred, data, rdir, cfg, main)
    if plots_flag:
        plots.month_plots(pred, plots.plots_dir(cfg, run_name), cfg, main)
    return metrics
