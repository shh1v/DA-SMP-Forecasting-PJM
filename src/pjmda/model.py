"""Scaling, LightGBM fit/predict, LEAR and naive baselines, model persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from pjmda.clean import effective_start
from pjmda.config import config_hash, dir_path, git_commit
from pjmda.features import (
    MissingInputError, build_features, check_forecastable, feature_names, lear_inputs, missing_price_days,
)
from pjmda.vintage import date_range

log = logging.getLogger(__name__)

MODELS = ("lgbm_linear", "gbm_const", "lear")
NAIVE = {"naive_week": "smp_lag7", "naive_day": "smp_lag1"}


# ------------------------------------------------------------------- target
def target_fn(smp: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
    """Training target. Identity: the model predicts the SMP in $/MWh directly.
    Swap the target here (and in target_inverse) to change what the model learns."""
    return np.asarray(smp, dtype=float)


def target_inverse(pred: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
    """Map model output back to $/MWh (identity)."""
    return np.asarray(pred, dtype=float)


def clip_predictions(pred: np.ndarray, cfg: dict) -> tuple[np.ndarray, int]:
    lo, hi = float(cfg["model"]["clip_min"]), float(cfg["model"]["clip_max"])
    n = int(((pred < lo) | (pred > hi)).sum())
    return np.clip(pred, lo, hi), n


# --------------------------------------------------------------- training data
def prepare_data(cfg: dict, tables: dict, start, end) -> pd.DataFrame:
    """One row per delivery hour: model features, helper flags, LEAR inputs and the actual SMP."""
    days = date_range(start, end)
    feats, helpers = build_features(days, tables, cfg)
    df = training_frame(feats, helpers, tables["da_smp"])
    df = df.merge(lear_inputs(days, tables, cfg), on=["delivery_day", "hour"], how="left")
    return df.sort_values("ts_begin").reset_index(drop=True)


def training_frame(features: pd.DataFrame, helpers: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    """Features + helper flags + actual SMP (column `y_raw`) on ts_begin."""
    h = helpers[["ts_begin", "cref_warmup", "on_peak"]]
    df = features.merge(h, on="ts_begin", how="left").merge(
        smp[["ts_begin", "smp"]].rename(columns={"smp": "y_raw"}), on="ts_begin", how="left")
    return df


def window(df: pd.DataFrame, cfg: dict, start, last_day) -> pd.DataFrame:
    """Rows with start <= delivery day <= min(last_day, train.end) that have a target."""
    last = min(pd.Timestamp(last_day), pd.Timestamp(cfg["train"]["end"]))
    first = max(pd.Timestamp(start), pd.Timestamp(cfg["train"]["start"]))
    m = (df["delivery_day"] >= first) & (df["delivery_day"] <= last) & df["y_raw"].notna() & df["g_comp"].notna()
    if cfg["capacity"]["exclude_warmup_from_training"]:
        m &= df["cref_warmup"] == 0
    return df.loc[m]


# --------------------------------------------------------------------- scaler
def fit_scaler(X: pd.DataFrame, cfg: dict) -> dict:
    """Mean/std of every NUMERICAL feature on the training window (NaN aware)."""
    cats = set(cfg["features"]["categorical"])
    sc = {}
    for c in X.columns:
        if c in cats:
            continue
        mu, sd = float(np.nanmean(X[c])), float(np.nanstd(X[c]))
        sc[c] = {"mean": mu, "std": sd if sd > 1e-12 else 1.0}
    return sc


def apply_scaler(X: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Standardise numerical features; categoricals and NaNs are left untouched."""
    out = X.astype(float).copy()
    for c, s in scaler.items():
        if c in out.columns:
            out[c] = (out[c] - s["mean"]) / s["std"]
    return out


def sample_weights(days: pd.Series, ref_day, half_life: float) -> np.ndarray:
    """w = 0.5 ** (age_days / half_life), age measured to the forecast day."""
    age = (pd.Timestamp(ref_day) - days).dt.days.to_numpy(dtype=float)
    return 0.5 ** (np.maximum(age, 0.0) / half_life)


# ---------------------------------------------------------------------- LightGBM
def tuned_params(cfg: dict) -> dict:
    """Params saved by `tune` (models/params.json), when present and enabled."""
    p = dir_path(cfg, "models_dir") / "params.json"
    if cfg["model"].get("use_tuned_params") and p.exists():
        return json.loads(p.read_text()).get("params", {})
    return {}


def lgbm_params(cfg: dict, features: list[str], linear: bool, overrides: dict | None = None) -> tuple[dict, int]:
    """LightGBM parameter dict and the number of boosting rounds."""
    p = {**cfg["model"]["lgbm"], **tuned_params(cfg), **(overrides or {})}
    n_rounds = int(p.pop("n_estimators"))
    params = {
        **p,
        "linear_tree": bool(linear),
        "tree_learner": "serial",
        "seed": int(cfg["model"]["seed"]),
        "deterministic": True,
        "force_row_wise": True,
        "monotone_constraints": [int(cfg["features"]["monotone"].get(f, 0)) for f in features],
    }
    if not linear:
        params.pop("linear_lambda", None)
    return params, n_rounds


def fit_lgbm(df: pd.DataFrame, cfg: dict, ref_day, linear: bool = True,
             features: list[str] | None = None, overrides: dict | None = None) -> dict:
    """Fit LightGBM on the training window `df` (already restricted to the window).

    Early stopping uses the LAST valid_days days of the window (time ordered). With
    refit_full the model is refitted on the whole window using the best iteration.
    """
    features = features or feature_names(cfg)
    m = cfg["model"]
    cats = [c for c in cfg["features"]["categorical"] if c in features]
    df = df.sort_values("ts_begin")
    X = df[features]
    scaler = fit_scaler(X, cfg)
    Xs = apply_scaler(X, scaler).to_numpy()
    y = target_fn(df["y_raw"].to_numpy(), X)
    w = sample_weights(df["delivery_day"], ref_day, float(m["half_life_days"]))
    params, n_rounds = lgbm_params(cfg, features, linear, overrides)

    last_day = df["delivery_day"].max()
    is_val = (df["delivery_day"] > last_day - pd.Timedelta(days=int(m["valid_days"]))).to_numpy()
    kw = dict(feature_name=features, categorical_feature=cats, free_raw_data=False)
    ds_tr = lgb.Dataset(Xs[~is_val], y[~is_val], weight=w[~is_val], params=params, **kw)
    ds_va = lgb.Dataset(Xs[is_val], y[is_val], weight=w[is_val], reference=ds_tr, params=params, **kw)
    booster = lgb.train(
        params, ds_tr, num_boost_round=n_rounds, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(int(m["early_stopping_rounds"]), verbose=False)],
    )
    best = int(booster.best_iteration or n_rounds)
    if m["refit_full"]:
        ds_full = lgb.Dataset(Xs, y, weight=w, params=params, **kw)
        booster = lgb.train(params, ds_full, num_boost_round=best)
    return {
        "kind": "lgbm", "linear": linear, "booster": booster, "features": features,
        "categorical": cats, "scaler": scaler, "best_iteration": best,
        "window": [str(df["delivery_day"].min().date()), str(last_day.date())], "n_rows": int(len(df)),
        "params": {k: v for k, v in params.items() if k != "monotone_constraints"},
        "ref_day": str(pd.Timestamp(ref_day).date()),
    }


def predict_lgbm(bundle: dict, X: pd.DataFrame, cfg: dict, clip: bool = True) -> tuple[np.ndarray, int]:
    """Predictions in $/MWh (scaler applied inside) and the number of clipped hours."""
    Xs = apply_scaler(X[bundle["features"]], bundle["scaler"]).to_numpy()
    pred = target_inverse(bundle["booster"].predict(Xs), X)
    if not clip:
        return pred, 0
    return clip_predictions(pred, cfg)


def save_lgbm(bundle: dict, name: str, cfg: dict) -> Path:
    """models/<name>.txt (LightGBM native, includes leaf linear models) + <name>.json."""
    mdir = dir_path(cfg, "models_dir", create=True)
    path = mdir / f"{name}.txt"
    bundle["booster"].save_model(str(path))
    meta = {k: v for k, v in bundle.items() if k != "booster"}
    meta.update({
        "feature_list": bundle["features"], "categorical_features": bundle["categorical"],
        "scaler": bundle["scaler"], "training_window": bundle["window"],
        "config_hash": config_hash(cfg), "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "lightgbm": lgb.__version__, "pandas": pd.__version__,
    })
    (mdir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("saved %s (window %s, %d rows, %d trees)", path.name, bundle["window"], bundle["n_rows"],
             bundle["booster"].num_trees())
    return path


def load_lgbm(path: str | Path) -> dict:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["booster"] = lgb.Booster(model_file=str(path))
    meta["features"] = meta["feature_list"]
    meta["categorical"] = meta["categorical_features"]
    return meta


def default_model_path(cfg: dict, name: str = "lgbm_linear") -> Path:
    ext = ".joblib" if name == "lear" else ".txt"
    return dir_path(cfg, "models_dir") / f"{name}_{cfg['train']['end']}{ext}"


# -------------------------------------------------------------------------- LEAR
def _lear_cols(cfg: dict) -> tuple[list[str], list[str]]:
    price = [f"smp_lag{k}" for k in cfg["lear"]["price_lags"]] + [f"smp_d1_h{h}" for h in range(1, 25)]
    return price, list(cfg["lear"]["features"])


def _lear_matrix(df: pd.DataFrame, cfg: dict, a: float, b: float) -> pd.DataFrame:
    """asinh-transformed price lags + fundamentals + calendar dummies."""
    price, fund = _lear_cols(cfg)
    X = pd.DataFrame(index=df.index)
    for c in price:
        X[c] = np.arcsinh((df[c] - a) / b)
    for c in fund:
        X[c] = df[c]
    for d in range(1, 7):
        X[f"dow_{d}"] = (df["dow"] == d).astype(float)
    for mth in range(2, 13):
        X[f"month_{mth}"] = (df["month"] == mth).astype(float)
    X["is_holiday"] = df["is_holiday"].astype(float)
    return X


def fit_lear(df: pd.DataFrame, cfg: dict) -> dict:
    """24 per-hour LassoCV models on asinh-transformed, standardised SMP (Lago et al. 2021)."""
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit

    c = cfg["lear"]
    y_all = df["y_raw"].to_numpy()
    a = float(np.nanmedian(y_all))
    mad = float(np.nanmedian(np.abs(y_all - a))) * 1.4826
    b = mad if mad > 1e-9 else float(np.nanstd(y_all))
    models = {}
    for h in range(1, 25):
        d = df[df["hour"] == h].sort_values("ts_begin")
        X = _lear_matrix(d, cfg, a, b)
        y = np.arcsinh((d["y_raw"] - a) / b)
        ok = X.notna().all(axis=1) & y.notna()
        X, y = X[ok], y[ok]
        if len(X) < 50:
            log.warning("LEAR hour %d: only %d usable rows, skipped", h, len(X))
            continue
        mu, sd = X.mean(), X.std().replace(0, 1.0)
        ym, ys = float(y.mean()), float(y.std() or 1.0)
        model = LassoCV(cv=TimeSeriesSplit(n_splits=int(c["n_splits"])), alphas=int(c["n_alphas"]),
                        max_iter=int(c["max_iter"]), random_state=int(cfg["model"]["seed"]))
        model.fit(((X - mu) / sd).to_numpy(), ((y - ym) / ys).to_numpy())
        models[h] = {"model": model, "mu": mu, "sd": sd, "ym": ym, "ys": ys, "cols": list(X.columns)}
    return {"kind": "lear", "a": a, "b": b, "models": models,
            "window": [str(df["delivery_day"].min().date()), str(df["delivery_day"].max().date())],
            "n_rows": int(len(df))}


def predict_lear(bundle: dict, df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    out = np.full(len(df), np.nan)
    hours = df["hour"].to_numpy()
    X_all = _lear_matrix(df, cfg, bundle["a"], bundle["b"])
    for h, m in bundle["models"].items():
        idx = np.flatnonzero(hours == h)
        if not len(idx):
            continue
        X = X_all.iloc[idx][m["cols"]].fillna(m["mu"])               # NaN -> training mean
        z = m["model"].predict(((X - m["mu"]) / m["sd"]).to_numpy()) * m["ys"] + m["ym"]
        out[idx] = bundle["a"] + bundle["b"] * np.sinh(z)
    ok = ~np.isnan(out)
    out[ok], n_clip = clip_predictions(out[ok], cfg)
    return out, n_clip


def save_lear(bundle: dict, name: str, cfg: dict) -> Path:
    mdir = dir_path(cfg, "models_dir", create=True)
    path = mdir / f"{name}.joblib"
    joblib.dump(bundle, path)
    meta = {"kind": "lear", "training_window": bundle["window"], "n_rows": bundle["n_rows"],
            "hours_fitted": sorted(bundle["models"]), "config_hash": config_hash(cfg),
            "git_commit": git_commit(), "created": datetime.now().isoformat(timespec="seconds")}
    (mdir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return path


def load_lear(path: str | Path) -> dict:
    return joblib.load(path)


# ------------------------------------------------------------------- dispatch
def fit_model(name: str, df: pd.DataFrame, cfg: dict, ref_day, features: list[str] | None = None,
              overrides: dict | None = None) -> dict:
    """Fit one of MODELS on the training frame `df`."""
    if name == "lgbm_linear":
        return fit_lgbm(df, cfg, ref_day, linear=True, features=features, overrides=overrides)
    if name == "gbm_const":
        return fit_lgbm(df, cfg, ref_day, linear=False, features=features, overrides=overrides)
    if name == "lear":
        return fit_lear(df, cfg)
    raise ValueError(f"unknown model {name}")


def predict_model(name: str, bundle: dict | None, df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    """Predictions for the rows of `df` (features [+ LEAR inputs])."""
    if name in NAIVE:
        return df[NAIVE[name]].to_numpy(dtype=float), 0
    if name in ("lgbm_linear", "gbm_const"):
        return predict_lgbm(bundle, df, cfg)
    if name == "lear":
        return predict_lear(bundle, df, cfg)
    raise ValueError(f"unknown model {name}")


# ------------------------------------------------------- train / forecast entry points
def train_and_save(cfg: dict, tables: dict, until, names: tuple[str, ...] = MODELS) -> dict[str, Path]:
    """Fit the models on delivery days effective_start..until (never beyond train.end) and save them."""
    until = pd.Timestamp(until)
    if until > pd.Timestamp(cfg["train"]["end"]):
        raise ValueError(f"--until {until.date()} is after train.end {cfg['train']['end']}: "
                         "no fit may use delivery days after train.end")
    eff = effective_start(cfg)
    data = prepare_data(cfg, tables, eff, until)
    win = window(data, cfg, eff, until)
    if win["delivery_day"].nunique() < int(cfg["model"]["min_train_days"]):
        raise ValueError(f"only {win['delivery_day'].nunique()} training days available")
    log.info("TRAIN until %s: %d rows, %s -> %s", until.date(), len(win), win["delivery_day"].min().date(),
             win["delivery_day"].max().date())
    paths = {}
    for name in names:
        bundle = fit_model(name, win, cfg, until + pd.Timedelta(days=1))
        stem = f"{name}_{until.date()}"
        paths[name] = save_lear(bundle, stem, cfg) if name == "lear" else save_lgbm(bundle, stem, cfg)
    return paths


def forecast_days(cfg: dict, days, tables: dict, model_path: str | Path | None = None,
                  retrain: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Forecast every hour of the given delivery days with the saved (or freshly refitted) model.
    Returns (forecast table, model bundle, feature table).

    Features use each day's own cut-off. If any DA price input (D-1 or D-7 results) is
    missing the function stops with an error: there is no fallback model.
    """
    days = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().sort_values()
    days = pd.DatetimeIndex(days.to_numpy(dtype="datetime64[ns]"))
    missing = missing_price_days(days, tables["da_smp"], cfg)
    if missing:
        need = sorted({d for v in missing.values() for d in v})
        raise MissingInputError(
            f"DA price inputs (D-1 / D-7 results) are missing or incomplete for {len(need)} day(s): "
            f"{[str(d.date()) for d in need[:15]]}{' ...' if len(need) > 15 else ''}; they are needed to "
            f"forecast {[str(d.date()) for d in sorted(missing)[:5]]}{' ...' if len(missing) > 5 else ''}. "
            f"Expected in {cfg['paths']['raw_dir']}/{cfg['raw_files']['da_prices']} (then run `clean`).")
    feats, helpers = build_features(days, tables, cfg)
    check_forecastable(feats, cfg)

    if retrain:
        eff = effective_start(cfg)
        last = min(days.min() - pd.Timedelta(days=1), pd.Timestamp(cfg["train"]["end"]))
        win = window(prepare_data(cfg, tables, eff, last), cfg, eff, last)
        log.info("RETRAIN on %s -> %s (%d rows), capped at train.end %s", win["delivery_day"].min().date(),
                 win["delivery_day"].max().date(), len(win), cfg["train"]["end"])
        bundle = fit_lgbm(win, cfg, days.min(), linear=True)
    else:
        path = Path(model_path) if model_path else default_model_path(cfg, "lgbm_linear")
        if not path.exists():
            raise FileNotFoundError(f"model {path} not found: run `python -m pjmda train --until "
                                    f"{cfg['train']['end']}` first")
        bundle = load_lgbm(path)
    raw, _ = predict_lgbm(bundle, feats, cfg, clip=False)
    pred, _ = clip_predictions(raw, cfg)
    h = helpers.set_index("ts_begin")
    out = feats[["ts_begin", "delivery_day", "hour"]].copy()
    out["datetime_beginning_ept"] = out["ts_begin"].dt.tz_convert(cfg["timezone"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["da_price_forecast"] = pred
    out["clipped"] = ((raw < float(cfg["model"]["clip_min"])) | (raw > float(cfg["model"]["clip_max"]))).astype(int)
    out["on_peak"] = h.loc[out["ts_begin"], "on_peak"].to_numpy()
    for c in h.columns:
        if c.startswith(("stale_", "filled_")) or c in ("gas_filled", "cref_warmup"):
            out[c] = h.loc[out["ts_begin"], c].to_numpy()
    return out, bundle, feats
