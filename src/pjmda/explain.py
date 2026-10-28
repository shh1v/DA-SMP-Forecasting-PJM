"""Post-hoc explainability: SHAP, permutation importance, ALE, error breakdown, per-day drivers.

The target is the SMP in $/MWh, so every attribution is directly in $/MWh. All wrappers
apply the model's saved scaler inside the predict function, so raw feature units are used
everywhere (permuted values are scaled exactly like real ones). SHAP uses the
model-agnostic PermutationExplainer because tree-path SHAP can ignore the leaf linear models.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pjmda import model as M  # noqa: E402
from pjmda.backtest import score_predictions  # noqa: E402
from pjmda.clean import effective_start, read_clean  # noqa: E402
from pjmda.config import dir_path  # noqa: E402

log = logging.getLogger(__name__)
DAY = pd.Timedelta(days=1)
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded", category=FutureWarning)


# ------------------------------------------------------------------ plumbing
def make_predict(bundle: dict):
    """Callable on RAW feature values (ndarray / DataFrame) -> predicted SMP ($/MWh)."""
    feats = bundle["features"]

    def predict(X) -> np.ndarray:
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=feats)
        return M.target_inverse(bundle["booster"].predict(M.apply_scaler(df, bundle["scaler"]).to_numpy()))

    return predict


class _Estimator:
    """Minimal estimator so sklearn's permutation_importance can call predict()."""

    def __init__(self, predict):
        self._p = predict

    def fit(self, X=None, y=None):          # sklearn only checks that `fit` exists
        return self

    def predict(self, X):
        return self._p(X)


def explain_dir(cfg: dict) -> Path:
    p = dir_path(cfg, "outputs_dir") / "explain"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _background(train_rows: pd.DataFrame, feats: list[str], n: int, seed: int) -> pd.DataFrame:
    return train_rows[feats].sample(n=min(n, len(train_rows)), random_state=seed).reset_index(drop=True)


def shap_values(bundle: dict, X: pd.DataFrame, background: pd.DataFrame, cfg: dict):
    """Model-agnostic SHAP values (PermutationExplainer), no interaction values."""
    import shap

    e = cfg["explain"]
    predict = make_predict(bundle)
    masker = shap.maskers.Independent(background.to_numpy(dtype=float), max_samples=len(background))
    explainer = shap.PermutationExplainer(predict, masker, feature_names=list(bundle["features"]),
                                          seed=int(e["seed"]))
    exp = explainer(X.to_numpy(dtype=float), max_evals=int(e["shap_max_evals"]), silent=True)
    base = float(np.mean(predict(background.to_numpy(dtype=float))))
    return np.asarray(exp.values), base


# ------------------------------------------------------------------- SHAP plots
def shap_plots(values: np.ndarray, X: pd.DataFrame, cfg: dict, out: Path) -> None:
    import shap

    feats = list(X.columns)
    dpi = int(cfg["plots"]["dpi"])
    imp = pd.Series(np.abs(values).mean(axis=0), index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot.barh(ax=ax, color="tab:blue")
    ax.set(xlabel="mean |SHAP| ($/MWh)", title="Global importance (mean |SHAP|, $/MWh)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=dpi)
    plt.close(fig)
    imp.sort_values(ascending=False).rename("mean_abs_shap").to_csv(out / "shap_importance.csv")

    plt.figure(figsize=(9, 8))
    shap.summary_plot(values, X.to_numpy(dtype=float), feature_names=feats, show=False, max_display=len(feats))
    plt.title("SHAP beeswarm ($/MWh)")
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=dpi)
    plt.close("all")

    colour = cfg["explain"].get("dependence_color", {})
    for f in cfg["explain"]["dependence_features"]:
        if f not in feats:
            continue
        c = colour.get(f, "hour")
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(X[f], values[:, feats.index(f)], c=X[c] if c in feats else None, s=8, cmap="viridis", alpha=0.7)
        if c in feats:
            fig.colorbar(sc, ax=ax, label=c)
        ax.set(xlabel=f, ylabel=f"SHAP value of {f} ($/MWh)", title=f"SHAP dependence: {f} (colour: {c})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"shap_dependence_{f}.png", dpi=dpi)
        plt.close(fig)


# ------------------------------------------------------- permutation importance
def permutation_table(bundle: dict, X: pd.DataFrame, y: pd.Series, cfg: dict, out: Path) -> pd.DataFrame:
    """Increase in MAE ($/MWh) when a feature is shuffled (10 repeats)."""
    from sklearn.inspection import permutation_importance

    ok = y.notna().to_numpy()
    X, y = X.loc[ok].reset_index(drop=True), y.loc[ok].reset_index(drop=True)

    def neg_mae(est, Xp, yp):
        return -float(np.mean(np.abs(est.predict(Xp) - np.asarray(yp))))

    r = permutation_importance(_Estimator(make_predict(bundle)), X, y, scoring=neg_mae,
                               n_repeats=int(cfg["explain"]["perm_repeats"]),
                               random_state=int(cfg["explain"]["seed"]), n_jobs=1)
    tab = pd.DataFrame({"feature": X.columns, "mae_increase": r.importances_mean, "std": r.importances_std})
    tab = tab.sort_values("mae_increase", ascending=False).reset_index(drop=True)
    tab.to_csv(out / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    t = tab.iloc[::-1]
    ax.barh(t["feature"], t["mae_increase"], xerr=t["std"], color="tab:orange")
    ax.set(xlabel="increase in MAE when shuffled ($/MWh)", title="Permutation importance (MAE, 10 repeats)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "permutation_importance.png", dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    return tab


# -------------------------------------------------------------------------- ALE
def ale_1d(predict, X: pd.DataFrame, feature: str, bins: int) -> pd.DataFrame:
    """First-order accumulated local effects with quantile bins (centred, $/MWh)."""
    x = X[feature].to_numpy(dtype=float)
    ok = ~np.isnan(x)
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame({"x": edges, "ale": 0.0})
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    effects, counts = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        rows = np.flatnonzero(ok & (idx == k))
        if not len(rows):
            continue
        lo, hi = X.iloc[rows].copy(), X.iloc[rows].copy()
        lo[feature], hi[feature] = edges[k], edges[k + 1]
        effects[k] = np.mean(predict(hi.to_numpy(dtype=float)) - predict(lo.to_numpy(dtype=float)))
        counts[k] = len(rows)
    ale = np.concatenate([[0.0], np.cumsum(effects)])
    mid = (ale[:-1] + ale[1:]) / 2
    centre = np.average(mid, weights=np.maximum(counts, 1e-9))
    return pd.DataFrame({"x": edges, "ale": ale - centre})


def ale_plots(bundle: dict, X: pd.DataFrame, cfg: dict, out: Path) -> pd.DataFrame:
    """ALE (not PDP: nl, tight and ren_share are correlated). Also shows if monotone shapes hold."""
    predict = make_predict(bundle)
    feats = [f for f in cfg["explain"]["ale_features"] if f in X.columns]
    tabs, fig = [], None
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 4), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        t = ale_1d(predict, X, f, int(cfg["explain"]["ale_bins"]))
        tabs.append(t.assign(feature=f))
        ax.plot(t["x"], t["ale"], marker="o", ms=3)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(xlabel=f, ylabel="ALE ($/MWh)", title=f"ALE: {f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ale.png", dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(out / "ale.csv", index=False)
    return tab


# ----------------------------------------------------------- per-day explanation
def top_drivers(values: np.ndarray, X: pd.DataFrame, on_peak: np.ndarray, base: float, n: int) -> pd.DataFrame:
    """Top n drivers of the daily on-peak average forecast vs the base value ($/MWh)."""
    m = np.asarray(on_peak) == 1
    note = "on-peak hours"
    if not m.any():
        m = np.ones(len(X), dtype=bool)
        note = "all hours (no on-peak hours on this day)"
    contrib = pd.Series(values[m].mean(axis=0), index=X.columns)
    tab = pd.DataFrame({
        "feature": contrib.index, "shap_usd_mwh": contrib.to_numpy(),
        "mean_value": X.loc[m].mean().to_numpy(),
    })
    tab = tab.reindex(tab["shap_usd_mwh"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)
    tab.attrs.update(base=base, pred=base + float(contrib.sum()), note=note)
    return tab


def explain_day(bundle: dict, day_feats: pd.DataFrame, on_peak: np.ndarray, background: pd.DataFrame,
                cfg: dict) -> pd.DataFrame:
    """SHAP drivers of one forecast day (used by `forecast --explain`)."""
    X = day_feats[bundle["features"]].reset_index(drop=True)
    vals, base = shap_values(bundle, X, background, cfg)
    return top_drivers(vals, X, on_peak, base, int(cfg["explain"]["top_drivers"]))


def format_drivers(day, tab: pd.DataFrame) -> str:
    a = tab.attrs
    lines = [f"Top drivers of the {pd.Timestamp(day).date()} average over {a['note']} "
             f"(base {a['base']:.2f} -> forecast {a['pred']:.2f} $/MWh):"]
    lines.append(tab.assign(shap_usd_mwh=tab["shap_usd_mwh"].round(2), mean_value=tab["mean_value"].round(3))
                 .to_string(index=False))
    return "\n".join(lines)


def training_background(cfg: dict, tables: dict, last_day, n: int | None = None) -> pd.DataFrame:
    """Background rows drawn from the training window (delivery days up to last_day)."""
    eff = effective_start(cfg)
    last = min(pd.Timestamp(last_day), pd.Timestamp(cfg["train"]["end"]))
    data = M.prepare_data(cfg, tables, eff, last)
    win = M.window(data, cfg, eff, last)
    return _background(win, M.feature_names(cfg), n or int(cfg["explain"]["shap_background"]),
                       int(cfg["explain"]["seed"]))


# ------------------------------------------------------------------------ driver
def run_explain(cfg: dict, start, end, model_path: str | Path | None = None, tables: dict | None = None) -> Path:
    """Explain the model on a backtest period; outputs go to outputs/explain/."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tables = tables or read_clean(cfg)
    out = explain_dir(cfg)
    e = cfg["explain"]
    eff = effective_start(cfg)
    data = M.prepare_data(cfg, tables, eff, end)
    if model_path:
        bundle = M.load_lgbm(model_path)
        log.info("explaining saved model %s", model_path)
    else:                                   # the model trained just before the period
        win = M.window(data, cfg, eff, start - DAY)
        bundle = M.fit_lgbm(win, cfg, start, linear=True)
        log.info("trained a model on %s -> %s for the explanation", bundle["window"][0], bundle["window"][1])
    feats = bundle["features"]
    period = data[(data["delivery_day"] >= start) & (data["delivery_day"] <= end) & data["g_comp"].notna()]
    train_rows = M.window(data, cfg, eff, min(start - DAY, pd.Timestamp(bundle["window"][1])))
    background = _background(train_rows, feats, int(e["shap_background"]), int(e["seed"]))
    sample = period.sample(n=min(int(e["shap_rows"]), len(period)), random_state=int(e["seed"])).sort_values("ts_begin")
    X = sample[feats].reset_index(drop=True)

    log.info("SHAP: %d rows, background %d, max_evals %s", len(X), len(background), e["shap_max_evals"])
    vals, base = shap_values(bundle, X, background, cfg)
    shap_plots(vals, X, cfg, out)
    pd.concat([sample[["ts_begin"]].reset_index(drop=True),
               pd.DataFrame(vals, columns=[f"shap_{f}" for f in feats])], axis=1).assign(base_value=base).to_parquet(
        out / "shap_values.parquet", index=False)

    log.info("permutation importance on %d rows", len(period))
    perm = permutation_table(bundle, period[feats].reset_index(drop=True), period["y_raw"], cfg, out)
    log.info("permutation importance (MAE increase, $/MWh):\n%s", perm.head(10).round(3).to_string(index=False))
    ale_plots(bundle, period[feats].reset_index(drop=True), cfg, out)

    # error breakdown of the explained model on the period (same tables as the backtest)
    p = period.copy()
    pred = pd.DataFrame({"ts_begin": p["ts_begin"], "actual": p["y_raw"]})
    pred["lgbm_linear"], _ = M.predict_lgbm(bundle, p, cfg)
    pred["naive_week"] = p["smp_lag7"].to_numpy()
    score_predictions(pred, data, cfg, ["lgbm_linear", "naive_week"], "naive_week").to_csv(
        out / "error_breakdown.csv", index=False)

    # per-day drivers: last day of the period
    day = period["delivery_day"].max()
    dd = period[period["delivery_day"] == day]
    Xd = dd[feats].reset_index(drop=True)
    vd, bd = shap_values(bundle, Xd, background, cfg)
    tab = top_drivers(vd, Xd, dd["on_peak"].to_numpy(), bd, int(e["top_drivers"]))
    tab.to_csv(out / f"drivers_{day.date()}.csv", index=False)
    print(format_drivers(day, tab))
    log.info("explainability outputs written to %s", out)
    return out
