"""Forecast-vs-actual plots for ANY predictions parquet (ts_begin, actual, one column per model).

The same code serves walk-forward backtest results (outputs/backtest/<run>) and 2026 test
results (outputs/test/<run>). Plain matplotlib, PNG output, one function per plot type.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pjmda.config import dir_path  # noqa: E402

log = logging.getLogger(__name__)

COLORS = {"actual": "black", "lgbm_linear": "tab:blue", "gbm_const": "tab:green", "lear": "tab:purple",
          "naive_week": "tab:orange", "naive_day": "tab:red", "submission": "tab:blue"}


# -------------------------------------------------------------------- loading
def plots_dir(cfg: dict, run: str, create: bool = True) -> Path:
    p = dir_path(cfg, "outputs_dir") / "plots" / run
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def load_predictions(cfg: dict, pred_path: str | Path | None = None, run: str | None = None,
                     source: str = "backtest") -> tuple[pd.DataFrame, str, str]:
    """Return (predictions, run name, main model). Works on a backtest that is still running."""
    if pred_path:
        path = Path(pred_path)
        run = run or path.parent.name
    elif run:
        path = dir_path(cfg, "outputs_dir") / source / run / "predictions.parquet"
    else:
        raise ValueError("give --pred or --run")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    pred = pd.read_parquet(path)
    meta_path = path.parent / "run_meta.json"
    main = cfg["backtest"]["main_model"]
    if meta_path.exists():
        main = json.loads(meta_path.read_text()).get("main_model", main)
    models = [c for c in pred.columns if c not in ("ts_begin", "actual")]
    if main not in models:
        main = models[0]
    return pred, run, main


def _local(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = pred.copy()
    ts = p["ts_begin"].dt.tz_convert(cfg["timezone"])
    p["day"] = ts.dt.tz_localize(None).dt.normalize()
    p["he"] = ts.dt.hour + 1                      # local hour ending label
    return p


def _mae_rmse(a: pd.Series, f: pd.Series) -> tuple[float, float]:
    ok = a.notna() & f.notna()
    if not ok.any():
        return float("nan"), float("nan")
    e = (f[ok] - a[ok]).to_numpy()
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def _style(name: str) -> dict:
    if name == "actual":
        return {"color": COLORS["actual"], "lw": 2.2, "marker": "o", "ms": 3.5}
    return {"color": COLORS.get(name, None), "lw": 1.7,
            "ls": "-" if name in ("lgbm_linear", "submission") else "--"}


# ------------------------------------------------------------------- day plot
def day_plot(pred: pd.DataFrame, day, cfg: dict, main: str, models: list[str] | None = None,
             outdir: Path | None = None) -> Path:
    """Hourly forecast vs actual SMP for one delivery day (x = hour ending)."""
    models = models or [main]
    day = pd.Timestamp(day).normalize()
    p = _local(pred, cfg)
    d = p[p["day"] == day]
    if d.empty:
        raise ValueError(f"no predictions for {day.date()}")
    d = d.groupby("he").mean(numeric_only=True)       # a repeated hour is averaged into its label
    fig, ax = plt.subplots(figsize=tuple(cfg["plots"]["figsize_day"]))
    have_actual = d["actual"].notna().any()
    if have_actual:
        ax.plot(d.index, d["actual"], label="actual SMP", **_style("actual"))
    for m in models:
        mae, _ = _mae_rmse(d["actual"], d[m])
        lab = f"{m} (MAE {mae:.2f})" if have_actual else m
        ax.plot(d.index, d[m], label=lab, **_style(m))
    mae, rmse = _mae_rmse(d["actual"], d[main])
    if have_actual:
        title = f"{day.date()} | {main} | MAE {mae:.2f}  RMSE {rmse:.2f} $/MWh"
    else:
        title = f"{day.date()} | {main} | forecast only (actuals not available)"
    ax.set(title=title, xlabel="hour ending (local)", ylabel="SMP ($/MWh)", xticks=range(1, 25))
    ax.grid(alpha=0.3)
    ax.legend()
    outdir = outdir or plots_dir(cfg, "adhoc")
    out = Path(outdir) / f"day_{day.date()}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    return out


# ----------------------------------------------------------------- month plot
def _month_profile(p: pd.DataFrame, month: str, models: list[str]):
    """Mean actual and mean forecast by hour ending over days that have both values."""
    m = p[p["day"].dt.strftime("%Y-%m") == month]
    prof, n_days = {}, 0
    for name in models:
        ok = m[m["actual"].notna() & m[name].notna()]
        n_days = max(n_days, ok["day"].nunique())
        prof[name] = ok.groupby("he")[name].mean()
    ok_any = m[m["actual"].notna()]
    prof["actual"] = ok_any.groupby("he")["actual"].mean()
    return prof, n_days, m


def _draw_month(ax, p: pd.DataFrame, month: str, main: str, models: list[str], small: bool = False) -> None:
    prof, n_days, m = _month_profile(p, month, models)
    if len(prof["actual"]):
        ax.plot(prof["actual"].index, prof["actual"], label="actual", **_style("actual"))
    for name in models:
        ax.plot(prof[name].index, prof[name], label=name, **_style(name))
    ok = m[m["actual"].notna() & m[main].notna()]
    mae, rmse = _mae_rmse(ok["actual"], ok[main])         # on the hourly values, not the averaged profile
    ax.set_title(f"{month} | {main} | {n_days} days | MAE {mae:.1f} RMSE {rmse:.1f}", fontsize=8 if small else 11)
    ax.grid(alpha=0.3)


def month_plot(pred: pd.DataFrame, month: str, cfg: dict, main: str, models: list[str] | None = None,
               outdir: Path | None = None) -> Path:
    """Month-average hourly profile: actual vs forecast(s)."""
    models = models or [main]
    p = _local(pred, cfg)
    if not (p["day"].dt.strftime("%Y-%m") == month).any():
        raise ValueError(f"no predictions for month {month}")
    fig, ax = plt.subplots(figsize=tuple(cfg["plots"]["figsize_day"]))
    _draw_month(ax, p, month, main, models)
    ax.set(xlabel="hour ending (local)", ylabel="mean SMP ($/MWh)", xticks=range(1, 25))
    ax.legend()
    outdir = outdir or plots_dir(cfg, "adhoc")
    out = Path(outdir) / f"month_{month}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    return out


def months_overview(pred: pd.DataFrame, cfg: dict, main: str, models: list[str] | None = None,
                    outdir: Path | None = None) -> Path:
    """One month profile per subplot for every month in the file, shared y-axis."""
    models = models or [main]
    p = _local(pred, cfg)
    months = sorted(p["day"].dt.strftime("%Y-%m").unique())
    ncol = 4 if len(months) > 6 else 3
    nrow = int(np.ceil(len(months) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=tuple(cfg["plots"]["figsize_overview"]), sharey=True, squeeze=False)
    for ax, month in zip(axes.ravel(), months):
        _draw_month(ax, p, month, main, models, small=True)
        ax.set_xticks([1, 6, 12, 18, 24])
    for ax in axes.ravel()[len(months):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=len(labels))
    fig.supxlabel("hour ending (local)")
    fig.supylabel("mean SMP ($/MWh)")
    fig.suptitle(f"Month-average hourly profile: actual vs {', '.join(models)}", fontsize=13)
    outdir = outdir or plots_dir(cfg, "adhoc")
    out = Path(outdir) / "months_overview.png"
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    fig.savefig(out, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    return out


def month_plots(pred: pd.DataFrame, outdir: Path, cfg: dict, main: str, models: list[str] | None = None) -> list[Path]:
    """Every month plot plus the overview (used by backtest runs and `evaluate --plots`)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models = models or [main]
    p = _local(pred, cfg)
    outs = [month_plot(pred, m, cfg, main, models, outdir) for m in sorted(p["day"].dt.strftime("%Y-%m").unique())]
    outs.append(months_overview(pred, cfg, main, models, outdir))
    log.info("wrote %d plots to %s", len(outs), outdir)
    return outs


# --------------------------------------------------------------- summary plots
def summary_plots(pred: pd.DataFrame, data: pd.DataFrame, rdir: Path, cfg: dict, main: str) -> None:
    """Actual vs predicted time series (full period + worst weeks), error by hour, error by month."""
    rdir = Path(rdir)
    models = [c for c in pred.columns if c not in ("ts_begin", "actual")]
    p = _local(pred, cfg)
    dpi = int(cfg["plots"]["dpi"])
    shown = [m for m in (main, cfg["backtest"]["baseline"]) if m in models]

    # full period
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(p["ts_begin"], p["actual"], color="black", lw=0.6, label="actual")
    for m in shown:
        ax.plot(p["ts_begin"], p[m], lw=0.6, alpha=0.8, label=m, color=COLORS.get(m))
    ax.set(ylabel="SMP ($/MWh)", title=f"Actual vs predicted, full period ({main})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rdir / "timeseries_full.png", dpi=dpi)
    plt.close(fig)

    # worst weeks (by MAE of the main model)
    ok = p[p["actual"].notna() & p[main].notna()].copy()
    ok["week"] = ok["day"] - pd.to_timedelta(ok["day"].dt.dayofweek, unit="D")
    wk = ok.assign(ae=(ok[main] - ok["actual"]).abs()).groupby("week")["ae"].agg(["mean", "size"])
    wk = wk[wk["size"] >= 100].sort_values("mean", ascending=False)
    worst = list(wk.head(int(cfg["backtest"]["worst_weeks"])).index)
    if worst:
        fig, axes = plt.subplots(len(worst), 1, figsize=(14, 3.6 * len(worst)), squeeze=False)
        for ax, w in zip(axes.ravel(), worst):
            s = p[(p["day"] >= w) & (p["day"] < w + pd.Timedelta(days=7))]
            ax.plot(s["ts_begin"], s["actual"], color="black", lw=1.2, label="actual")
            for m in shown:
                ax.plot(s["ts_begin"], s[m], lw=1.1, label=m, color=COLORS.get(m))
            ax.set_title(f"week of {w.date()} | {main} MAE {wk.loc[w, 'mean']:.1f} $/MWh", fontsize=10)
            ax.grid(alpha=0.3)
        axes.ravel()[0].legend()
        fig.tight_layout()
        fig.savefig(rdir / "worst_weeks.png", dpi=dpi)
        plt.close(fig)

    # error by hour and by month, on hours scored for every model
    cm = p[p["actual"].notna() & p[models].notna().all(axis=1)].copy()
    cm["month"] = cm["day"].dt.strftime("%Y-%m")
    for col, name in (("he", "error_by_hour.png"), ("month", "error_by_month.png")):
        tab = pd.DataFrame({m: (cm[m] - cm["actual"]).abs().groupby(cm[col]).mean() for m in models})
        fig, ax = plt.subplots(figsize=(12, 4.5))
        tab.plot.bar(ax=ax, width=0.85, color=[COLORS.get(m) for m in models])
        ax.set(ylabel="MAE ($/MWh)", xlabel="hour ending" if col == "he" else "month",
               title=f"MAE by {'hour of day' if col == 'he' else 'month'}")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(rdir / name, dpi=dpi)
        plt.close(fig)
    log.info("summary plots written to %s", rdir)
