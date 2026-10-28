"""Command-line entry point: python -m pjmda <command>."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from pjmda.config import dir_path, load_config, resolve, setup_logging

log = logging.getLogger("pjmda")


# ------------------------------------------------------------------- helpers
def _csv_list(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _days_from_args(args) -> pd.DatetimeIndex:
    from pjmda.vintage import date_range
    if getattr(args, "date", None):
        return date_range(args.date, args.date)
    if args.start and args.end:
        return date_range(args.start, args.end)
    raise SystemExit("give --date DATE or --start DATE --end DATE")


# ------------------------------------------------------------------ commands
def cmd_clean(args, cfg):
    from pjmda.clean import clean_all
    clean_all(cfg, raw_dir=args.raw_dir)


def cmd_features(args, cfg):
    from pjmda.clean import effective_start, read_clean
    from pjmda.features import build_features
    from pjmda.vintage import date_range
    start = pd.Timestamp(args.start) if args.start else effective_start(cfg)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(cfg["train"]["end"])
    feats, helpers = build_features(date_range(start, end), read_clean(cfg), cfg)
    fdir = dir_path(cfg, "features_dir", create=True)
    feats.to_parquet(fdir / "features.parquet", index=False)
    helpers.to_parquet(fdir / "helpers.parquet", index=False)
    log.info("wrote %s (%d rows, 19 model features) and helpers.parquet (%d columns)",
             fdir / "features.parquet", len(feats), helpers.shape[1])


def cmd_tune(args, cfg):
    from pjmda.backtest import tune
    tune(cfg)


def cmd_backtest(args, cfg):
    from pjmda.backtest import run_backtest
    run_backtest(cfg, args.start, args.end, models=_csv_list(args.models), retrain_every=args.retrain_every,
                 run_name=args.run_name, train_start=args.train_start, ablation=args.ablation)


def cmd_train(args, cfg):
    from pjmda.clean import read_clean
    from pjmda.model import train_and_save
    paths = train_and_save(cfg, read_clean(cfg), args.until)
    for name, p in paths.items():
        print(f"{name}: {p}")


def cmd_forecast(args, cfg):
    import numpy as np
    from pjmda import explain, plots
    from pjmda import model as M
    from pjmda.clean import clean_all, read_clean

    t0 = time.time()
    clean_all(cfg, raw_dir=args.raw_dir, only_changed=True)        # re-cleans only changed raw files
    tables = read_clean(cfg)
    days = _days_from_args(args)
    out, bundle, feats = M.forecast_days(cfg, days, tables, model_path=args.model, retrain=args.retrain)

    out_path = resolve(args.out) if args.out else dir_path(cfg, "outputs_dir", create=True) / \
        f"forecast_{days.min().date()}_{days.max().date()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["datetime_beginning_ept", "da_price_forecast"]].round({"da_price_forecast": 4}).to_csv(
        out_path, index=False)

    print(f"\nForecast written to {out_path} ({len(out)} hours, {len(days)} day(s))")
    rows = []
    for d, g in out.groupby("delivery_day"):
        on, off = g[g["on_peak"] == 1], g[g["on_peak"] == 0]
        imax, imin = g["da_price_forecast"].idxmax(), g["da_price_forecast"].idxmin()
        flags = [c for c in g.columns if c.startswith(("stale_", "filled_", "gas_filled")) and g[c].fillna(0).sum() > 0]
        rows.append({
            "day": d.date(), "hours": len(g),
            "on_peak_avg": on["da_price_forecast"].mean() if len(on) else np.nan,
            "off_peak_avg": off["da_price_forecast"].mean() if len(off) else np.nan,
            "min": f"{g.loc[imin, 'da_price_forecast']:.1f}@HE{g.loc[imin, 'hour']}",
            "max": f"{g.loc[imax, 'da_price_forecast']:.1f}@HE{g.loc[imax, 'hour']}",
            "clipped": int(g["clipped"].sum()), "flags": ",".join(flags) or "-"})
    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 40):
        print(summary.round(2).to_string(index=False) if len(summary) <= 40
              else summary.head(20).round(2).to_string(index=False) + f"\n... ({len(summary)} days)")
    if len(summary) > 40:
        summary.to_csv(out_path.with_suffix(".summary.csv"), index=False)
    if out["clipped"].sum():
        log.warning("%d forecast hour(s) were clipped to [%s, %s]", int(out["clipped"].sum()),
                    cfg["model"]["clip_min"], cfg["model"]["clip_max"])

    if args.explain:
        background = explain.training_background(cfg, tables, min(days.min() - pd.Timedelta(days=1),
                                                                  pd.Timestamp(cfg["train"]["end"])))
        for d, g in out.groupby("delivery_day"):
            f = feats[feats["delivery_day"] == d]
            tab = explain.explain_day(bundle, f, g["on_peak"].to_numpy(), background, cfg)
            print("\n" + explain.format_drivers(d, tab))
    if args.plot:
        actual = tables["da_smp"].set_index("ts_begin")["smp"]
        pred = pd.DataFrame({"ts_begin": out["ts_begin"], "actual": actual.reindex(out["ts_begin"]).to_numpy(),
                             "lgbm_linear": out["da_price_forecast"].to_numpy()})
        pdir = plots.plots_dir(cfg, "forecast")
        for d in sorted(pred["ts_begin"].dt.tz_convert(cfg["timezone"]).dt.tz_localize(None).dt.normalize().unique()):
            print("plot:", plots.day_plot(pred, d, cfg, "lgbm_linear", outdir=pdir))
    print(f"\nelapsed {time.time() - t0:.1f}s")


def cmd_evaluate(args, cfg):
    from pjmda.backtest import evaluate
    from pjmda.clean import clean_all
    clean_all(cfg, raw_dir=args.raw_dir, only_changed=True)
    evaluate(cfg, args.start, args.end, model_path=args.model, forecast_file=args.forecast,
             run_name=args.run_name, plots_flag=args.plots)


def cmd_plot(args, cfg):
    from pjmda import plots
    pred, run, main = plots.load_predictions(cfg, pred_path=args.pred, run=args.run, source=args.source)
    models = _csv_list(args.models) or [main]
    missing = [m for m in models if m not in pred.columns]
    if missing:
        raise SystemExit(f"models {missing} not in the predictions file (columns: "
                         f"{[c for c in pred.columns if c not in ('ts_begin', 'actual')]})")
    outdir = plots.plots_dir(cfg, run)
    if args.day:
        print(plots.day_plot(pred, args.day, cfg, main, models, outdir))
    elif args.month:
        print(plots.month_plot(pred, args.month, cfg, main, models, outdir))
    elif args.all_months:
        print(plots.months_overview(pred, cfg, main, models, outdir))
    else:
        raise SystemExit("give --day DATE, --month YYYY-MM or --all-months")


def cmd_explain(args, cfg):
    from pjmda.explain import run_explain
    run_explain(cfg, args.start, args.end, model_path=args.model)


def cmd_run_all(args, cfg):
    from pjmda import plots
    from pjmda.backtest import run_backtest
    from pjmda.clean import clean_all, read_clean
    from pjmda.model import train_and_save
    clean_all(cfg)
    cmd_features(argparse.Namespace(start=None, end=None), cfg)
    run = cfg["backtest"]["runs"]["wf2025"]
    pred = run_backtest(cfg, run["start"], run["end"], run_name="wf2025")
    print("2025 all-months overview:", plots.plots_dir(cfg, "wf2025") / "months_overview.png")
    paths = train_and_save(cfg, read_clean(cfg), cfg["train"]["end"])
    for name, p in paths.items():
        print(f"{name}: {p}")
    return pred


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pjmda", description="PJM day-ahead SMP forecaster")
    ap.add_argument("--config", help="path to config.yaml (default: project config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("clean", help="parse and clean all raw files -> data/clean/")
    p.add_argument("--raw-dir")
    p.set_defaults(fn=cmd_clean)

    p = sub.add_parser("features", help="build the feature matrix -> data/features/")
    p.add_argument("--start")
    p.add_argument("--end")
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("tune", help="grid search inside the training data -> models/params.json")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("backtest", help="walk-forward backtest")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--train-start")
    p.add_argument("--retrain-every", type=int)
    p.add_argument("--models", help="comma list: lgbm_linear,gbm_const,lear,naive_week,naive_day")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--run-name")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("train", help="fit and save the frozen models")
    p.add_argument("--until", required=True)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("forecast", help="forecast one day or a date range")
    p.add_argument("--date")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--raw-dir")
    p.add_argument("--model")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("evaluate", help="score the frozen model (or a forecast CSV) on supplied actuals")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--raw-dir")
    p.add_argument("--model")
    p.add_argument("--forecast")
    p.add_argument("--run-name")
    p.add_argument("--plots", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("plot", help="forecast-vs-actual plots from a predictions file")
    p.add_argument("--pred")
    p.add_argument("--run")
    p.add_argument("--source", choices=["backtest", "test"], default="backtest")
    p.add_argument("--day")
    p.add_argument("--month")
    p.add_argument("--all-months", action="store_true")
    p.add_argument("--models")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("explain", help="SHAP, permutation importance, ALE, error breakdown")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--model")
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("run-all", help="clean -> features -> backtest 2025 -> train")
    p.set_defaults(fn=cmd_run_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    from pjmda.features import MissingInputError
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        args.fn(args, cfg)
    except MissingInputError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
