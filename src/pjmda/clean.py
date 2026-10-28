"""One loader/cleaner per raw dataset -> tidy parquet in data/clean/.

Every clean table gets `available_at` (tz-aware EPT): when the value was first
knowable. Hourly tables also get `ts_begin` (tz-aware hour beginning),
`delivery_day` (naive local date) and `hour` (1-24 wall-clock hour ending).
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pjmda.config import dir_path, resolve
from pjmda.vintage import label_to_ts, localize, ns

log = logging.getLogger(__name__)

DATASETS = [
    "da_prices", "gas_prices", "gen_outages",
    "load_forecast", "solar_forecast", "wind_forecast", "gen_by_fuel",
]
OUTPUT_FILES = {
    "da_prices": ["da_smp.parquet", "da_nodes.parquet"],
    "gas_prices": ["gas.parquet"],
    "gen_outages": ["outages.parquet"],
    "load_forecast": ["load.parquet"],
    "solar_forecast": ["solar.parquet"],
    "wind_forecast": ["wind.parquet"],
    "gen_by_fuel": ["gen_fuel.parquet"],
}
META_FILE = "clean_meta.json"


# ------------------------------------------------------------------ file utils
def dataset_files(cfg: dict, dataset: str, raw_dir: str | Path | None = None) -> list[Path]:
    """All raw files matching the dataset glob, oldest modification time first."""
    base = resolve(raw_dir) if raw_dir else dir_path(cfg, "raw_dir")
    pattern = cfg["raw_files"][dataset]
    pattern = pattern if Path(pattern).is_absolute() else str(base / pattern)
    files = [Path(p) for p in glob.glob(pattern)]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def _read_csvs(files: list[Path], **kw) -> pd.DataFrame:
    """Concatenate all files; `_mtime` ranks files so the newest wins on conflicts."""
    parts = []
    for f in files:
        part = pd.read_csv(f, **kw)
        part["_mtime"] = f.stat().st_mtime
        log.info("  read %s: %d rows", f.name, len(part))
        parts.append(part)
    if not parts:
        raise FileNotFoundError("no raw files found")
    return pd.concat(parts, ignore_index=True)


def _dedupe(df: pd.DataFrame, key: list[str], name: str) -> pd.DataFrame:
    """Drop rows repeated across (or within) files; newest file wins, warn on conflicts."""
    df = df.sort_values("_mtime", kind="stable")
    keep = ~df.duplicated(key, keep="last")
    n_dup = int((~keep).sum())
    if n_dup:
        old, new = df.loc[~keep], df.loc[keep]
        value_cols = [c for c in df.columns if c not in key and c != "_mtime"]
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        differ = np.zeros(len(m), dtype=bool)
        for c in value_cols:
            a, b = m[f"{c}_old"], m[f"{c}_new"]
            differ |= ~((a == b) | (a.isna() & b.isna())).to_numpy()
        if differ.any():
            log.warning("%s: %d duplicated keys with DIFFERENT values; kept the most "
                        "recently modified file", name, int(differ.sum()))
        else:
            log.info("%s: %d exact duplicate rows removed", name, n_dup)
    return df.loc[keep].drop(columns="_mtime").reset_index(drop=True)


def _coverage(name: str, df: pd.DataFrame, day_col: str) -> None:
    nan = {c: int(v) for c, v in df.isna().sum().items() if v}
    log.info("%s: %d rows, %s -> %s, NaN counts: %s", name, len(df),
             pd.Timestamp(df[day_col].min()).date(), pd.Timestamp(df[day_col].max()).date(), nan or "none")


def _hourly_index(df: pd.DataFrame, day_col: str, label_col: str, convention: str,
                  tz: str, name: str) -> pd.DataFrame:
    """Add ts_begin / delivery_day / hour; drop labels that cannot be mapped."""
    day = ns(pd.to_datetime(df[day_col]))
    m = label_to_ts(day, df[label_col], convention, tz)
    df = df.reset_index(drop=True)
    df["delivery_day"] = day.reset_index(drop=True)
    df["ts_begin"] = m["ts_begin"]
    df["hour"] = m["hour"]
    bad = ~m["valid"].to_numpy()
    if bad.any():
        log.info("%s: %d rows dropped: hour label maps outside the delivery day or to a "
                 "non-existent local hour (%s)", name, int(bad.sum()),
                 sorted(df.loc[bad, label_col].unique().tolist())[:6])
    return df.loc[~bad]


# ----------------------------------------------------------------- 2.1 DA prices
def clean_da(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz, c = cfg["timezone"], cfg["da"]
    cols = ["opr_date", "hour_ending", "pnode_id", "pnode_name", "congestion_price_da",
            "marginal_loss_price_da", "system_energy_price_da"]
    df = _read_csvs(files, usecols=cols, engine="pyarrow")
    n_in = len(df)
    df = _dedupe(df, ["opr_date", "hour_ending", "pnode_id"], "da_prices")
    df = _hourly_index(df, "opr_date", "hour_ending", "hour_ending", tz, "da_prices")

    # publication: results of day X are posted at publish_time on X - publish_days_before
    pub = df["delivery_day"] - pd.Timedelta(days=int(c["publish_days_before"]))
    hh, mm = (int(x) for x in c["publish_time"].split(":"))
    df["available_at"] = localize(pub + pd.Timedelta(hours=hh, minutes=mm), tz)

    # SMP is the same at every node: one value per hour (median guards against stray rows)
    g = df.groupby("ts_begin")["system_energy_price_da"]
    smp = pd.DataFrame({"smp": g.median(), "spread": g.max() - g.min()}).reset_index()
    if (smp["spread"] > 1e-6).any():
        log.warning("da_prices: SMP differs across nodes in %d hours (spread > 1e-6)",
                    int((smp["spread"] > 1e-6).sum()))
    smp = smp.drop(columns="spread")
    keys = df.drop_duplicates("ts_begin")[["ts_begin", "delivery_day", "hour", "available_at"]]
    smp = smp.merge(keys, on="ts_begin").sort_values("ts_begin")[
        ["ts_begin", "delivery_day", "hour", "available_at", "smp"]
    ].reset_index(drop=True)

    # nodes needed for features: two hubs + every zone aggregate; merchant lines excluded
    name = df["pnode_name"].astype(str)
    keep = name.isin([c["hub_west"], c["hub_east"]]) | name.str.endswith(c["zone_suffix"])
    keep &= ~name.isin(c["excluded_nodes"])
    nodes = df.loc[keep, ["ts_begin", "delivery_day", "hour", "available_at", "pnode_name",
                          "congestion_price_da", "marginal_loss_price_da"]].rename(
        columns={"pnode_name": "node", "congestion_price_da": "congestion",
                 "marginal_loss_price_da": "loss"})
    nodes["node"] = nodes["node"].astype(str)
    nodes = nodes.sort_values(["ts_begin", "node"]).reset_index(drop=True)

    log.info("da_prices: %d raw rows -> %d hourly SMP rows; %d node-hour rows kept (%d nodes); "
             "price spikes are kept (they are real)", n_in, len(smp), len(nodes), nodes["node"].nunique())
    per_day = smp.groupby("delivery_day").size()
    odd = per_day[per_day != 24]
    if len(odd):
        log.info("da_prices: %d days do not have 24 hours (DST days): %s", len(odd),
                 {str(k.date()): int(v) for k, v in odd.head(6).items()})
    _coverage("da_smp", smp, "delivery_day")
    return {"da_smp": smp, "da_nodes": nodes}


# --------------------------------------------------------------- 2.2 gas prices
def _parse_gas_lines(path: Path) -> tuple[list[tuple], int]:
    """The `description` field contains unquoted commas, so split from both ends."""
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line or line.startswith("flow_date"):
                continue
            f = line.split(",")
            while f and f[-1].strip() == "":          # trailing empty fields of varying count
                f.pop()
            if len(f) < 5:
                bad += 1
                continue
            rows.append((f[0].strip(), f[1].strip(), ",".join(f[2:-2]).strip(), f[-2].strip(), f[-1].strip()))
    return rows, bad


def clean_gas(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz, c = cfg["timezone"], cfg["gas"]
    parts = []
    for f in files:
        rows, bad = _parse_gas_lines(f)
        part = pd.DataFrame(rows, columns=["flow_date", "code", "description", "price", "report_date"])
        part["_mtime"] = f.stat().st_mtime
        log.info("  read %s: %d rows (%d malformed lines skipped)", f.name, len(part), bad)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    n_in = len(df)

    df["price_raw"] = df["price"]
    df["price"] = pd.to_numeric(df["price"].where(df["price"] != "(null)"), errors="coerce")
    unparsed = df["price"].isna() & ~df["price_raw"].isin(["(null)", ""])
    if unparsed.any():
        log.warning("gas: %d prices could not be parsed and were set to NaN", int(unparsed.sum()))
    df["flow_date"] = ns(pd.to_datetime(df["flow_date"], errors="coerce"))
    df["report_date"] = ns(pd.to_datetime(df["report_date"], errors="coerce"))
    bad_dates = df["flow_date"].isna() | df["report_date"].isna()
    if bad_dates.any():
        log.warning("gas: %d rows dropped: unparseable dates", int(bad_dates.sum()))
        df = df.loc[~bad_dates]

    df["stem"], df["suffix"] = df["code"].str[:5], df["code"].str[5:]
    stem_to_hub = {v: k for k, v in c["hubs"].items()}
    keep = df["stem"].isin(stem_to_hub)
    log.info("gas: %d rows from hubs outside the configured list dropped", int((~keep).sum()))
    df = df.loc[keep].copy()
    df["hub"] = df["stem"].map(stem_to_hub)

    # revisions stay as separate rows (different report_date); the vintage logic picks
    df = _dedupe(df[["flow_date", "code", "report_date", "price", "hub", "suffix", "_mtime"]],
                 ["flow_date", "code", "report_date"], "gas")
    df["available_at"] = localize(df["report_date"] + pd.Timedelta(days=int(c["publish_lag_days"])), tz)

    late = (df["report_date"] - df["flow_date"]).dt.days
    log.info("gas: %d -> %d rows; %d null prices; %d rows with report_date after flow_date "
             "(late revisions / packages, excluded by the vintage rule when too late)",
             n_in, len(df), int(df["price"].isna().sum()), int((late > 0).sum()))
    df = df.sort_values(["hub", "suffix", "flow_date", "available_at"]).reset_index(drop=True)
    _coverage("gas", df, "flow_date")
    return {"gas": df}


# ---------------------------------------------------------------- 2.3 outages
def clean_outages(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz = cfg["timezone"]
    cols = ["posting_timestamp", "opr_date", "region", "total_outages", "planned_outages",
            "maintenance_outages", "forced_outages"]
    df = _read_csvs(files, usecols=cols)          # date_inserted is ignored
    n_in = len(df)
    df = _dedupe(df, ["posting_timestamp", "opr_date", "region"], "gen_outages")
    df["available_at"] = localize(pd.to_datetime(df["posting_timestamp"]), tz)
    df["delivery_day"] = ns(pd.to_datetime(df["opr_date"]))
    df["region"] = df["region"].astype(str)
    out = df[["delivery_day", "region", "available_at", "total_outages", "planned_outages",
              "maintenance_outages", "forced_outages"]].sort_values(
        ["region", "delivery_day", "available_at"]).reset_index(drop=True)
    log.info("gen_outages: %d -> %d rows, regions %s", n_in, len(out), sorted(out["region"].unique()))
    _coverage("outages", out, "delivery_day")
    return {"outages": out}


# -------------------------------------------------------------- 2.4-2.6 forecasts
def clean_load(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz = cfg["timezone"]
    regions = cfg["load"]["regions"]
    cols = ["forecast_timestamp", "opr_date", "opr_hour", "region_zone", "load_value"]
    df = _read_csvs(files, usecols=cols, engine="pyarrow")
    n_in = len(df)
    df = df[df["region_zone"].isin(list(regions.values()))]
    df = _dedupe(df, ["forecast_timestamp", "opr_date", "opr_hour", "region_zone"], "load_forecast")
    inv = {v: k for k, v in regions.items()}
    df["region"] = df["region_zone"].map(inv)
    df["available_at"] = localize(pd.to_datetime(df["forecast_timestamp"]), tz)
    df = _hourly_index(df, "opr_date", "opr_hour", "hour_ending", tz, "load_forecast")
    out = df.rename(columns={"load_value": "load_mw"})[
        ["ts_begin", "delivery_day", "hour", "available_at", "region", "load_mw"]
    ].sort_values(["region", "ts_begin", "available_at"]).reset_index(drop=True)
    log.info("load_forecast: %d raw rows -> %d rows for regions %s", n_in, len(out), sorted(out["region"].unique()))
    _coverage("load", out, "delivery_day")
    return {"load": out}


def clean_solar(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz, c = cfg["timezone"], cfg["solar"]
    cols = ["forecast_timestamp", "opr_date", "hour_ending", "region_zone", "forecast_value"]
    df = _read_csvs(files, usecols=cols)
    n_in = len(df)
    df = df[df["region_zone"] == c["region"]]
    df = _dedupe(df, ["forecast_timestamp", "opr_date", "hour_ending", "region_zone"], "solar_forecast")
    df["available_at"] = localize(pd.to_datetime(df["forecast_timestamp"]), tz)
    n_zero = int((df["hour_ending"] == 0).sum())
    if n_zero:
        log.info("solar_forecast: %d rows carry label 0 (the first vintages run 0-23). Convention "
                 "'%s' maps them outside the delivery day; those rows are dropped and the "
                 "missing night hour is filled by the short-gap rule in features.",
                 n_zero, c["hour_convention"])
    df = _hourly_index(df, "opr_date", "hour_ending", c["hour_convention"], tz, "solar_forecast")
    out = df.rename(columns={"forecast_value": "solar_mw"})[
        ["ts_begin", "delivery_day", "hour", "available_at", "solar_mw"]
    ].sort_values(["ts_begin", "available_at"]).reset_index(drop=True)
    log.info("solar_forecast: %d raw rows -> %d rows (hour convention %s)", n_in, len(out), c["hour_convention"])
    _coverage("solar", out, "delivery_day")
    return {"solar": out}


def clean_wind(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz = cfg["timezone"]
    cols = ["forecast_timestamp", "opr_date", "opr_hour", "wind_value"]
    df = _read_csvs(files, usecols=cols)
    n_in = len(df)
    df = _dedupe(df, ["forecast_timestamp", "opr_date", "opr_hour"], "wind_forecast")
    df["available_at"] = localize(pd.to_datetime(df["forecast_timestamp"]), tz)
    df = _hourly_index(df, "opr_date", "opr_hour", "hour_ending", tz, "wind_forecast")
    out = df.rename(columns={"wind_value": "wind_mw"})[
        ["ts_begin", "delivery_day", "hour", "available_at", "wind_mw"]
    ].sort_values(["ts_begin", "available_at"]).reset_index(drop=True)
    log.info("wind_forecast: %d raw rows -> %d rows", n_in, len(out))
    _coverage("wind", out, "delivery_day")
    return {"wind": out}


# ------------------------------------------------------------ 2.7 gen by fuel
def _slug(fuel: str) -> str:
    return fuel.strip().lower().replace(" ", "_")


def clean_gen_by_fuel(cfg: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    tz, c = cfg["timezone"], cfg["gen_by_fuel"]
    cols = ["datetime_beginning_utc", "fuel_type", "mw"]
    df = _read_csvs(files, usecols=cols)
    n_in = len(df)
    fmt = "%m/%d/%Y %I:%M:%S %p"
    try:
        utc = pd.to_datetime(df["datetime_beginning_utc"], format=fmt)
    except ValueError:
        log.warning("gen_by_fuel: timestamp format %s did not fit, falling back to inference", fmt)
        utc = pd.to_datetime(df["datetime_beginning_utc"])
    # UTC is unambiguous: it resolves the repeated fall-back hour
    df["ts_begin"] = utc.dt.tz_localize("UTC").dt.tz_convert(tz).dt.as_unit("ns")
    df = _dedupe(df[["ts_begin", "fuel_type", "mw", "_mtime"]], ["ts_begin", "fuel_type"], "gen_by_fuel")

    wide = df.pivot(index="ts_begin", columns="fuel_type", values="mw")
    core = [f for f in c["core_fuels"] if f in wide.columns]
    missing_core = [f for f in c["core_fuels"] if f not in wide.columns]
    if missing_core:
        log.warning("gen_by_fuel: core fuels absent from the file: %s", missing_core)
    gen_total = wide.sum(axis=1, min_count=1)
    incomplete = wide[core].isna().any(axis=1)          # a core fuel missing -> total would be understated
    gen_total[incomplete] = np.nan
    wide.columns = [_slug(x) for x in wide.columns]
    wide["gen_total"] = gen_total
    wide = wide.reset_index().sort_values("ts_begin").reset_index(drop=True)
    wide["delivery_day"] = ns(wide["ts_begin"].dt.tz_localize(None).dt.normalize())
    wide["hour"] = wide["ts_begin"].dt.hour + 1
    # no publication time in the file: conservative rule
    wide["available_at"] = wide["ts_begin"] + pd.Timedelta(hours=1 + int(c["publish_lag_hours"]))
    first = ["ts_begin", "delivery_day", "hour", "available_at"]
    wide = wide[first + [x for x in wide.columns if x not in first]]

    gaps = wide["ts_begin"].diff().dt.total_seconds().div(3600)
    log.info("gen_by_fuel: %d raw rows -> %d hourly rows; %d hours with an incomplete fuel set "
             "(total set to NaN); %d gaps of >1 hour in the hourly series; storage may be negative (not clipped)",
             n_in, len(wide), int(incomplete.sum()), int((gaps > 1).sum()))
    _coverage("gen_fuel", wide, "delivery_day")
    return {"gen_fuel": wide}


CLEANERS = {
    "da_prices": clean_da,
    "gas_prices": clean_gas,
    "gen_outages": clean_outages,
    "load_forecast": clean_load,
    "solar_forecast": clean_solar,
    "wind_forecast": clean_wind,
    "gen_by_fuel": clean_gen_by_fuel,
}
FIRST_DAY_COL = {"da_smp": "delivery_day", "da_nodes": "delivery_day", "gas": "flow_date",
                 "outages": "delivery_day",
                 "load": "delivery_day", "solar": "delivery_day", "wind": "delivery_day",
                 "gen_fuel": "delivery_day"}


# ------------------------------------------------------------------- driver
def _signature(files: list[Path]) -> list[dict]:
    return [{"file": str(f), "mtime": f.stat().st_mtime, "size": f.stat().st_size} for f in files]


def _read_meta(clean_dir: Path) -> dict:
    p = clean_dir / META_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def clean_all(cfg: dict, raw_dir: str | Path | None = None, only_changed: bool = False) -> dict:
    """Clean every dataset. With only_changed, skip datasets whose files are unchanged (mtime/size)."""
    clean_dir = dir_path(cfg, "clean_dir", create=True)
    meta = _read_meta(clean_dir)
    for ds in DATASETS:
        files = dataset_files(cfg, ds, raw_dir)
        if not files:
            raise FileNotFoundError(f"no raw files match {cfg['raw_files'][ds]!r} for dataset {ds}")
        sig = _signature(files)
        outputs_ok = all((clean_dir / o).exists() for o in OUTPUT_FILES[ds])
        if only_changed and outputs_ok and meta.get(ds, {}).get("files") == sig:
            log.info("[%s] unchanged, skipping re-clean", ds)
            continue
        log.info("[%s] cleaning %d file(s): %s", ds, len(files), [f.name for f in files])
        tables = CLEANERS[ds](cfg, files)
        info = {"files": sig, "tables": {}}
        for name, tbl in tables.items():
            tbl.to_parquet(clean_dir / f"{name}.parquet", index=False)
            col = FIRST_DAY_COL[name]
            info["tables"][name] = {
                "rows": int(len(tbl)),
                "first_day": str(pd.Timestamp(tbl[col].min()).date()),
                "last_day": str(pd.Timestamp(tbl[col].max()).date()),
            }
        meta[ds] = info
        (clean_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    start = effective_start(cfg)
    log.info("EFFECTIVE TRAINING START: %s (latest dataset start + %d warm-up days, not before train.start)",
             start.date(), cfg["train"]["warmup_days"])
    return meta


def read_clean(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load all clean parquet tables."""
    clean_dir = dir_path(cfg, "clean_dir")
    out = {}
    for ds in DATASETS:
        for fname in OUTPUT_FILES[ds]:
            p = clean_dir / fname
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run `python -m pjmda clean` first")
            out[Path(fname).stem] = pd.read_parquet(p)
    return out


def effective_start(cfg: dict) -> pd.Timestamp:
    """Latest of the datasets' first days + warm-up days, never before train.start."""
    meta = _read_meta(dir_path(cfg, "clean_dir"))
    firsts = []
    for ds in meta.values():
        firsts += [pd.Timestamp(t["first_day"]) for t in ds["tables"].values()]
    if not firsts:
        raise FileNotFoundError("no clean data: run `python -m pjmda clean` first")
    start = max(firsts) + pd.Timedelta(days=int(cfg["train"]["warmup_days"]))
    return max(start, pd.Timestamp(cfg["train"]["start"]))
