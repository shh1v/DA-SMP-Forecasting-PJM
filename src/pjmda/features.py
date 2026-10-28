"""Helper variables and the 19 model features (pure functions).

Every input is chosen with `asof_select(..., cutoff(D))`, so a feature for delivery
day D only sees information that was available before the D-1 10:30 EPT cut-off.
HELPER variables are never passed to the model; only the 19 MODEL features are.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from pjmda.vintage import (
    asof_select, cutoff_for_day, date_range, delivery_hours, ns, stale_flag,
)

log = logging.getLogger(__name__)

HOURS = np.arange(1, 25)
DAY = pd.Timedelta(days=1)


class MissingInputError(RuntimeError):
    """Raised when a forecast cannot be built because required inputs are absent."""


# ----------------------------------------------------------------- feature lists
def feature_names(cfg: dict) -> list[str]:
    """The 19 model features in config order."""
    return [f for g in cfg["features"]["groups"].values() for f in g]


def feature_group(cfg: dict, name: str) -> str:
    for g, members in cfg["features"]["groups"].items():
        if name in members:
            return g
    raise KeyError(name)


# ------------------------------------------------------------------- calendar
def nerc_holidays(years) -> set[pd.Timestamp]:
    """NERC holidays with Sunday -> Monday observance."""
    out: set[pd.Timestamp] = set()
    for y in years:
        fixed = [pd.Timestamp(date(y, 1, 1)), pd.Timestamp(date(y, 7, 4)), pd.Timestamp(date(y, 12, 25))]
        fixed = [d + DAY if d.dayofweek == 6 else d for d in fixed]          # Sunday -> Monday
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31")
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30")
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        memorial = may[may.dayofweek == 0][-1]        # last Monday of May
        labor = sep[sep.dayofweek == 0][0]            # first Monday of September
        thanksgiving = nov[nov.dayofweek == 3][3]     # fourth Thursday of November
        out.update(fixed + [memorial, labor, thanksgiving])
    return out


def is_holiday(days: pd.Series) -> pd.Series:
    years = range(int(days.dt.year.min()) - 1, int(days.dt.year.max()) + 2)
    hol = nerc_holidays(years)
    return days.isin(list(hol)).astype(int)


# --------------------------------------------------- hourly forecasts (load/wind/solar)
def _long(M: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "delivery_day": ns(pd.Series(np.repeat(M.index.to_numpy(), len(HOURS)))),
        "hour": np.tile(HOURS, len(M)),
        name: M.to_numpy().ravel(),
    })


def _matrix(sel: pd.DataFrame, col: str, ext_days: pd.DatetimeIndex) -> pd.DataFrame:
    """(delivery_day x hour) matrix of a selected vintage column."""
    m = sel.pivot(index="delivery_day", columns="hour", values=col)
    return m.reindex(index=ext_days, columns=HOURS)


def _fill_gaps(M: pd.DataFrame, exists: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Short gaps inside a day's forecast are interpolated; hours with no vintage at all
    are copied from the same hour 7 / 14 days earlier (their own vintage), flagged stale."""
    v = cfg["vintage"]
    lim = int(v["max_fill_gap_hours"])
    M0 = M.where(exists)
    M1 = M0.interpolate(axis=1, limit=lim, limit_area="inside").ffill(axis=1, limit=lim)
    filled = M0.isna() & M1.notna() & exists
    base = M1.copy()
    stale = pd.DataFrame(False, index=M.index, columns=M.columns)
    for k in range(1, int(v["fallback_max_lags"]) + 1):
        lag = base.copy()
        lag.index = lag.index + pd.Timedelta(days=int(v["fallback_lag_days"]) * k)
        lag = lag.reindex(M1.index)
        m = M1.isna() & lag.notna() & exists
        M1 = M1.where(~m, lag)
        stale |= m
    return M1.where(exists), filled, stale


def _fill_by_identity(mats: dict[str, pd.DataFrame], total: str, parts: list[str]
                      ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """total = sum(parts) holds in the PJM load data (relative residual < 0.2%): when exactly one
    member of the identity is missing for an hour, rebuild it from the others (same vintage)."""
    names = [total, *parts]
    out = {k: v.copy() for k, v in mats.items()}
    filled = pd.DataFrame(False, index=mats[total].index, columns=mats[total].columns)
    for miss in names:
        others = [n for n in names if n != miss]
        ok = mats[others[0]].notna()
        for o in others[1:]:
            ok = ok & mats[o].notna()
        cond = mats[miss].isna() & ok
        if miss == total:
            val = sum(mats[p] for p in parts)
        else:
            val = mats[total] - sum(mats[p] for p in parts if p != miss)
        out[miss] = mats[miss].where(~cond, val)
        filled = filled | cond
    return out, filled


def hourly_forecast(tbl: pd.DataFrame, value_col: str, group_col: str | None, source: str,
                    ext_days: pd.DatetimeIndex, grid: pd.DataFrame, cfg: dict,
                    identity: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest vintage before cutoff(delivery day) for every target hour.

    Returns (values, flags): values has one column per group (or value_col),
    flags has stale_<source> and filled_<source>; both are indexed by the grid rows.
    `identity` ({"total": g, "parts": [g, ...]}) rebuilds one missing group from the others.
    """
    t = tbl[tbl["delivery_day"].isin(ext_days)].reset_index(drop=True)
    cut = cutoff_for_day(t["delivery_day"], cfg)
    keys = ["delivery_day", "hour"] + ([group_col] if group_col else [])
    sel = asof_select(t, keys, cut).copy()
    thr = float(cfg["vintage"]["stale_after_hours"][source])
    sel["_stale"] = stale_flag(sel["available_at"], cutoff_for_day(sel["delivery_day"], cfg), thr)

    exists = grid.assign(v=1.0).pivot(index="delivery_day", columns="hour", values="v")
    exists = exists.reindex(index=ext_days, columns=HOURS).notna()
    cols = sorted(sel[group_col].unique()) if group_col else [value_col]
    age_stale = _matrix(sel.groupby(["delivery_day", "hour"], as_index=False)["_stale"].max(),
                        "_stale", ext_days).fillna(0).astype(bool)

    raw = {c: _matrix(sel[sel[group_col] == c] if group_col else sel, value_col, ext_days) for c in cols}
    id_filled = None
    if identity and group_col and all(n in raw for n in [identity["total"], *identity["parts"]]):
        raw, id_filled = _fill_by_identity(raw, identity["total"], identity["parts"])
        id_filled = id_filled & exists
        if id_filled.any().any():
            log.info("%s: %d hour(s) of a missing region rebuilt from %s = sum(%s) (same vintage, flagged filled_%s)",
                     source, int(id_filled.to_numpy().sum()), identity["total"], identity["parts"], source)

    vals, any_filled, any_stale = None, None, age_stale.copy()
    for c in cols:
        M, filled, stale = _fill_gaps(raw[c], exists, cfg)
        long = _long(M, c)
        vals = long if vals is None else vals.merge(long, on=["delivery_day", "hour"])
        any_filled = filled if any_filled is None else (any_filled | filled)
        any_stale = any_stale | stale
    if id_filled is not None:
        any_filled = any_filled | id_filled
    flags = _long(any_stale.astype(int), f"stale_{source}").merge(
        _long(any_filled.astype(int), f"filled_{source}"), on=["delivery_day", "hour"])
    vals = grid[["delivery_day", "hour"]].merge(vals, on=["delivery_day", "hour"], how="left")
    flags = grid[["delivery_day", "hour"]].merge(flags, on=["delivery_day", "hour"], how="left")
    return vals, flags


def prev_last_hour(tbl: pd.DataFrame, value_col: str, group_col: str | None, days: pd.DatetimeIndex,
                   cfg: dict) -> pd.DataFrame:
    """Value of hour 24 of D-1 chosen with cutoff(D) (not cutoff(D-1)). Indexed by D."""
    t = tbl[tbl["delivery_day"].isin(days - DAY) & (tbl["hour"] == 24)].copy()
    t["for_day"] = t["delivery_day"] + DAY
    t = t.reset_index(drop=True)
    cut = cutoff_for_day(t["for_day"], cfg)
    keys = ["for_day"] + ([group_col] if group_col else [])
    sel = asof_select(t, keys, cut)
    if group_col:
        return sel.pivot(index="for_day", columns=group_col, values=value_col)
    return sel.set_index("for_day")[[value_col]]


# ---------------------------------------------------------------------- outages
def daily_outages(tbl: pd.DataFrame, ext_days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """out_total for each day (PJM RTO total outages, MW) with the vintage rule."""
    o = cfg["outages"]
    t = tbl[(tbl["region"] == o["region"]) & tbl["delivery_day"].isin(ext_days)].reset_index(drop=True)
    sel = asof_select(t, ["delivery_day"], cutoff_for_day(t["delivery_day"], cfg)).set_index("delivery_day")
    s = sel["total_outages"].reindex(ext_days)
    age_stale = pd.Series(
        stale_flag(sel["available_at"], cutoff_for_day(sel.index.to_series(), cfg),
                   float(cfg["vintage"]["stale_after_hours"]["outages"])).to_numpy(),
        index=sel.index).reindex(ext_days).fillna(0).astype(bool)
    filled = s.ffill(limit=int(o["max_ffill_days"]))
    stale = age_stale | (s.isna() & filled.notna())
    if (s.isna() & filled.notna()).any():
        log.warning("outages: no posting before the cut-off for %d day(s), reused the latest earlier "
                    "posting (stale_outages=1)", int((s.isna() & filled.notna()).sum()))
    return pd.DataFrame({"out_total": filled, "stale_outages": stale.astype(int)})


# -------------------------------------------------------------------------- gas
def daily_gas(gas: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Hub prices usable for delivery day D: the latest flow date <= D-1 whose index was
    published before cutoff(D) (report_date <= D-2). Forward-filled up to max_ffill_days."""
    c = cfg["gas"]
    g = gas[(gas["suffix"] == c["suffix"]) & gas["price"].notna()]
    cut = cutoff_for_day(pd.Series(days), cfg).dt.tz_convert("UTC").dt.tz_localize(None)
    cut_i = cut.to_numpy(dtype="datetime64[ns]").astype("int64")
    day_i = days.to_numpy(dtype="datetime64[D]").astype("int64")
    out = pd.DataFrame(index=days)
    filled_any = np.zeros(len(days), dtype=bool)
    for hub in c["hubs"]:
        h = g[g["hub"] == hub].sort_values(["flow_date", "available_at"])
        flow_i = h["flow_date"].to_numpy(dtype="datetime64[D]").astype("int64")
        av_i = h["available_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
            dtype="datetime64[ns]").astype("int64")
        price = h["price"].to_numpy()
        res = np.full(len(days), np.nan)
        for j in range(len(days)):
            idx = np.flatnonzero((av_i < cut_i[j]) & (flow_i <= day_i[j] - 1))
            if not len(idx):
                continue
            last = idx[-1]                                   # max flow date, newest revision
            age = day_i[j] - 1 - flow_i[last]
            if age > int(c["max_ffill_days"]):
                continue
            res[j] = price[last]
            filled_any[j] |= age > 0
        out[hub] = res
    out["gas_filled"] = filled_any.astype(int)
    if filled_any.any():
        log.info("gas: forward-filled at least one hub on %d day(s), e.g. %s (gas_filled=1)",
                 int(filled_any.sum()), [str(d.date()) for d in days[filled_any][:5]])
    return out


def regional_gas(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """g_midatl, g_south, g_west: mean of whichever hubs are present."""
    out = pd.DataFrame(index=daily.index)
    for region, hubs in cfg["gas"]["regions"].items():
        out[f"g_{region}"] = daily[hubs].mean(axis=1, skipna=True)
    return out


# ----------------------------------------------------- generation by fuel actuals
def _to_ns_int(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")


def gen_features(gen: pd.DataFrame, outages: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """C_ref, cref_warmup and coal_share_7 for each delivery day, from lagged actuals only."""
    cap_cfg, gcfg = cfg["capacity"], cfg["gen_by_fuel"]
    # outages of the operating day of every generation hour (latest posting per day)
    o = outages[outages["region"] == cfg["outages"]["region"]].sort_values("available_at")
    o = o.drop_duplicates("delivery_day", keep="last").set_index("delivery_day")
    g = gen.sort_values("ts_begin").reset_index(drop=True)
    out_total = g["delivery_day"].map(o["total_outages"])
    out_avail = g["delivery_day"].map(o["available_at"])
    avail = g["available_at"].where(out_avail.isna() | (g["available_at"] >= out_avail), out_avail)
    cap = (g["gen_total"] + out_total).to_numpy(dtype=float)           # lower bound of installed capacity
    ts_i, av_i = _to_ns_int(g["ts_begin"]), _to_ns_int(avail.fillna(g["available_at"]))
    coal = g["coal"].to_numpy(dtype=float)
    gas_mw = g["gas"].to_numpy(dtype=float)
    cut_i = _to_ns_int(cutoff_for_day(pd.Series(days), cfg))
    window = np.int64(cap_cfg["window_days"]) * np.int64(86_400 * 10**9)
    n = int(gcfg["coal_window_hours"])
    c_ref, warm, coal_share = np.full(len(days), np.nan), np.zeros(len(days), dtype=int), np.full(len(days), np.nan)
    ok = ~np.isnan(cap)
    for j, cut in enumerate(cut_i):
        avail_m = av_i < cut
        m = avail_m & ok & (ts_i >= cut - window)
        if m.any():
            c_ref[j] = cap[m].max()
        first = ts_i[avail_m].min() if avail_m.any() else cut
        warm[j] = int(cut - first < window)                            # < window_days of history
    # coal share over the last n available hours (available_at is monotone in ts_begin)
    gav = _to_ns_int(g["available_at"])
    for j, cut in enumerate(cut_i):
        idx = np.flatnonzero(gav < cut)[-n:]
        idx = idx[~np.isnan(coal[idx]) & ~np.isnan(gas_mw[idx])]
        if len(idx) >= int(gcfg["coal_min_hours"]):
            tot = coal[idx].sum() + gas_mw[idx].sum()
            coal_share[j] = coal[idx].sum() / tot if tot > 0 else np.nan
    return pd.DataFrame({"C_ref": c_ref, "cref_warmup": warm, "coal_share_7": coal_share}, index=days)


# --------------------------------------------------------------------- DA prices
def smp_lag(smp: pd.DataFrame, days: pd.DatetimeIndex, k: int, cfg: dict) -> pd.DataFrame:
    """SMP(D-k, h) for each D, kept only if the result was posted before cutoff(D)."""
    t = smp[smp["delivery_day"].isin(days - k * DAY)].copy()
    t["for_day"] = t["delivery_day"] + k * DAY
    t = t.reset_index(drop=True)
    sel = asof_select(t, ["for_day", "hour"], cutoff_for_day(t["for_day"], cfg))
    return sel.drop(columns="delivery_day").rename(columns={"for_day": "delivery_day"})[
        ["delivery_day", "hour", "smp"]]


def grid_lag(nodes: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """C and M of D-1 (same hour), vintage-checked: C_east, C_west, M_east, M_west, C_zone_absmean."""
    d = cfg["da"]
    t = nodes[nodes["delivery_day"].isin(days - DAY)].copy()
    t["for_day"] = t["delivery_day"] + DAY
    t = t.reset_index(drop=True)
    sel = asof_select(t, ["for_day", "hour", "node"], cutoff_for_day(t["for_day"], cfg))
    keys = ["for_day", "hour"]
    C = sel.pivot(index=keys, columns="node", values="congestion")
    M = sel.pivot(index=keys, columns="node", values="loss")
    zones = [c for c in C.columns if str(c).endswith(d["zone_suffix"])]
    out = pd.DataFrame({
        "C_east": C.get(d["hub_east"]), "C_west": C.get(d["hub_west"]),
        "M_east": M.get(d["hub_east"]), "M_west": M.get(d["hub_west"]),
        "C_zone_absmean": C[zones].abs().mean(axis=1, skipna=True),
    })
    return out.reset_index().rename(columns={"for_day": "delivery_day"})


def missing_price_days(days, smp: pd.DataFrame, cfg: dict) -> dict[pd.Timestamp, list[pd.Timestamp]]:
    """For each requested day D: the days among D-1 and D-7 whose DA results are missing or
    incomplete under the vintage rule (fewer than da.min_hours_per_day usable hours)."""
    days = pd.DatetimeIndex(days)
    need = {}
    for k in (1, 7):
        lag = smp_lag(smp, days, k, cfg)
        counts = lag.groupby("delivery_day").size().reindex(days).fillna(0)
        for d, n in counts.items():
            if n < int(cfg["da"]["min_hours_per_day"]):
                need.setdefault(d, []).append(d - k * DAY)
    return need


# ----------------------------------------------------------------------- builder
def build_features(days, tables: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model features (19) and helper variables for every hour of the given delivery days."""
    tz = cfg["timezone"]
    days = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
    v = cfg["vintage"]
    pad = int(v["fallback_lag_days"]) * int(v["fallback_max_lags"]) + 1
    ext_days = date_range(days.min() - pad * DAY, days.max())
    ext_days = pd.DatetimeIndex(ext_days.to_numpy(dtype="datetime64[ns]"))
    days = pd.DatetimeIndex(days.to_numpy(dtype="datetime64[ns]"))
    grid_ext = delivery_hours(ext_days, tz)
    regions = cfg["load"]["regions"]

    # ---- hourly load / wind / solar under their own-day cut-off ----
    load, f_load = hourly_forecast(tables["load"], "load_mw", "region", "load", ext_days, grid_ext, cfg,
                                   identity=cfg["load"].get("identity"))
    wind, f_wind = hourly_forecast(tables["wind"], "wind_mw", None, "wind", ext_days, grid_ext, cfg)
    solar, f_solar = hourly_forecast(tables["solar"], "solar_mw", None, "solar", ext_days, grid_ext, cfg)
    hf = grid_ext.merge(load.rename(columns={r: f"L_{r}" for r in regions}), on=["delivery_day", "hour"])
    hf = hf.merge(wind.rename(columns={"wind_mw": "W"}), on=["delivery_day", "hour"])
    hf = hf.merge(solar.rename(columns={"solar_mw": "S"}), on=["delivery_day", "hour"])
    for f in (f_load, f_wind, f_solar):
        hf = hf.merge(f, on=["delivery_day", "hour"])
    hf["nl_own"] = hf["L_rto"] - hf["W"] - hf["S"]            # nl chosen at the day's own cut-off

    # ---- hour 24 of D-1 chosen with cutoff(D) (for ramp in hour 1) ----
    pl = prev_last_hour(tables["load"], "load_mw", "region", days, cfg)
    pw = prev_last_hour(tables["wind"], "wind_mw", None, days, cfg)["wind_mw"]
    ps = prev_last_hour(tables["solar"], "solar_mw", None, days, cfg)["solar_mw"]
    own24 = hf[hf["hour"] == 24].assign(for_day=lambda x: x["delivery_day"] + DAY).set_index("for_day")
    prev_nl = (pl["rto"].combine_first(own24["L_rto"]) - pw.combine_first(own24["W"])
               - ps.combine_first(own24["S"])).reindex(days)

    # ---- daily series ----
    out_d = daily_outages(tables["outages"], ext_days, cfg)
    gas_d = daily_gas(tables["gas"], days, cfg)
    reg_g = regional_gas(gas_d, cfg)
    gen_d = gen_features(tables["gen_fuel"], tables["outages"], days, cfg)

    # ---- assemble on the requested days ----
    grid = delivery_hours(days, tz)
    h = grid.merge(hf.drop(columns="ts_begin"), on=["delivery_day", "hour"], how="left")
    daily_mean = hf.groupby("delivery_day")[[f"L_{r}" for r in regions]].mean()
    w = pd.DataFrame({f"w_{r}": daily_mean[f"L_{r}"] / daily_mean["L_rto"] for r in ("midatl", "south", "west")})
    d = pd.concat([reg_g, w.reindex(days), out_d.reindex(days), gen_d, gas_d.add_prefix("gas_")], axis=1)
    d = d.rename(columns={"gas_gas_filled": "gas_filled"})
    h = h.merge(d.rename_axis("delivery_day").reset_index(), on="delivery_day", how="left")

    # g_comp: load-weighted regional gas; renormalise over regions present
    G = h[["g_midatl", "g_south", "g_west"]].to_numpy()
    Wt = h[["w_midatl", "w_south", "w_west"]].to_numpy()
    valid = ~np.isnan(G) & ~np.isnan(Wt)
    num = np.where(valid, G * Wt, 0.0).sum(axis=1)
    den = np.where(valid, Wt, 0.0).sum(axis=1)
    h["g_comp"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    h["gas_basis_ew"] = h["g_midatl"] - h["g_west"]

    # balance
    h["nl"] = h["L_rto"] - h["W"] - h["S"]
    h["tight"] = h["nl"] / (h["C_ref"] - h["out_total"])
    h["ren_share"] = (h["W"] + h["S"]) / h["L_rto"]
    h["s_east"] = (h["L_midatl"] + h["L_south"]) / h["L_rto"]
    h = h.sort_values("ts_begin").reset_index(drop=True)
    prev_nl_in_day = h.groupby("delivery_day")["nl"].shift(1)
    first_hour = h.groupby("delivery_day")["hour"].transform("min") == h["hour"]
    h["ramp"] = h["nl"] - np.where(first_hour, h["delivery_day"].map(prev_nl), prev_nl_in_day)
    h["nl_max"] = h.groupby("delivery_day")["nl"].transform("max")
    prev_own = hf.set_index(["delivery_day", "hour"])["nl_own"]
    key_prev = pd.MultiIndex.from_arrays([h["delivery_day"] - DAY, h["hour"]])
    h["dnl"] = h["nl"] - prev_own.reindex(key_prev).to_numpy()

    # grid constraints and price history (vintage checked)
    s1 = smp_lag(tables["da_smp"], days, 1, cfg).rename(columns={"smp": "smp_lag1"})
    s7 = smp_lag(tables["da_smp"], days, 7, cfg).rename(columns={"smp": "smp_lag7"})
    gl = grid_lag(tables["da_nodes"], days, cfg)
    h = h.merge(s1, on=["delivery_day", "hour"], how="left").merge(s7, on=["delivery_day", "hour"], how="left")
    h = h.merge(gl, on=["delivery_day", "hour"], how="left")
    h["smp1_abs"] = h["smp_lag1"].abs().clip(lower=float(cfg["da"]["smp_floor_abs"]))
    h["ci_lag1"] = h["C_zone_absmean"] / h["smp1_abs"]
    h["spr_ew_lag1"] = (h["C_east"] - h["C_west"]) / h["smp1_abs"]
    h["mlf_ew_lag1"] = (h["M_east"] - h["M_west"]) / h["smp1_abs"]

    # calendar
    h["hour"] = h["hour"].astype(int)
    h["dow"] = h["delivery_day"].dt.dayofweek.astype(int)
    h["month"] = h["delivery_day"].dt.month.astype(int)
    h["is_holiday"] = is_holiday(h["delivery_day"])
    lo, hi = cfg["backtest"]["on_peak_hours"]
    h["on_peak"] = ((h["hour"] >= lo) & (h["hour"] <= hi) & (h["dow"] < 5) & (h["is_holiday"] == 0)).astype(int)

    names = feature_names(cfg)
    features = h[["ts_begin", "delivery_day"] + names].copy()
    helper_cols = [c for c in h.columns if c not in names and c not in ("delivery_day",)]
    helpers = h[helper_cols].copy()
    _summarise(features, helpers, cfg)
    return features, helpers


def _summarise(features: pd.DataFrame, helpers: pd.DataFrame, cfg: dict) -> None:
    names = feature_names(cfg)
    nan = {c: int(features[c].isna().sum()) for c in names if features[c].isna().any()}
    log.info("features: %d rows, %d days (%s -> %s); NaN counts: %s", len(features),
             features["delivery_day"].nunique(), features["delivery_day"].min().date(),
             features["delivery_day"].max().date(), nan or "none")
    flags = [c for c in helpers.columns if c.startswith(("stale_", "filled_", "gas_filled", "cref_warmup"))]
    log.info("flags (hours flagged): %s", {c: int(helpers[c].fillna(0).sum()) for c in flags})
    bad = features.loc[features["g_comp"].isna(), "delivery_day"].unique()
    if len(bad):
        log.error("g_comp is NaN (no regional gas or load weight) on %d day(s): %s", len(bad),
                  [str(pd.Timestamp(x).date()) for x in bad[:10]])


def check_forecastable(features: pd.DataFrame, cfg: dict) -> None:
    """Hard stop for forecast/evaluate when g_comp cannot be formed."""
    bad = features.loc[features["g_comp"].isna(), "delivery_day"].unique()
    if len(bad):
        raise MissingInputError(
            "g_comp is undefined (all regional gas prices or load weights missing) for: "
            + ", ".join(str(pd.Timestamp(x).date()) for x in bad))


# ------------------------------------------------------------------ LEAR inputs
def lear_inputs(days, tables: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Extra price inputs of the LEAR benchmark per (delivery_day, hour): SMP lags of
    the same hour that are not already model features (D-2, D-3) and all 24 hours of
    D-1 as smp_d1_h1..smp_d1_h24."""
    days = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
    days = pd.DatetimeIndex(days.to_numpy(dtype="datetime64[ns]"))
    base = delivery_hours(days, cfg["timezone"])[["delivery_day", "hour"]]
    for k in cfg["lear"]["price_lags"]:
        if f"smp_lag{k}" in feature_names(cfg):          # already a model feature column
            continue
        lag = smp_lag(tables["da_smp"], days, int(k), cfg).rename(columns={"smp": f"smp_lag{k}"})
        base = base.merge(lag, on=["delivery_day", "hour"], how="left")
    d1 = smp_lag(tables["da_smp"], days, 1, cfg)
    wide = d1.pivot(index="delivery_day", columns="hour", values="smp").reindex(columns=HOURS)
    wide.columns = [f"smp_d1_h{h}" for h in HOURS]
    return base.merge(wide.reset_index(), on="delivery_day", how="left")
