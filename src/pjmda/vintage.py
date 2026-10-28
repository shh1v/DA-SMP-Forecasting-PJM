"""Cut-off logic, local-time helpers and the generic as-of vintage selection.

Every clean table carries `available_at` (tz-aware EPT): when a value was first
knowable. Every feature is built from `asof_select(..., cutoff(D))`, which is the
single mechanism that enforces the information cut-off (no look-ahead).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------- time helpers
def ns(x):
    """Cast naive datetimes to nanosecond resolution so merges never mix units."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.astype("datetime64[ns]")
    return pd.Timestamp(x).as_unit("ns")


def localize(naive, tz: str, nonexistent: str = "shift_forward"):
    """Naive wall-clock timestamps -> tz-aware (DST aware), nanosecond resolution.

    A repeated (fall-back) hour resolves to its first occurrence. A skipped
    (spring-forward) hour is shifted forward or set to NaT, as requested.
    """
    s = pd.Series(naive)
    out = s.dt.tz_localize(
        tz, ambiguous=np.ones(len(s), dtype=bool), nonexistent=nonexistent
    )
    return out.dt.as_unit("ns")


def cutoff_for_day(day, cfg: dict):
    """cutoff(D) = (D - days_before) at cutoff.time EPT. Scalar or Series of days."""
    hh, mm = (int(x) for x in cfg["cutoff"]["time"].split(":"))
    shift = pd.Timedelta(days=int(cfg["cutoff"]["days_before"]))
    tod = pd.Timedelta(hours=hh, minutes=mm)
    tz = cfg["timezone"]
    if isinstance(day, (pd.Series, pd.Index)):
        naive = ns(pd.Series(pd.to_datetime(day))).dt.normalize() - shift + tod
        return localize(naive, tz)
    naive = pd.Timestamp(day).normalize() - shift + tod
    return naive.tz_localize(tz).as_unit("ns")


def label_to_ts(day: pd.Series, label: pd.Series, convention: str, tz: str) -> pd.DataFrame:
    """Map (delivery day, hour label) to a tz-aware hour-beginning timestamp.

    convention "hour_ending":    label k covers [k-1, k)  -> wall clock day + (k-1) h
    convention "hour_beginning": label k covers [k, k+1)  -> wall clock day + k h
    Returns ts_begin (NaT where the local time does not exist), hour (1-24, hour
    ending of the delivery day) and `valid`: False when the label falls outside
    the delivery day (e.g. label 0 under hour_ending) or does not exist.
    The wall-clock mapping gives unique timestamps on 23- and 24-label days.
    """
    day = ns(pd.Series(pd.to_datetime(day)).reset_index(drop=True))
    label = pd.Series(label).reset_index(drop=True).astype(int)
    offset = label - 1 if convention == "hour_ending" else label
    naive = day + pd.to_timedelta(offset, unit="h")
    ts = localize(naive, tz, nonexistent="NaT")
    inside = naive.dt.normalize() == day
    out = pd.DataFrame(
        {
            "ts_begin": ts,
            "hour": (naive.dt.hour + 1).astype(int),
            "valid": inside & ts.notna(),
        }
    )
    return out


def delivery_hours(days, tz: str) -> pd.DataFrame:
    """One row per delivery hour of the given days: delivery_day, hour (1-24), ts_begin.

    Hours are wall-clock labels 1-24; the non-existent spring-forward hour is
    absent (23 rows) and the repeated fall-back hour appears once (first occurrence).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
    grid = pd.DataFrame(
        {
            "delivery_day": ns(pd.Series(np.repeat(idx.to_numpy(), 24))),
            "hour": np.tile(np.arange(1, 25), len(idx)),
        }
    )
    ts = localize(grid["delivery_day"] + pd.to_timedelta(grid["hour"] - 1, unit="h"), tz, "NaT")
    grid["ts_begin"] = ts
    return grid.dropna(subset=["ts_begin"]).reset_index(drop=True)


def date_range(start, end) -> pd.DatetimeIndex:
    """Inclusive daily range of naive dates."""
    idx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return pd.DatetimeIndex(idx.to_numpy(dtype="datetime64[ns]"))


# ----------------------------------------------------------- vintage selection
def asof_select(
    df: pd.DataFrame,
    keys: list[str],
    cutoff,
    avail_col: str = "available_at",
) -> pd.DataFrame:
    """Rows with available_at < cutoff, keeping the latest available_at per key.

    `cutoff` is a tz-aware Timestamp, or a Series positionally aligned with `df`
    (one cutoff per row; used to select for many delivery days at once).
    """
    if isinstance(cutoff, (pd.Series, pd.Index)):
        cut = pd.Series(cutoff).reset_index(drop=True)
        mask = (df[avail_col].reset_index(drop=True) < cut).to_numpy()
    else:
        mask = (df[avail_col] < cutoff).to_numpy()
    sel = df.loc[mask]
    sel = sel.sort_values(avail_col, kind="stable")
    return sel.drop_duplicates(keys, keep="last")


def stale_flag(available_at: pd.Series, cutoff, max_age_hours: float) -> pd.Series:
    """1 where the selected vintage was issued more than max_age_hours before the cut-off."""
    if isinstance(cutoff, (pd.Series, pd.Index)):
        cutoff = pd.Series(cutoff).reset_index(drop=True)
        age = (cutoff - available_at.reset_index(drop=True)).dt.total_seconds() / 3600.0
        age.index = available_at.index
    else:
        age = (cutoff - available_at).dt.total_seconds() / 3600.0
    return (age > max_age_hours).astype(int)
