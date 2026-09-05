"""
Per-site x year metrics collector for model variants.

Every variant run under the concept-shift work plan appends rows here, so variants
are diffable against the baseline in one place instead of per-run directories.

Metric definitions follow Evaluator.compute_metrics (src/evaluator.py:99) so the
numbers stay commensurable with the published tables.

Layout:
  outputs/model_variants/<variant>/        per-variant artefacts
  outputs/model_variants/summary_metrics.csv   one row per (variant, station, year)
                                               plus 'ALL' year rows and a site mean

Nothing under the published outputs/ tree is written to.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VARIANTS_ROOT = BASE / "outputs" / "model_variants"
SUMMARY_CSV = VARIANTS_ROOT / "summary_metrics.csv"

COLUMNS = [
    "variant", "experiment", "test_station", "year", "season",
    "r2", "rmse", "mae", "bias", "correlation", "nse", "n_samples",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Mirror of Evaluator.compute_metrics, tolerant of degenerate groups."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                "correlation": np.nan, "nse": np.nan, "n_samples": n}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(y_pred, y_true)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan

    return {
        "r2": float(r2) if r2 == r2 else np.nan,
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "bias": float(np.mean(y_pred - y_true)),
        "correlation": float(corr) if corr == corr else np.nan,
        "nse": float(r2) if r2 == r2 else np.nan,   # identical definition, as in evaluator.py:137
        "n_samples": int(n),
    }


def rows_for_fold(
    variant: str,
    experiment: str,
    test_station: str,
    dates: Iterable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    growing_season: tuple = (5, 10),
) -> List[dict]:
    """
    Build per-year rows plus an 'ALL' pooled row for one held-out station.

    Each is emitted twice: season='all' (whatever months the run covers) and
    season='may_oct', matching the r2_may_oct convention in run_cv.py:251-260.
    """
    df = pd.DataFrame({
        "Date": pd.to_datetime(list(dates)),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    lo, hi = growing_season
    subsets = {
        "all": df,
        "may_oct": df[(df["month"] >= lo) & (df["month"] <= hi)],
    }

    rows: List[dict] = []
    for season, sub in subsets.items():
        if sub.empty:
            continue
        for year, g in sub.groupby("year"):
            m = compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
            rows.append({"variant": variant, "experiment": experiment,
                         "test_station": test_station, "year": int(year),
                         "season": season, **m})
        m = compute_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append({"variant": variant, "experiment": experiment,
                     "test_station": test_station, "year": "ALL",
                     "season": season, **m})
    return rows


def append_rows(rows: List[dict], summary_csv: Path = SUMMARY_CSV) -> Path:
    """Append rows, replacing any existing rows for the same variant."""
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=COLUMNS)

    if summary_csv.exists():
        old = pd.read_csv(summary_csv)
        variants = set(new["variant"].unique())
        old = old[~old["variant"].isin(variants)]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out.to_csv(summary_csv, index=False, float_format="%.6f")
    return summary_csv


def comparison_table(
    variant: str,
    baseline: str = "baseline",
    season: str = "may_oct",
    summary_csv: Path = SUMMARY_CSV,
) -> Optional[pd.DataFrame]:
    """Per-site x year R² for `variant` beside `baseline`, with deltas."""
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    df = df[df["season"] == season]

    def pivot(v):
        s = df[df["variant"] == v]
        if s.empty:
            return None
        return s.set_index(["test_station", "year"])["r2"]

    b, v = pivot(baseline), pivot(variant)
    if v is None:
        return None

    out = pd.DataFrame({variant: v})
    if b is not None and baseline != variant:
        out.insert(0, baseline, b)
        out["delta"] = out[variant] - out[baseline]
    return out.reset_index().sort_values(["test_station", "year"])


def render(table: pd.DataFrame) -> str:
    """Plain-text rendering for reporting in the terminal."""
    if table is None or table.empty:
        return "(no rows)"
    return table.to_string(index=False, float_format=lambda x: f"{x:7.4f}")
