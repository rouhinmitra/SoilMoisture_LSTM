"""
Load precomputed Presto (or other geospatial) embeddings for use as static features.

Expects a CSV with columns: site, year, e0, e1, ..., e127 (128 embedding dims),
or a .npz with arrays keyed by (site, year).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

PRESTO_DIM = 128
EMBED_COL_PREFIXES = ("e", "presto_")
INTERPOLATED_EMBED_COLS = [f"emb_{i}" for i in range(PRESTO_DIM)]


def _normalize_site(s: str) -> str:
    """Normalize site string to ne1 / ne2 / ne3 (lowercase, strip, remove us- prefix)."""
    s = str(s).strip().lower()
    if s.startswith("us-"):
        s = s[3:]
    if s in ("ne1", "ne2", "ne3"):
        return s
    if s in ("1", "2", "3"):
        return f"ne{s}"
    return s


def _embed_columns(df: pd.DataFrame) -> list:
    """Return list of embedding column names in order (e0..e127 or presto_0..presto_127)."""
    for prefix in EMBED_COL_PREFIXES:
        cols = [f"{prefix}{i}" for i in range(PRESTO_DIM)]
        if all(c in df.columns for c in cols):
            return cols
    return []


def load_presto_interpolated_df(path: str) -> pd.DataFrame:
    """
    Load Presto embeddings interpolated to daily (Date, Site, emb_0..emb_127) as a DataFrame.
    Use this to merge at daily scale; Site is normalized to ne1/ne2/ne3.

    Parameters
    ----------
    path : str
        Path to CSV with columns Date, Site, emb_0..emb_127.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Site, emb_0..emb_127. Site normalized to ne1/ne2/ne3.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Presto embeddings file not found: {path}")
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Site" not in df.columns:
        raise ValueError("Interpolated Presto CSV must have columns Date and Site")
    if not all(c in df.columns for c in INTERPOLATED_EMBED_COLS):
        raise ValueError(f"Interpolated Presto CSV must have columns emb_0..emb_{PRESTO_DIM - 1}")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Site"] = df["Site"].astype(str).apply(_normalize_site)
    return df[["Date", "Site"] + INTERPOLATED_EMBED_COLS]


def load_presto_embeddings(path: str) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Load Presto embeddings from CSV or NPZ.

    CSV formats supported:
    - Existing: columns `site`, `year`, and 128 embedding columns (e0..e127 or presto_0..presto_127).
    - Interpolated: columns `Date`, `Site`, and emb_0..emb_127 (daily per site). Aggregated to
      (site, year) by mean. Site normalized to ne1/ne2/ne3 (e.g. US-Ne1 -> ne1).

    NPZ: should contain arrays keyed by "<site>_<year>" (e.g. "ne1_2018") with shape (128,).

    Parameters
    ----------
    path : str
        Path to .csv or .npz file.

    Returns
    -------
    Dict[Tuple[str, int], np.ndarray]
        (site, year) -> array of shape (128,) float64.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Presto embeddings file not found: {path}")

    out: Dict[Tuple[str, int], np.ndarray] = {}

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        # Interpolated format: Date, Site, emb_0..emb_127 (daily per site)
        if "Date" in df.columns and "Site" in df.columns and all(c in df.columns for c in INTERPOLATED_EMBED_COLS):
            df["Date"] = pd.to_datetime(df["Date"])
            df["year"] = df["Date"].dt.year
            df["site"] = df["Site"].astype(str).apply(_normalize_site)
            agg = df.groupby(["site", "year"], as_index=False)[INTERPOLATED_EMBED_COLS].mean()
            for _, row in agg.iterrows():
                key = (row["site"], int(row["year"]))
                out[key] = row[INTERPOLATED_EMBED_COLS].values.astype(np.float64)
            logger.info(f"Loaded {len(out)} Presto embeddings from interpolated CSV ({path.name})")
            return out
        # Existing format: site, year, e0..e127 or presto_0..presto_127
        for col in ["site", "year"]:
            if col not in df.columns:
                raise ValueError(f"Presto CSV must have column '{col}'")
        embed_cols = _embed_columns(df)
        if len(embed_cols) != PRESTO_DIM:
            raise ValueError(
                f"Presto CSV must have 128 embedding columns (e0..e127 or presto_0..presto_127), found {len(embed_cols)}"
            )
        df["site"] = df["site"].astype(str).str.strip().str.lower()
        df["year"] = df["year"].astype(int)
        for _, row in df.iterrows():
            key = (row["site"], row["year"])
            out[key] = row[embed_cols].values.astype(np.float64)
        logger.info(f"Loaded {len(out)} Presto embeddings from CSV ({path.name})")
        return out

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.shape[0] == PRESTO_DIM:
                # key format "site_year" e.g. "ne1_2018"
                parts = key.split("_", 1)
                if len(parts) == 2:
                    site = parts[0].strip().lower()
                    try:
                        year = int(parts[1])
                        out[(site, year)] = arr.astype(np.float64)
                    except ValueError:
                        logger.warning(f"Skipping NPZ key (bad year): {key}")
            else:
                logger.warning(f"Skipping NPZ key (wrong shape/type): {key}")
        logger.info(f"Loaded {len(out)} Presto embeddings from NPZ ({path.name})")
        return out

    raise ValueError(f"Unsupported Presto embeddings format: {path.suffix}. Use .csv or .npz")


def get_embedding(
    lookup: Dict[Tuple[str, int], np.ndarray],
    site: str,
    year: int,
    missing: str = "zero",
) -> np.ndarray:
    """
    Get 128-d embedding for (site, year). Normalize site to lowercase.

    Parameters
    ----------
    lookup : Dict[Tuple[str, int], np.ndarray]
        From load_presto_embeddings().
    site : str
        Site id (e.g. ne1, ne2, ne3).
    year : int
        Year.
    missing : str
        If (site, year) not in lookup: "zero" -> return zeros; "raise" -> raise KeyError.

    Returns
    -------
    np.ndarray
        Shape (128,) float64.
    """
    key = (str(site).strip().lower(), int(year))
    if key in lookup:
        return lookup[key].copy()
    if missing == "zero":
        logger.warning(f"Presto embedding missing for {key}, using zeros")
        return np.zeros(PRESTO_DIM, dtype=np.float64)
    raise KeyError(f"Presto embedding missing for (site={site!r}, year={year})")
