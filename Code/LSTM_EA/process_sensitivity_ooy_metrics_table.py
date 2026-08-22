#!/usr/bin/env python
"""
Process per-station metrics from `run_sensitivity_ooy.py`.

This script crawls experiment outputs under:
  outputs/sensitivity_ooy/<seq>/<config>/<years>/<station>/*_ooy_metrics.csv

For each file, it extracts the `OVERALL` row and records:
  station, config, r2, rmse, bias

Output is a long-format CSV table suitable for further analysis/plotting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CONFIG_ORDER: List[str] = [
    "baseline",
    "no_ssm",
    "no_sentinel2",
    "no_meteo",
    "no_precip_irrig",
]

REQUIRED_COLS = {"fold", "r2", "rmse", "bias"}


def parse_args() -> argparse.Namespace:
    default_output_root = (
        Path(__file__).resolve().parent / "outputs" / "sensitivity_ooy"
    )
    default_out = default_output_root / "sensitivity_ooy_station_metrics_table.csv"

    parser = argparse.ArgumentParser(
        description="Create a station-level r2/rmse/bias table from "
        "`run_sensitivity_ooy.py` outputs."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_output_root),
        help="Base output directory containing `seq*/...` (default: %(default)s).",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="seq20",
        help="Sequence directory name under `output-root` (default: %(default)s).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="years_2016_2023",
        help="Years directory name under each config (default: %(default)s).",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config subdirectory names (default: autodetect).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_out),
        help="Output CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required files/columns are missing or if any station is "
        "missing for a config.",
    )
    return parser.parse_args()


def autodetect_configs(seq_root: Path, years: str) -> List[str]:
    existing = []
    for cfg in CONFIG_ORDER:
        if (seq_root / cfg / years).exists():
            existing.append(cfg)
    return existing


def read_overall_metrics(csv_path: Path, strict: bool) -> Tuple[float, float, float]:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        msg = f"Missing required columns {sorted(missing)} in {csv_path}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        raise KeyError(msg)

    # `fold` holds per-year folds and an `OVERALL` row at the end.
    overall_rows = df[df["fold"].astype(str).str.upper().eq("OVERALL")]
    if overall_rows.empty:
        msg = f"No OVERALL row found in {csv_path}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        raise ValueError(msg)

    row = overall_rows.iloc[0]
    r2 = pd.to_numeric(row["r2"], errors="coerce")
    rmse = pd.to_numeric(row["rmse"], errors="coerce")
    bias = pd.to_numeric(row["bias"], errors="coerce")

    if pd.isna(r2) or pd.isna(rmse) or pd.isna(bias):
        msg = f"Non-numeric r2/rmse/bias in OVERALL row for {csv_path}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)

    return float(r2), float(rmse), float(bias)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_root = Path(args.output_root)
    seq_root = output_root / args.seq
    out_path = Path(args.out)

    if not seq_root.exists():
        raise FileNotFoundError(f"Missing seq directory: {seq_root}")

    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = autodetect_configs(seq_root, args.years)

    if not configs:
        raise ValueError(
            "No configs found. Provide --configs or ensure directories exist under: "
            f"{seq_root}/{args.years}"
        )

    logging.info(
        "Aggregating per-station OVERALL metrics for seq=%s years=%s configs=%s",
        args.seq,
        args.years,
        configs,
    )

    records: List[Dict[str, object]] = []
    seen_keys: set[Tuple[str, str]] = set()

    stations_by_config: Dict[str, set[str]] = {cfg: set() for cfg in configs}

    # Crawl each config explicitly so we only target the intended layout.
    for cfg in configs:
        years_root = seq_root / cfg / args.years
        if not years_root.exists():
            msg = f"Missing years directory: {years_root}"
            if args.strict:
                raise FileNotFoundError(msg)
            logging.warning(msg)
            continue

        for csv_path in sorted(years_root.rglob("*_ooy_metrics.csv")):
            # Expected layout:
            #   .../<years>/<station>/<file>_ooy_metrics.csv
            station = csv_path.parent.name

            key = (station, cfg)
            if key in seen_keys:
                if args.strict:
                    raise ValueError(f"Duplicate station/config metrics: {station=} {cfg=}")
                logging.warning(
                    "Duplicate metrics for %s/%s at %s; skipping this file.",
                    station,
                    cfg,
                    csv_path,
                )
                continue

            try:
                r2, rmse, bias = read_overall_metrics(csv_path, strict=args.strict)
            except (KeyError, ValueError):
                if args.strict:
                    raise
                continue

            seen_keys.add(key)
            stations_by_config[cfg].add(station)
            records.append(
                {
                    "seq": args.seq,
                    "years": args.years,
                    "station": station,
                    "config": cfg,
                    "r2": r2,
                    "rmse": rmse,
                    "bias": bias,
                }
            )

    if args.strict and records:
        all_stations = set().union(*stations_by_config.values())
        missing: Dict[str, List[str]] = {}
        for cfg, stations in stations_by_config.items():
            missing_stations = sorted(all_stations - stations)
            if missing_stations:
                missing[cfg] = missing_stations
        if missing:
            details = ", ".join(
                f"{cfg}: {len(st)} missing" for cfg, st in missing.items()
            )
            raise ValueError(f"Missing stations in some configs ({details})")

    df_out = pd.DataFrame.from_records(
        records, columns=["seq", "years", "station", "config", "r2", "rmse", "bias"]
    )

    if not df_out.empty:
        df_out["config"] = pd.Categorical(
            df_out["config"],
            categories=CONFIG_ORDER,
            ordered=True,
        )
        df_out = df_out.sort_values(["station", "config"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logging.info(
        "Wrote %d rows to %s",
        len(df_out),
        out_path,
    )


if __name__ == "__main__":
    main()

