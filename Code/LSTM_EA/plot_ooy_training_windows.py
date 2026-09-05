#!/usr/bin/env python
"""
Plot R² vs. training window length from multi-window OOY experiments.

Reads all_windows_lstm_ooy_metrics.csv produced by run_lstm_out_of_year_single_station.py
and plots R² (and optionally RMSE, MAE) vs. window_years per station to visualize
how much training data is optimal.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TITLE_FONTSIZE = 26
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16

LINE_COLORS = ["red", "blue", "green"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot R² vs. training window length from OOY multi-window experiments.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="outputs/lstm_out_of_year_single_station/all_windows_lstm_ooy_metrics.csv",
        help="Path to all_windows_lstm_ooy_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots; default is same as metrics file parent",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        choices=["r2", "rmse", "mae"],
        help="Primary metric to plot vs. window_years (default: r2)",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Stations to include; default: all in CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Run run_lstm_out_of_year_single_station.py first to generate all_windows_lstm_ooy_metrics.csv"
        )

    df = pd.read_csv(metrics_path)
    if "window_years" not in df.columns:
        raise ValueError(
            f"CSV must contain 'window_years' column. "
            "Ensure you ran the multi-window experiments (run_lstm_out_of_year_single_station.py)."
        )

    overall = df[df["fold"] == "OVERALL"].copy()
    if overall.empty:
        raise ValueError("No OVERALL fold rows found in metrics CSV.")

    if args.stations:
        overall = overall[overall["station"].isin(args.stations)]
    stations = overall["station"].unique().tolist()

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    ylabel = {"r2": "R²", "rmse": "RMSE", "mae": "MAE"}[metric]

    # Per-station plot (aggregate by window_years when multiple windows share same length)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, station in enumerate(sorted(stations)):
        sub = (
            overall[overall["station"] == station]
            .groupby("window_years", as_index=False)[metric]
            .mean()
            .sort_values("window_years")
        )
        ax.plot(
            sub["window_years"],
            sub[metric],
            marker="o",
            markersize=6,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            label=station,
            linewidth=2,
        )

    ax.set_xlabel("Training window length (years)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    # ax.set_title(
    #     f"OOY LSTM: {ylabel} vs. training window length", fontsize=TITLE_FONTSIZE
    # )
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    if metric == "r2":
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = output_dir / f"r2_vs_training_years_per_station.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Mean across stations per window (for windows that appear for all stations)
    grp = overall.groupby("window_years")[metric]
    mean_df = pd.DataFrame(
        {
            "window_years": grp.mean().index,
            "mean": grp.mean().values,
            "std": grp.std().values,
            "count": grp.count().values,
        }
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        mean_df["window_years"],
        mean_df["mean"],
        marker="s",
        markersize=8,
        color="blue",
        linewidth=2,
        label="Mean across stations",
    )
    if len(stations) > 1 and (mean_df["std"] > 0).any():
        ax2.fill_between(
            mean_df["window_years"],
            mean_df["mean"] - mean_df["std"],
            mean_df["mean"] + mean_df["std"],
            alpha=0.2,
            color="gray",
        )
    ax2.set_xlabel("Training window length (years)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel(f"Mean {ylabel} across stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title(
        f"OOY LSTM: mean {ylabel} vs. training window length", fontsize=TITLE_FONTSIZE
    )
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    if metric == "r2":
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    out_path2 = output_dir / f"r2_vs_training_years_mean.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
