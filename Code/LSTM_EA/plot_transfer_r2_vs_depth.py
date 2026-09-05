#!/usr/bin/env python
"""
Plot R2 versus soil moisture depth for temporal and spatial transfer.

Creates:
1) Station-averaged R2 vs depth (2 lines: temporal solid, spatial dotted)
2) Station-specific R2 vs depth (6 lines: temporal/spatial for Ne1/Ne2/Ne3)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "outputs" / "transfer_plots"

TEMPORAL_50_CSV = (
    BASE
    / "outputs"
    / "lstm_out_of_year_single_station_50cm"
    / "anchored_windows_station_mean_metrics.csv"
)
TEMPORAL_100_CSV = (
    BASE
    / "outputs"
    / "lstm_out_of_year_single_station_100cm"
    / "anchored_windows_station_mean_metrics.csv"
)

SPATIAL_25_CSV = BASE / "outputs" / "cv_results_lstm" / "cv_results.csv"
SPATIAL_50_CSV = BASE / "outputs" / "cv_results_lstm_50cm" / "cv_results.csv"
SPATIAL_100_CSV = BASE / "outputs" / "cv_results_lstm_100cm" / "cv_results.csv"

STATIONS = ["Ne1", "Ne2", "Ne3"]
DEPTHS = [25, 50, 100]

# Figure typography (axis labels, ticks, legend)
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 18

# Provided by user for 25 cm temporal interpolation.
TEMPORAL_25 = {
    "Ne1": 0.12,
    "Ne2": 0.63,
    "Ne3": 0.85,
}

# Manually provided temporal transfer R2 values.
TEMPORAL_50_MANUAL = {
    "Ne1": 0.37,
    "Ne2": 0.50,
    "Ne3": 0.73,
}

TEMPORAL_100_MANUAL = {
    "Ne1": 0.29,
    "Ne2": -0.34,
    "Ne3": 0.58,
}

# Manually provided 25 cm spatial transfer R2 values.
SPATIAL_25_MANUAL = {
    "Ne1": 0.52,
    "Ne2": 0.67,
    "Ne3": 0.68,
}


def _read_temporal(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["station"] = df["station"].str.strip()
    temporal = {}
    for station in STATIONS:
        row = df.loc[df["station"] == station]
        if row.empty:
            raise ValueError(f"Missing temporal row for station={station} in {csv_path}")
        temporal[station] = float(row.iloc[0]["mean_r2"])
    return temporal


def _read_spatial(csv_path: Path, r2_col: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.loc[df["test_station"].isin(STATIONS)].copy()
    spatial = {}
    for station in STATIONS:
        row = df.loc[df["test_station"] == station]
        if row.empty:
            raise ValueError(f"Missing spatial row for station={station} in {csv_path}")
        spatial[station] = float(row.iloc[0][r2_col])
    return spatial


def assemble_metrics() -> pd.DataFrame:
    temporal_by_depth = {
        25: TEMPORAL_25,
        50: TEMPORAL_50_MANUAL,
        100: TEMPORAL_100_MANUAL,
    }
    spatial_by_depth = {
        25: SPATIAL_25_MANUAL,
        50: _read_spatial(SPATIAL_50_CSV, "r2"),
        100: _read_spatial(SPATIAL_100_CSV, "r2"),
    }

    rows = []
    for station in STATIONS:
        for depth in DEPTHS:
            rows.append(
                {
                    "station": station,
                    "depth_cm": depth,
                    "r2_temporal": temporal_by_depth[depth][station],
                    "r2_spatial": spatial_by_depth[depth][station],
                }
            )
    return pd.DataFrame(rows)


def plot_average(df: pd.DataFrame) -> None:
    avg_df = (
        df.groupby("depth_cm", as_index=False)[["r2_temporal", "r2_spatial"]]
        .mean()
        .sort_values("depth_cm")
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(
        avg_df["r2_spatial"],
        avg_df["depth_cm"],
        linestyle=":",
        linewidth=3.5,
        color="red",
        marker="s",
        markersize=9,
        label="Spatial transfer",
    )
    ax.plot(
        avg_df["r2_temporal"],
        avg_df["depth_cm"],
        linestyle="-",
        linewidth=3.5,
        color="blue",
        marker="o",
        markersize=9,
        label="Temporal transfer",
    )

    ax.set_xlabel("$R^2$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Soil moisture depth (cm)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-0.9, 1.0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linestyle=":", linewidth=2.0, zorder=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "r2_vs_depth_avg.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_by_station(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {
        "Ne1": "purple",
        "Ne2": "green",
        "Ne3": "orange",
    }

    for station in STATIONS:
        sub = df.loc[df["station"] == station].sort_values("depth_cm")
        color = colors[station]

        ax.plot(
            sub["r2_spatial"],
            sub["depth_cm"],
            linestyle=":",
            linewidth=3.0,
            marker="s",
            markersize=8,
            color=color,
            label=f"{station} spatial",
        )
        ax.plot(
            sub["r2_temporal"],
            sub["depth_cm"],
            linestyle="-",
            linewidth=3.0,
            marker="o",
            markersize=8,
            color=color,
            label=f"{station} temporal",
        )

    ax.set_xlabel("$R^2$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Soil moisture depth (cm)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-0.9, 1.0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linestyle=":", linewidth=2.0, zorder=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color=colors["Ne1"], linewidth=3.0, label="Ne1"),
        Line2D([0], [0], color=colors["Ne2"], linewidth=3.0, label="Ne2"),
        Line2D([0], [0], color=colors["Ne3"], linewidth=3.0, label="Ne3"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=3.0, label="Spatial Transfer"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=3.0, label="Temporal Transfer"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
        framealpha=0.95,
        frameon=False
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "r2_vs_depth_by_station.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = assemble_metrics()
    metrics_path = OUTPUT_DIR / "transfer_r2_depth_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.6f")
    print(f"Saved -> {metrics_path}")
    print(metrics_df.to_string(index=False))

    plot_average(metrics_df)
    plot_by_station(metrics_df)


if __name__ == "__main__":
    main()
