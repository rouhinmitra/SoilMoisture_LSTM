#!/usr/bin/env python3
"""
Plot SSM_avg vs RZSM_25_avg relationships for US-Ne1, US-Ne2, and US-Ne3.

1. Annual Pearson r by year (line plot across sites).
2. Daily scatter with per-year OLS lines (1×3 panels by site).
3. Monthly Pearson r within each (year, month) — heatmap grid (years × May–Oct).
4. Daily SSM & RZSM25 time series for year–months with monthly r < threshold (default 0.5).

Growing season: May–October. Default years: 2016–2023.

Monthly cells use ~30 daily pairs; treat the heatmap as qualitative unless
--mask-insignificant is set (hides cells with p >= --alpha).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr as _pearsonr
except ImportError:
    _pearsonr = None

from compare_annual_sm_irrigation import (
    DEFAULT_DATA_DIR,
    MONTH_END,
    MONTH_START,
    RZSM_25,
    load_site_csvs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "ssm_rzsm25_annual_correlation"

SSM_COL = "SSM_avg"
DEFAULT_YEAR_RANGE = (2016, 2023)
MIN_SAMPLES_PER_YEAR = 10
MIN_SAMPLES_PER_MONTH = 15
DEFAULT_ALPHA = 0.05
DEFAULT_LOW_R_THRESHOLD = 0.5

GROWING_MONTHS = list(range(MONTH_START, MONTH_END + 1))
MONTH_LABELS = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]

SITE_STYLE = {
    "ne1": {"label": "US-Ne1 (Irrigated)", "color": "#1f77b4"},
    "ne2": {"label": "US-Ne2 (Irrigated)", "color": "#2ca02c"},
    "ne3": {"label": "US-Ne3 (Rainfed)", "color": "#d62728"},
}

SITE_PANEL_TITLES = {
    "ne1": "US-Ne1",
    "ne2": "US-Ne2",
    "ne3": "US-Ne3",
}

M3M3 = r"$\mathrm{m^3\,m^{-3}}$"
# Scatter-by-year figure: explicit m³/m³ and larger tick labels
SCATTER_UNIT = r"m$^3$/m$^3$"
SCATTER_TICK_LABELSIZE = 15
SCATTER_AXIS_LABELSIZE = 15
# Site CSVs store SSM / RZSM as percent (0–100); convert for m³/m³ axes.
SM_PERCENT_SCALE = 100.0


def _percent_to_m3m3(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / SM_PERCENT_SCALE


def _growing_season_mask(df: pd.DataFrame) -> pd.Series:
    return (df["Month"] >= MONTH_START) & (df["Month"] <= MONTH_END)


def annual_correlations(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_YEAR,
) -> pd.DataFrame:
    """Return site, year, r, n for each year with enough paired observations."""
    sub = df.loc[_growing_season_mask(df), ["Year", SSM_COL, RZSM_25]].dropna()
    rows = []
    for year, g in sub.groupby("Year"):
        if year < year_range[0] or year > year_range[1]:
            continue
        if len(g) < min_samples:
            continue
        r = float(np.corrcoef(g[SSM_COL], g[RZSM_25])[0, 1])
        rows.append({"year": int(year), "r": r, "n": len(g)})
    return pd.DataFrame(rows).sort_values("year")


def build_correlation_table(
    site_dfs: list[tuple[str, pd.DataFrame]],
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_YEAR,
) -> pd.DataFrame:
    frames = []
    for site, df in site_dfs:
        if site not in SITE_STYLE:
            continue
        if SSM_COL not in df.columns or RZSM_25 not in df.columns:
            raise ValueError(f"Site {site}: missing {SSM_COL} and/or {RZSM_25}")
        annual = annual_correlations(df, year_range, min_samples=min_samples)
        if annual.empty:
            continue
        annual["site"] = site
        annual["label"] = SITE_STYLE[site]["label"]
        frames.append(annual)
    if not frames:
        raise ValueError("No annual correlations computed; check data_dir and year range.")
    return pd.concat(frames, ignore_index=True)


def _cell_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (r, p_value, n); p_value is nan if scipy is unavailable."""
    n = len(x)
    if n < 2:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    if _pearsonr is not None and n >= 3:
        _, p = _pearsonr(x, y)
        return r, float(p), n
    return r, np.nan, n


def monthly_correlations(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_MONTH,
) -> pd.DataFrame:
    """Pearson r for each (year, month) using only that month's daily pairs."""
    sub = df.loc[_growing_season_mask(df), ["Year", "Month", SSM_COL, RZSM_25]].dropna()
    rows = []
    for (year, month), g in sub.groupby(["Year", "Month"], sort=True):
        year, month = int(year), int(month)
        if year < year_range[0] or year > year_range[1]:
            continue
        if month not in GROWING_MONTHS or len(g) < min_samples:
            continue
        r, p, n = _cell_pearson(g[SSM_COL].to_numpy(), g[RZSM_25].to_numpy())
        rows.append(
            {
                "year": year,
                "month": month,
                "month_label": MONTH_LABELS[month - MONTH_START],
                "r": r,
                "p_value": p,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def build_monthly_correlation_table(
    site_dfs: list[tuple[str, pd.DataFrame]],
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_MONTH,
) -> pd.DataFrame:
    frames = []
    for site, df in site_dfs:
        if site not in SITE_STYLE:
            continue
        monthly = monthly_correlations(df, year_range, min_samples=min_samples)
        if monthly.empty:
            continue
        monthly["site"] = site
        monthly["label"] = SITE_STYLE[site]["label"]
        frames.append(monthly)
    if not frames:
        raise ValueError("No monthly correlations computed; check data_dir and year range.")
    return pd.concat(frames, ignore_index=True)


def _monthly_r_grid(
    monthly_df: pd.DataFrame,
    site: str,
    years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_grid, p_grid, n_grid) shaped (n_years, n_months)."""
    r_grid = np.full((len(years), len(GROWING_MONTHS)), np.nan)
    p_grid = np.full_like(r_grid, np.nan)
    n_grid = np.zeros_like(r_grid)
    sub = monthly_df.loc[monthly_df["site"] == site]
    year_to_i = {y: i for i, y in enumerate(years)}
    for row in sub.itertuples(index=False):
        i = year_to_i.get(int(row.year))
        if i is None:
            continue
        j = int(row.month) - MONTH_START
        r_grid[i, j] = row.r
        p_grid[i, j] = row.p_value
        n_grid[i, j] = row.n
    return r_grid, p_grid, n_grid


def plot_monthly_correlation_heatmaps(
    monthly_df: pd.DataFrame,
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_MONTH,
    alpha: float = DEFAULT_ALPHA,
    mask_insignificant: bool = False,
    annotate_cells: bool = True,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    1×3 heatmaps: rows = years, columns = May–Oct, color = r.
    Cells with n < min_samples are left blank (NaN). Optional masking by p-value.
    """
    years = list(range(year_range[0], year_range[1] + 1))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), sharey=True)
    last_im = None

    for ax, site in zip(axes, ("ne1", "ne2", "ne3")):
        r_grid, p_grid, n_grid = _monthly_r_grid(monthly_df, site, years)
        display = r_grid.copy()
        insufficient = n_grid < min_samples
        display[insufficient] = np.nan
        if mask_insignificant:
            if _pearsonr is None:
                raise RuntimeError(
                    "scipy is required for --mask-insignificant; install scipy or omit the flag."
                )
            display[(p_grid >= alpha) & np.isfinite(p_grid)] = np.nan

        last_im = ax.imshow(
            display,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(GROWING_MONTHS)), MONTH_LABELS, fontsize=11)
        ax.set_yticks(range(len(years)), [str(y) for y in years], fontsize=11)
        ax.set_title(SITE_PANEL_TITLES[site], fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)

        if annotate_cells:
            for i, year in enumerate(years):
                for j, month in enumerate(GROWING_MONTHS):
                    n = int(n_grid[i, j])
                    if n < min_samples:
                        continue
                    r_val = r_grid[i, j]
                    if not np.isfinite(r_val):
                        continue
                    if mask_insignificant and np.isfinite(p_grid[i, j]) and p_grid[i, j] >= alpha:
                        continue
                    text_color = "white" if abs(r_val) > 0.55 else "black"
                    ax.text(
                        j,
                        i,
                        f"{r_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

    axes[0].set_ylabel("Year", fontsize=12)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$r_{\mathrm{SSM - RZSM25}}$", fontsize=12)

    note = (
        f"Each cell: Pearson r from daily pairs in that month only (n ≥ {min_samples}). "
        "~30 days per cell — interpret qualitatively."
    )
    if mask_insignificant:
        note += f" Only p < {alpha} shown."
    elif _pearsonr is None:
        note += " p-values unavailable (install scipy)."
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.text(0.5, 0.99, note, ha="center", va="top", fontsize=10, wrap=True)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def low_r_monthly_periods(
    monthly_df: pd.DataFrame,
    r_threshold: float = DEFAULT_LOW_R_THRESHOLD,
) -> pd.DataFrame:
    """Year–months where monthly Pearson r is strictly below r_threshold."""
    mask = monthly_df["r"].notna() & (monthly_df["r"] < r_threshold)
    return (
        monthly_df.loc[mask]
        .sort_values(["site", "year", "month"])
        .reset_index(drop=True)
    )


def _daily_series_for_period(
    df: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Daily Date, SSM, RZSM for one calendar month."""
    need = ["Date", "Year", "Month", SSM_COL, RZSM_25]
    sub = df.loc[(df["Year"] == year) & (df["Month"] == month), need].dropna(
        subset=[SSM_COL, RZSM_25]
    )
    return sub.sort_values("Date")


def plot_low_r_monthly_timeseries(
    site_dfs: list[tuple[str, pd.DataFrame]],
    monthly_df: pd.DataFrame,
    r_threshold: float = DEFAULT_LOW_R_THRESHOLD,
    n_cols: int = 3,
    save_dir: Path | None = None,
    show: bool = False,
) -> list[plt.Figure]:
    """
    For each site, plot daily SSM_avg and RZSM_25_avg in every year–month with r < r_threshold.
    One multi-panel figure per site.
    """
    site_map = {site: df for site, df in site_dfs if site in SITE_PANEL_TITLES}
    periods = low_r_monthly_periods(monthly_df, r_threshold=r_threshold)
    figures: list[plt.Figure] = []

    if periods.empty:
        print(f"No year–month cells with r < {r_threshold}; skipping low-r time series.")
        return figures

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for site in ("ne1", "ne2", "ne3"):
        site_periods = periods.loc[periods["site"] == site]
        if site_periods.empty:
            continue
        df = site_map.get(site)
        if df is None:
            continue

        n_panels = len(site_periods)
        n_rows = math.ceil(n_panels / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 2.8 * n_rows),
            squeeze=False,
            sharey=True,
        )

        for ax, row in zip(axes.flat, site_periods.itertuples(index=False)):
            daily = _daily_series_for_period(df, int(row.year), int(row.month))
            if daily.empty:
                ax.set_title(f"{row.month_label} {row.year} (no data)")
                ax.set_visible(False)
                continue

            dates = daily["Date"]
            ax.plot(
                dates,
                _percent_to_m3m3(daily[SSM_COL]),
                color="#2ecc71",
                linewidth=1.4,
                label="SSM",
            )
            ax.plot(
                dates,
                _percent_to_m3m3(daily[RZSM_25]),
                color="#3498db",
                linewidth=1.4,
                label="RZSM 25 cm",
            )
            ax.set_title(
                f"{row.month_label} {int(row.year)}  ($r$ = {row.r:.2f}, $n$ = {int(row.n)})",
                fontsize=11,
            )
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        for ax in axes.flat[n_panels:]:
            ax.set_visible(False)

        fig.supylabel(f"Soil moisture ({SCATTER_UNIT})", fontsize=12)
        fig.suptitle(
            f"{SITE_STYLE[site]['label']}: months with "
            f"$r_{{\\mathrm{{SSM-RZSM25}}}}$ < {r_threshold}",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"ssm_rzsm25_low_r_timeseries_{site}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved -> {out_path}")

        figures.append(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


def plot_annual_correlations(
    corr_df: pd.DataFrame,
    year_range: tuple[int, int],
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    for site in ("ne1", "ne2", "ne3"):
        sub = corr_df.loc[corr_df["site"] == site].sort_values("year")
        if sub.empty:
            continue
        style = SITE_STYLE[site]
        ax.plot(
            sub["year"],
            sub["r"],
            "o-",
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=7,
        )

    ax.set_xlim(year_range[0] - 0.3, year_range[1] + 0.3)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(year_range[0], year_range[1] + 1))
    ax.set_ylabel(r"$r_{\mathrm{SSM - RZSM25}}$", fontsize=13)
    ax.set_xlabel("Year", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _year_color_map(years: list[int]) -> dict[int, tuple]:
    """Sequential palette: warm (early years) to cool (late years)."""
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=min(years), vmax=max(years))
    return {int(y): cmap(norm(y)) for y in years}


def growing_season_daily_pairs(
    df: pd.DataFrame,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    sub = df.loc[_growing_season_mask(df), ["Year", SSM_COL, RZSM_25]].dropna()
    return sub[
        (sub["Year"] >= year_range[0]) & (sub["Year"] <= year_range[1])
    ].copy()


def plot_ssm_rzsm_scatter_by_year(
    site_dfs: list[tuple[str, pd.DataFrame]],
    year_range: tuple[int, int],
    min_samples: int = MIN_SAMPLES_PER_YEAR,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    1×3 scatter: SSM vs RZSM 25 cm, colored by year with per-year regression lines.
    """
    years = list(range(year_range[0], year_range[1] + 1))
    year_colors = _year_color_map(years)
    site_map = {site: df for site, df in site_dfs if site in SITE_PANEL_TITLES}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    all_ssm: list[float] = []
    all_rzsm: list[float] = []

    for ax, site in zip(axes, ("ne1", "ne2", "ne3")):
        df = site_map.get(site)
        if df is None:
            ax.set_visible(False)
            continue

        sub = growing_season_daily_pairs(df, year_range)
        for year, g in sub.groupby("Year"):
            year = int(year)
            if year not in year_colors or len(g) < min_samples:
                continue
            color = year_colors[year]
            x = _percent_to_m3m3(g[SSM_COL].to_numpy())
            y = _percent_to_m3m3(g[RZSM_25].to_numpy())
            all_ssm.extend(x.tolist())
            all_rzsm.extend(y.tolist())

            ax.scatter(
                x,
                y,
                c=[color],
                s=12,
                edgecolors="k",
                linewidths=0.25,
                alpha=0.65,
                zorder=2,
            )
            if len(x) >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(
                    xline,
                    slope * xline + intercept,
                    color=color,
                    linewidth=1.6,
                    zorder=3,
                )

        ax.set_title(SITE_PANEL_TITLES[site], fontsize=16, fontweight="bold")
        ax.set_xlabel(f"SSM ({SCATTER_UNIT})", fontsize=SCATTER_AXIS_LABELSIZE)
        ax.tick_params(axis="both", labelsize=SCATTER_TICK_LABELSIZE)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"RZSM 25cm ({SCATTER_UNIT})", fontsize=SCATTER_AXIS_LABELSIZE)

    if all_ssm and all_rzsm:
        pad = 0.02
        xmin, xmax = min(all_ssm), max(all_ssm)
        ymin, ymax = min(all_rzsm), max(all_rzsm)
        for ax in axes:
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=year_colors[y],
            markeredgecolor="k",
            markeredgewidth=0.4,
            markersize=8,
            label=str(y),
        )
        for y in years
    ]
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.legend(
        handles=handles,
        title="Year",
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=12,
        frameon=True,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run(
    data_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    min_samples: int = MIN_SAMPLES_PER_YEAR,
    min_samples_month: int = MIN_SAMPLES_PER_MONTH,
    alpha: float = DEFAULT_ALPHA,
    mask_insignificant: bool = False,
    low_r_threshold: float = DEFAULT_LOW_R_THRESHOLD,
    save_plot: bool = True,
    save_monthly_heatmap: bool = True,
    save_low_r_timeseries: bool = True,
    show: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    out_dir = Path(out_dir or DEFAULT_OUTPUT_DIR)

    site_dfs = load_site_csvs(data_dir)
    corr_df = build_correlation_table(site_dfs, year_range, min_samples=min_samples)
    monthly_df = build_monthly_correlation_table(
        site_dfs, year_range, min_samples=min_samples_month
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ssm_rzsm25_annual_correlation.csv"
    corr_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved -> {csv_path}")
    print(corr_df.pivot(index="year", columns="label", values="r").to_string(float_format="%.3f"))

    monthly_csv = out_dir / "ssm_rzsm25_monthly_correlation.csv"
    monthly_df.to_csv(monthly_csv, index=False, float_format="%.6f")
    print(f"Saved -> {monthly_csv}")
    n_years = year_range[1] - year_range[0] + 1
    n_cells = n_years * len(GROWING_MONTHS)
    print(
        f"Monthly grid: {n_cells} year–month cells per site "
        f"({n_years} years × {len(GROWING_MONTHS)} months)"
    )

    low_r_df = low_r_monthly_periods(monthly_df, r_threshold=low_r_threshold)
    low_r_csv = out_dir / "ssm_rzsm25_low_r_periods.csv"
    low_r_df.to_csv(low_r_csv, index=False, float_format="%.6f")
    print(f"Saved -> {low_r_csv}")
    if low_r_df.empty:
        print(f"No year–month cells with r < {low_r_threshold}.")
    else:
        print(
            f"Low-r periods (r < {low_r_threshold}): {len(low_r_df)} cells —\n"
            + low_r_df[["site", "year", "month_label", "r", "n"]].to_string(index=False)
        )

    if save_plot:
        plot_annual_correlations(
            corr_df,
            year_range,
            save_path=out_dir / "ssm_rzsm25_annual_correlation.png",
            show=show,
        )
        plot_ssm_rzsm_scatter_by_year(
            site_dfs,
            year_range,
            min_samples=min_samples,
            save_path=out_dir / "ssm_rzsm_scatter_by_year.png",
            show=show,
        )

    if save_plot and save_monthly_heatmap:
        plot_monthly_correlation_heatmaps(
            monthly_df,
            year_range,
            min_samples=min_samples_month,
            alpha=alpha,
            mask_insignificant=mask_insignificant,
            save_path=out_dir / "ssm_rzsm25_monthly_correlation_heatmap.png",
            show=show,
        )
        if mask_insignificant:
            plot_monthly_correlation_heatmaps(
                monthly_df,
                year_range,
                min_samples=min_samples_month,
                alpha=alpha,
                mask_insignificant=False,
                save_path=out_dir / "ssm_rzsm25_monthly_correlation_heatmap_all_cells.png",
                show=False,
            )

    if save_plot and save_low_r_timeseries and not low_r_df.empty:
        plot_low_r_monthly_timeseries(
            site_dfs,
            monthly_df,
            r_threshold=low_r_threshold,
            save_dir=out_dir,
            show=show,
        )

    return corr_df, monthly_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot annual and monthly SSM–RZSM25 correlations and daily scatter "
            "by year for US-Ne sites (May–Oct)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Directory of site CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_RANGE[0])
    parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_RANGE[1])
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES_PER_YEAR,
        help="Minimum daily pairs required per year (annual line plot)",
    )
    parser.add_argument(
        "--min-samples-month",
        type=int,
        default=MIN_SAMPLES_PER_MONTH,
        help="Minimum daily pairs per (year, month) heatmap cell",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level for --mask-insignificant (default: 0.05)",
    )
    parser.add_argument(
        "--mask-insignificant",
        action="store_true",
        help="Hide heatmap cells with p >= --alpha (also saves unmasked copy)",
    )
    parser.add_argument(
        "--no-monthly-heatmap",
        action="store_true",
        help="Skip the year × month correlation heatmap",
    )
    parser.add_argument(
        "--low-r-threshold",
        type=float,
        default=DEFAULT_LOW_R_THRESHOLD,
        help="Plot daily SSM/RZSM time series for months with r below this value (default: 0.5)",
    )
    parser.add_argument(
        "--no-low-r-timeseries",
        action="store_true",
        help="Skip daily time series for low-r year–months",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip saving figures")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        year_range=(args.year_start, args.year_end),
        min_samples=args.min_samples,
        min_samples_month=args.min_samples_month,
        alpha=args.alpha,
        mask_insignificant=args.mask_insignificant,
        low_r_threshold=args.low_r_threshold,
        save_plot=not args.no_plot,
        save_monthly_heatmap=not args.no_monthly_heatmap,
        save_low_r_timeseries=not args.no_low_r_timeseries,
        show=args.show,
    )


if __name__ == "__main__":
    main()
