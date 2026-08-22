# Reproducing the manuscript results

Maps every figure and table in **`Paper/RZSM_JHydrology_Finalv2 SK.docx`** ("Spatial
Transferability of Root Zone Soil Moisture Estimation in Nebraska Croplands using Machine
Learning") to the script that produced it and the output file the numbers were read from.

Verified 2026-08-22 against the outputs then present in `Code/LSTM_EA/outputs/`.

All paths below are relative to `Code/LSTM_EA/`. Run every command from that directory with
`../../venv/bin/python` (the 3.12 venv; `.venv/` has no torch).

## Experiment design

Target `RZSM_25_avg`, growing season May–October, three models (exponential filter, Random
Forest, LSTM), two transfer experiments:

- **Temporal transfer** — leave-one-year-out within each station.
- **Spatial transfer** — leave-one-site-out (LOSO) across US-Ne1 / US-Ne2 / US-Ne3.

Section 5 replicates both at 50 cm and 100 cm.

## Provenance

| Paper item | Producer | Output read for the paper |
|---|---|---|
| Fig 1 study area | — | not in repo (hand-made) |
| Fig 2 flowchart | — | not in repo (hand-made) |
| Fig 3 2019 profiles | **no producer found** | — |
| Fig 4a–c SSM–RZSM scatter | `plot_ssm_rzsm25_annual_correlation.py` | `outputs/ssm_rzsm25_annual_correlation/ssm_rzsm_scatter_by_year.png` |
| Fig 4d annual Pearson r | same | `.../ssm_rzsm25_annual_correlation.png` + `.csv` |
| Fig 5 temporal scatter | `plot_temporal_transferability_scatter.py` | `outputs/temporal_transferability/temporal_transferability_scatter.png` |
| Fig 6 annual R² bars | same | `.../temporal_transferability_r2_bars.png` + `_per_year_metrics.csv` |
| **Table 2** temporal metrics | same | `.../temporal_transferability_metrics.csv` |
| Fig 7 spatial scatter | `plot_spatial_transferability_scatter.py` | `outputs/spatial_transferability/spatial_transferability_scatter.png` |
| **Table 3** spatial metrics | same | `.../spatial_transferability_metrics.csv` |
| Fig 8 3×3 fit-year series | `plot_spatial_transferability_timeseries.py` | `.../spatial_transferability_timeseries.png` + `_selected_years.csv` |
| Fig 9 left temporal window sweep | `plot_ooy_training_windows.py` | `outputs/lstm_out_of_year_single_station/all_windows_lstm_ooy_metrics.csv` |
| Fig 9 right spatial window sweep | `run_spatial_transfer_training_windows.py` | `outputs/spatial_transfer_training_windows/r2_vs_training_years_spatial.png` + `all_windows_spatial_metrics.csv` |
| **Table 4** temporal ablation | `run_sensitivity_ooy.py` → `process_sensitivity_ooy_metrics_table.py` | `outputs/sensitivity_ooy/sensitivity_ooy_station_metrics_table.csv` |
| **Table 5** spatial ablation | `run_sensitivity.py` → `compute_sensitivity_growing_season_table.py` | `outputs/sensitivity/seq20/sensitivity_growing_season_table.csv` |
| Fig 10 R² vs depth | `plot_transfer_r2_vs_depth.py` | `outputs/transfer_plots/*.png` + `transfer_r2_depth_metrics.csv` |
| Table S1 soil texture | — | `clay_percent_*` / `sand_percent_*` columns in `Data/Base/*.csv` |
| Table S2 pBias at US-Ne2 | `run_sensitivity.py` | `outputs/sensitivity/seq20/sensitivity_results.csv` |

### Upstream runs feeding the plotting layer

The `plot_*` scripts do not train anything — they read prediction CSVs from earlier experiment
runs by hardcoded path. Those must exist first.

- **Temporal** (Table 2, Figs 5–6): `run_lstm_out_of_year_single_station.py`,
  `run_rf_out_of_year_single_station.py`, `run_exp_filter_out_of_year_single_station.py`
  → `all_stations_*_ooy_predictions.csv`.
- **Spatial** (Table 3, Figs 7–8): LSTM from `outputs/sensitivity/seq20/baseline_no_doy`
  (produced by `run_sensitivity.py`, **not** `run_cv.py`); RF from `outputs/rf_cv_tuned`
  (`random_forest_tuned.py`); exponential filter from `outputs/exp_filter_spatial_cv`
  (`run_exp_filter_spatial_cv.py`).

Note the spatial LSTM source: `outputs/cv_results_lstm/cv_results.csv` gives 0.55 / 0.46 / 0.79
at 25 cm, which is **not** the published 0.52 / 0.67 / 0.68.

## Regenerating the two headline tables

```bash
cd Code/LSTM_EA
../../venv/bin/python plot_temporal_transferability_scatter.py   # Table 2, Figs 5-6
../../venv/bin/python plot_spatial_transferability_scatter.py    # Table 3, Fig 7
```

Both rewrite their PNGs as well as their CSVs. Back up the CSVs first if you need to diff:

```bash
cp -p outputs/temporal_transferability/temporal_transferability_metrics.csv{,.bak}
cp -p outputs/spatial_transferability/spatial_transferability_metrics.csv{,.bak}
```

**Result of the 2026-08-22 rerun:**

- Table 2 — byte-identical to the published values. Reproduces.
- Table 3 — RF and LSTM columns byte-identical; **all four exponential-filter columns differ.**
  See discrepancy 1.

## Known discrepancies

Ordered with the verified ones first.

1. **Table 3's exponential-filter column no longer reproduces.** Confirmed by rerun.

   | | published (Mar 10 run) | current code+data (Mar 16 run) |
   |---|---|---|
   | US-Ne1 | R² 0.3697, best T=1 | R² 0.0095, best T=4 |
   | US-Ne2 | R² −1.6995, best T=1 | R² −1.7842, best T=5 |
   | US-Ne3 | R² 0.3514, best T=1 | R² 0.3032, best T=1 |
   | Average | **−0.3261** | **−0.4905** |

   The Mar-10 run selected T=1 at all three sites; the current one selects 1/5/4, and the
   training-fold NSE values differ too (Ne3 fold 0.0519 → 0.3621), so the T-selection input
   changed, not just the tie-breaking. Compare
   `outputs/exp_filter_spatial_cv/experiment_20260310_095448.log` (published) against
   `experiment_20260316_095859.log` (current). `run_exp_filter_spatial_cv.py` has only one
   commit in git history, so the Mar-10 version of the script is not recoverable from the repo.

   Manuscript text affected: the abstract's "exponential filter failed entirely (R² = −0.33)"
   and §4.4's "reasonable performance at US-Ne1 (R² = 0.370) and US-Ne3 (R² = 0.351)" — under
   the current code US-Ne1 is 0.009, i.e. no skill, which strengthens the paper's argument but
   changes the sentence.

2. **Fig 10's temporal values are hardcoded and contradict the outputs.**
   `plot_transfer_r2_vs_depth.py` sets `TEMPORAL_50_MANUAL = {0.37, 0.50, 0.73}` and
   `TEMPORAL_100_MANUAL = {0.29, −0.34, 0.58}`. The actual runs give
   50 cm −0.04 / 0.32 / 0.40 and 100 cm −0.06 / −1.57 / −0.64
   (`outputs/lstm_out_of_year_single_station_{50,100}cm/anchored_windows_station_mean_metrics.csv`).

3. **Fig 10 mixes two spatial pipelines.** 25 cm uses `SPATIAL_25_MANUAL = {0.52, 0.67, 0.68}`
   (the sensitivity baseline, consistent with Tables 3/5); 50 and 100 cm are read from
   `outputs/cv_results_lstm_{50,100}cm/cv_results.csv`, a different configuration
   (`run_cv_50cm.py` / `run_cv_100cm.py`).

4. **Table 5 rows `wo AE_s2` and `wo presto_s2` are missing from the growing-season table.**
   Their prediction CSVs exist under `outputs/sensitivity/seq20/no_ae_s2/` and `no_presto_s2/`,
   but `sensitivity_growing_season_table.csv` predates them (13 rows). Rerun
   `compute_sensitivity_growing_season_table.py` to fold them in.

5. **Table S2's source has been overwritten.** `outputs/sensitivity/seq20/sensitivity_results.csv`
   holds only 3 rows — a `no_presto_s2` rerun replaced the full per-fold table, taking the
   per-fold `pbias` for `no_alpha_earth` at US-Ne2 with it.

6. **Table 2's LSTM average (0.514) and Table 4's baseline (0.53) are different runs.**
   Table 2 comes from `lstm_out_of_year_single_station` (0.135 / 0.564 / 0.843);
   Table 4 from `sensitivity_ooy` baseline (0.119 / 0.631 / 0.853).

7. **§3.3's hyperparameters describe the temporal model, not the spatial one.** The paper states
   single-layer, lr 5e-4, weight decay 1e-3, batch 32 — that is `OOYCVConfig`. Tables 3 and 5
   come from `run_cv.CVConfig`: **2 layers**, lr 1.357e-3, weight decay 5.52e-5, batch 16.
   Hidden 32, dropout 0.5, seq 20, patience 20, 150 epochs and val year 2022 are correct for both.

8. **§3 says the data span 2016–2023; the spatial runs use 2017–2024.**
   `CVConfig.year_range = (2017, 2024)`, and `all_windows_spatial_metrics.csv` lists allowed
   years `[2017, 2018, 2019, 2020, 2021, 2023, 2024]` with 2022 held out for validation.

9. **`baseline` and `baseline_no_doy` are the same run.** `run_cv.py` builds `FeatureConfig`
   with `add_temporal` commented out, so DOY features are never added and `--no-doy` has no
   effect. Both rows in `sensitivity_growing_season_table.csv` are identical to 16 digits.

10. **Fig 3 has no producer script** — presumably a notebook or external tool.

Items 7 and 8 are manuscript-text fixes. Items 1–5 are data-provenance issues needing a decision
about reruns.

## Recovering the published figures

`outputs/` is gitignored, so overwritten figures are not recoverable from git. The published
versions are embedded in the `.docx` and can be extracted:

```bash
unzip -o "Paper/RZSM_JHydrology_Finalv2 SK.docx" 'word/media/*' -d /tmp/paper_figs
```

`image8.png` is Fig 7 (spatial scatter), `image7.png` Fig 6, `image6.png` Fig 5,
`image12/13.png` Fig 10.
