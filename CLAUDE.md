# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo for predicting **Root Zone Soil Moisture (RZSM)** from daily flux-tower + satellite time series at three AmeriFlux maize sites near Mead, Nebraska (US-Ne1/Ne2/Ne3). The scientific product is a manuscript (`Paper/`); the code exists to produce the figures and tables in it. Models: Entity-Aware LSTM (static features gate the LSTM input gate), a standard LSTM, a DANN variant, plus Random Forest and exponential-filter baselines.

## Layout

```
Code/LSTM_EA/      all code (project root for every command — imports are `from src...`)
Data/Base/         station CSVs: ne1_1_maize.csv, ne2_1_maize.csv, ne3_1_maize.csv (~176 cols, daily)
Code/Data/         s2_pixels/, s1_pixels/ — Sentinel inputs + Presto embedding CSVs
Code/LSTM_EA/outputs/   per-experiment results (the canonical output tree)
outputs/           stale results from runs launched from the repo root — do not add to it
Paper/             manuscript drafts and reference PDFs
```

Data lives outside git; only code is versioned.

## Environment and commands

Use `venv/` (Python 3.12, has torch 2.9). `.venv/` (3.13) has no torch and will fail.

```bash
cd Code/LSTM_EA
../../venv/bin/python run_cv.py            # always run from Code/LSTM_EA
```

There is no test suite, linter, or CI. Verification means running an experiment script and reading the metrics CSV it writes. `presto_utils/test_load.py` and `run_seq_length_test.py` are scripts, not tests.

Common entry points:

```bash
python main.py --config configs/baseline.yaml     # single YAML-driven experiment
python run_cv.py                                  # leave-one-station-out spatial CV (25cm)
python run_cv_50cm.py / run_cv_100cm.py           # same, deeper targets
python run_lstm_out_of_year_single_station.py     # temporal transfer: hold out one year per fold
python run_sensitivity.py -e baseline no_sentinel2 --seq-length 20 --no-doy
python run_sensitivity_ooy.py --year-range 2017 2023
python run_hyperparam_search.py --n-trials 60     # Optuna over the run_cv pipeline
python random_forest_tuned.py                     # RF baseline on the same windows
python run_exp_filter_spatial_cv.py               # exponential-filter (SWI) baseline
python feature_analysis/run_all.py --data_dir ../../Data/Base
```

## Two configuration styles (know which one you're in)

1. **YAML + `main.py`** — `src/config.py` dataclasses (`DataConfig`, `FeatureConfig`, `ModelConfig`, `TrainingConfig`) loaded by `load_config()`. Only `main.py` uses this, and only it writes `outputs/experiments.csv` via `utils/experiment_tracker.py`.
2. **`run_*.py` with an embedded `@dataclass` config** (`CVConfig`, `OOYCVConfig`, …) edited in place. This is what the paper results actually come from. Each `run_*.py` builds `DataConfig`/`FeatureConfig`/etc. by hand and calls the same `src/` pipeline.

The two styles have drifted: `load_config()` treats `static_cols: "A00-A63"` as a shorthand and, under `use_presto_static: true`, **replaces** Alpha Earth with precip-only statics (Presto merged separately), whereas `run_cv.py` explicitly lists `emb_0..emb_127 + A00..A63 + precip_jan_apr` (augment, not replace). Don't assume a YAML config and a `run_*.py` with the "same" settings produce the same feature set.

Hyperparameters also differ per script by design (e.g. `run_cv.py` LSTM: hidden 32 / dropout 0.50 / 2 layers / tuned LR; `OOYCVConfig`: hidden 16 / dropout 0.20 / 1 layer). Copying a value between scripts changes published numbers — don't "harmonize" them unasked.

## Pipeline

`DataProcessor.prepare_data()` (`src/data_loader.py`, ~780 lines, the heart of the repo) → `RZSMDataset` → `get_model()` (`src/models.py`) → `Trainer` (`src/trainer.py`) → `Evaluator` (`src/evaluator.py`). The DANN path swaps in `src/models_da.py` + `src/trainer_da.py` and is driven by `run_cv_dann.py`.

Non-obvious behaviors baked into the loader:

- **Site id is inferred from the filename** (`_site_from_filepath` looks for the substring `ne1`/`ne2`/`ne3`). That id drives the Presto merge and the irrigation lookup. Renaming or relocating a station CSV silently disables both — the merge just logs a warning.
- **`precip_jan_apr` / `precip_may_oct` are auto-computed** per year from the first available `P_PI_F_*` column, before month filtering, so they survive `month_range=(4,10)`.
- **Irrigation is auto-appended as a static feature** (from `Irrigation-data.xlsx`, matched by site+year) whenever `'irrigation'` is not in `dynamic_cols`, unless `exclude_irrigation_static=True`. So the actual static dim is usually `len(static_cols) + 1`.
- **Presto embeddings merge at daily resolution** on `Date` + `Site` from `presto_embeddings_fused_interpolated.csv`; `src/presto_embeddings.py` normalizes `US-Ne1` → `ne1`.
- **Windowing**: `seq_length` consecutive days predict the last day's target. A window is dropped if any feature exceeds `nan_threshold` (default 10%) NaNs, otherwise linearly interpolated. `same_year_constraint` prevents windows spanning year boundaries.
- **Scaling**: separate `StandardScaler` for dynamic / static / target, fit on training rows only; when `val_years` is set those years are excluded from the fit too.
- **Static tensors carry a time axis** — `X_s` is `(batch, seq_len, stat_dim)`. `EALSTM.forward` uses only `x_stat[:, -1, :]`; `StandardLSTM` concatenates statics onto the dynamic features at every timestep.

`src/models.py` implements `EALSTMCell` explicitly and loops over timesteps in Python (no cuDNN fusion), so EA-LSTM runs are notably slower than the standard LSTM.

## Outputs and the plotting layer

Each run writes to `Code/LSTM_EA/outputs/<experiment_name>/`: `cv_results.csv` (R², RMSE, MAE, bias, correlation, NSE), per-fold `checkpoint_best.pt`, prediction CSVs, scatter/loss PNGs, and a timestamped log.

The `plot_*.py` / `compute_*.py` scripts are a second stage that reads those directories by **hardcoded path** — e.g. `plot_spatial_transferability_scatter.py` expects `outputs/sensitivity/seq20/baseline_no_doy`, `outputs/rf_cv_tuned`, and `outputs/exp_filter_spatial_cv/cv_predictions.csv` to already exist. Before editing or running a plot script, check the constants at the top and run the producing experiment first.

## Absolute paths

~33 Python files hardcode `/Users/rouhinmitra/SM_work/...` for `data_dir` and `presto_embeddings_path`. This is accepted in the repo; when adding a script, follow the existing pattern rather than introducing a new config mechanism, and use `Path(__file__).resolve().parent` for anything under `Code/LSTM_EA/`.

## Further reading

- `Code/LSTM_EA/README.md` — input CSV schema, full feature tables, and Earth Engine download instructions for Sentinel-1/2 and AlphaEarth embeddings.
- `Code/LSTM_EA/PRESTO_INTEGRATION.md` — what Presto embeddings are and the replace-vs-augment options for static features.
