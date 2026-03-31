# Using Geospatial Foundation Models (Presto) for RZSM Prediction

Yes, you can use **Presto** (or similar geospatial foundation models) to improve your soil moisture prediction pipeline. This document explains how Presto fits in and how to use it in this repo.

## What is Presto?

**Presto** is a lightweight, pre-trained transformer for remote sensing developed by NASA Harvest. It produces **128-dimensional embeddings** per location (and optionally per time window) by encoding:

- **Optical imagery** (Sentinel-2)  
- **Radar imagery** (Sentinel-1)  
- **Climate data** (ERA5)  
- **Elevation** (SRTM)  
- **Location** (lat/lon)

These embeddings are a compact, pre-trained representation of the land surface and climate at a point, which is well aligned with “site character” for your EA-LSTM (replacing or augmenting Alpha Earth A00–A63).

- Docs: [NASA Harvest Presto embeddings](https://nasaharvest.github.io/presto-embeddings/)  
- Code: [nasaharvest/presto](https://github.com/nasaharvest/presto)  
- Single-file inference: `single_file_presto.py` in the repo (for local runs)

## How it fits your setup

Your EA-LSTM uses:

- **Dynamic inputs**: SSM, SWC, P, TA, RH, irrigation, etc.  
- **Static inputs**: Currently **Alpha Earth A00–A63** (64 dims) + precip_jan_apr, precip_may_oct (and optionally irrigation). Static features drive the **input gate** (entity-aware behavior).

Presto can be used as the **static, site (and optionally year) representation**:

| Option | Description |
|--------|-------------|
| **Replace Alpha Earth** | Use **Presto 128-dim** instead of A00–A63. Static = Presto(128) + precip_jan_apr + precip_may_oct + [irrigation]. |
| **Augment** | Use **Presto 128 + Alpha 64** (192 dims) + precip + [irrigation]. More parameters, more data needed. |
| **Year-specific** | Presto can be computed per (lat, lon, year). Then each window uses the embedding for that site and year, so static features can vary by year (e.g. land use / climate that year). |

This repo implements the **“Replace Alpha Earth”** option: static features are Presto 128-d + precip_jan_apr + precip_may_oct (+ irrigation if configured as static). You can later extend to “Augment” by adding A00–A63 back into `static_cols`.

## How to get Presto embeddings

You need 128-d vectors per (site, year) or per site.

1. **NASA Harvest pipeline (Vertex AI + Earth Engine)**  
   - Use the [Presto embeddings pipeline](https://nasaharvest.github.io/presto-embeddings/) (Vertex AI + Google Earth Engine).  
   - Define your areas (e.g. small buffers around US-Ne1, US-Ne2, US-Ne3) and time window (e.g. one year).  
   - Export embeddings (e.g. to GCS or CSV) and then convert to the format below.

2. **Local inference (single_file_presto.py)**  
   - From [nasaharvest/presto](https://github.com/nasaharvest/presto), use `single_file_presto.py` if you have Sentinel-2/1, ERA5, SRTM inputs for your sites.  
   - Run it per (lat, lon) or per (lat, lon, year) and save 128-d vectors.

3. **Site coordinates**  
   - You need lat/lon for each site (e.g. US-Ne1, US-Ne2, US-Ne3).  
   - Example (Nebraska flux sites): Ne1 ~ (41.1651°N, 96.4766°W), Ne2 ~ (41.1649°N, 96.4701°W), Ne3 ~ (41.1797°N, 96.4397°W). Adjust to your actual coordinates.

## Using Presto in this repo

### 1. Precomputed Presto embeddings file

Create a CSV with one row per (site, year) and 128 embedding dimensions:

- **Required columns**: `site`, `year`, and 128 columns named `e0`, `e1`, …, `e127` (or `presto_0` … `presto_127`).  
- **Site names** must match what the pipeline infers from filenames (e.g. `ne1`, `ne2`, `ne3`).

Example:

```text
site,year,e0,e1,e2,...,e127
ne1,2018,0.12,-0.34,...
ne1,2019,0.11,-0.33,...
ne2,2018,...
```

You can also use a NumPy `.npz` with arrays keyed by `(site, year)` (see `src/presto_embeddings.py`).

### 2. Config

In your YAML config (e.g. `configs/baseline.yaml`), switch to Presto static features:

```yaml
features:
  dynamic_cols: [...]   # unchanged
  # Use Presto 128-d instead of Alpha Earth A00–A63
  use_presto_static: true
  presto_embeddings_path: "/path/to/presto_embeddings.csv"   # or .npz
  static_cols:
    - "precip_jan_apr"
    - "precip_may_oct"
  target_col: "RZSM_25_avg"
```

- With `use_presto_static: true`, the pipeline **ignores** Alpha Earth columns (A00–A63) and uses Presto 128-d + `static_cols` (+ irrigation if static).  
- Your CSV must still have `precip_jan_apr`, `precip_may_oct` (and optionally `irrigation`).  
- Model `stat_dim` becomes **128 + len(static_cols) + (1 if irrigation as static)** (e.g. 131 or 132).

### 3. Running

Same as before:

```bash
python main.py --config configs/your_presto_config.yaml
```

The data loader will:

- Load the Presto lookup from `presto_embeddings_path`.  
- For each window, take (site, year) from the file and dates, look up the 128-d embedding, and concatenate precip (and irrigation).  
- Fit/apply `StandardScaler` on the full static vector (Presto + precip + optional irrigation).

If a (site, year) is missing in the Presto file, the code will warn and can fall back to zeros or raise; see `src/presto_embeddings.py` and `src/data_loader.py`.

## Other geospatial foundation models

The same integration pattern (precomputed 128-d or 64-d vectors per site/year) works for other encoders:

- **Galileo** (NASA Harvest): [nasaharvest/galileo](https://github.com/nasaharvest/galileo)  
- **Prithvi** (IBM/NASA): if you export fixed-size site embeddings  
- **Custom encoders**: any model that outputs a fixed-size vector per location (and optionally time) can be dropped into the same `presto_embeddings` table and `use_presto_static` path by matching the column names (e.g. `e0..e127`).

So yes: **you can use Presto (and similar geospatial foundational models) for this prediction** by treating their embeddings as the static, entity-level input to your EA-LSTM, either replacing or augmenting Alpha Earth.
