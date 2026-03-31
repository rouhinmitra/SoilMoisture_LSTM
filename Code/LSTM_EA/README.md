# LSTM / EA-LSTM for Root Zone Soil Moisture Prediction

An LSTM-based framework for predicting **Root Zone Soil Moisture (RZSM)** from daily time-series data. The repository supports both a standard multi-layer LSTM and an **Entity-Aware LSTM (EA-LSTM)** whose input gate is controlled by static site attributes (e.g., satellite embeddings, soil properties). The EA-LSTM architecture allows the model to learn site-specific information gating, making it well-suited for transfer across locations.

The model was developed and validated on three AmeriFlux eddy-covariance stations near Mead, Nebraska (US-Ne1, US-Ne2, US-Ne3), but is designed to work with **any location** where you have daily time-series features and a soil moisture target.

---

## Table of Contents

1. [Installation](#installation)
2. [Input Data Format](#input-data-format)
3. [Adapting for Your Data](#adapting-for-your-data)
4. [Running the Model](#running-the-model)
5. [Downloading Satellite Data from Google Earth Engine](#downloading-satellite-data-from-google-earth-engine)
6. [Repository Structure](#repository-structure)
7. [Output](#output)

---

## Installation

Requires **Python 3.10+**.

```bash
git clone <repo-url>
cd LSTM_EA
pip install -r requirements.txt
```

The core dependencies are PyTorch, pandas, NumPy, scikit-learn, and matplotlib. If you plan to use Presto embeddings, the Presto package is installed from GitHub automatically via `requirements.txt`. If you only need the base LSTM, you can skip that line.

---

## Input Data Format

The model reads **CSV files** with one row per day. The only strictly required column is `Date`; every other column is configurable. You only need a subset of the features listed below -- the model adapts to whatever you provide.

### Required Column

| Column | Format | Description |
|--------|--------|-------------|
| `Date` | `YYYY-MM-DD` | Daily timestamp. Must be parseable by `pd.to_datetime`. |

### Target Column

The column the model predicts. Set via `target_col` in the configuration.

| Column | Description |
|--------|-------------|
| `RZSM_25_avg` | Root zone soil moisture averaged to 25 cm depth (default target) |
| `RZSM_50_avg` | Root zone soil moisture averaged to 50 cm depth |
| `RZSM_100_avg` | Root zone soil moisture averaged to 100 cm depth |

You can use **any column name** as the target -- just set `target_col` accordingly.

### Dynamic Features (time-varying)

These are the features that change daily and are fed as input sequences to the LSTM. Configure via `dynamic_cols`.

| Category | Column Name(s) | Description |
|----------|----------------|-------------|
| Surface soil moisture | `SSM` | Surface soil moisture (single sensor) |
| | `SSM_avg` | Surface soil moisture (averaged across sensors) |
| Deeper soil water content | `SWC_PI_F_2_1_1` | Soil water content at depth level 2 |
| | `SWC_PI_F_3_1_1` | Soil water content at depth level 3 |
| Precipitation | `P_PI_F_1_1_1` | Precipitation (sensor 1) |
| | `P_PI_F_2_2_1` | Precipitation (sensor 2) |
| Irrigation | `I` | Daily irrigation amount |
| Air temperature | `TA_1_1_1` | Air temperature |
| Relative humidity | `RH_1_1_1` | Relative humidity |
| Latent heat flux | `LE_1_1_1` | Latent heat flux |
| Net radiation | `NETRAD_1_1_1` | Net radiation |
| Sentinel-2 bands | `ndvi` | Normalized Difference Vegetation Index |
| | `b2`, `b3`, `b4` | Visible bands (Blue, Green, Red) |
| | `b5`, `b6`, `b7` | Red-edge bands |
| | `b8`, `b8a` | NIR bands |
| | `b11`, `b12` | SWIR bands |

**You do not need all of these.** If you only have precipitation and temperature, list just those in `dynamic_cols`.

### Static Features (site-level attributes)

Static features characterize the site and drive the LSTM input gate (or are concatenated with dynamic features in the standard LSTM). Configure via `static_cols`.

| Category | Column Name(s) | Description |
|----------|----------------|-------------|
| Alpha Earth embeddings | `A00` through `A63` | 64-dimensional satellite embedding from Google's AlphaEarth Foundations dataset. Annual, 10 m resolution. See [downloading instructions](#alpha-earth-embeddings). |
| Presto embeddings | `emb_0` through `emb_127` | 128-dimensional embedding from NASA Harvest Presto model. Merged from a separate CSV on `Date` + `Site`. See [downloading instructions](#presto-embeddings). |
| Accumulated precipitation | `precip_jan_apr` | January--April accumulated precipitation. **Auto-computed** from the precipitation column if present; you do not need to add this yourself. |

If you have **no static features**, set `static_cols = []` and `use_presto_static = False`. The model will still work (standard LSTM concatenates dynamic + static, so with no static it uses dynamic only).

### Minimal CSV Example

A CSV with just precipitation, temperature, and a soil moisture target is sufficient:

```
Date,precip_mm,temp_c,soil_moisture
2020-01-01,0.0,2.3,0.31
2020-01-02,5.2,1.1,0.33
2020-01-03,0.0,3.5,0.32
...
```

You would then set:
```python
dynamic_cols = ['precip_mm', 'temp_c']
static_cols = []
target_col = 'soil_moisture'
```

---

## Adapting for Your Data

### Option A: Edit `run_cv.py` directly (quickest)

Open `run_cv.py` and modify the `CVConfig` dataclass (around line 55):

```python
@dataclass
class CVConfig:
    # Point to your data directory
    data_dir: str = "/path/to/your/data/"
    output_dir: str = "outputs/my_experiment"

    # Sequence length (number of days per input window)
    seq_length: int = 20

    # Model type: "LSTM" or "EALSTM"
    model_type: str = "LSTM"

    # List ONLY the columns present in your CSV
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "soil_moisture"  # your target column name

    # Disable Presto if you don't have embeddings
    use_presto_static: bool = False

    def __post_init__(self):
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                'precip_mm', 'temp_c',  # your available features
            ]
        if self.static_cols is None:
            self.static_cols = []  # empty if no static features
```

Then update the `STATIONS` dictionary at the top to point to your CSV files:

```python
STATIONS = {
    'Site1': 'site1_data.csv',
    'Site2': 'site2_data.csv',
    'Site3': 'site3_data.csv',
}
```

And update `CV_FOLDS` to match:

```python
CV_FOLDS = [
    {'train': ['Site1', 'Site2'], 'test': 'Site3', 'name': 'fold1'},
    {'train': ['Site1', 'Site3'], 'test': 'Site2', 'name': 'fold2'},
    {'train': ['Site2', 'Site3'], 'test': 'Site1', 'name': 'fold3'},
]
```

If you have only **one station**, you can skip cross-validation and use `main.py` with a YAML config instead (see Option B), splitting your single file into train/test by time.

### Option B: Use a YAML config with `main.py`

Create a YAML file (e.g., `configs/my_experiment.yaml`):

```yaml
name: "my_rzsm_experiment"

data:
  train_files:
    - "train_data.csv"
  test_files:
    - "test_data.csv"
  data_dir: "/path/to/your/data/"
  seq_length: 20
  nan_threshold: 0.1
  same_year_constraint: true

features:
  dynamic_cols:
    - "precip_mm"
    - "temp_c"
  static_cols: []
  target_col: "soil_moisture"
  add_temporal: false

model:
  model_type: "LSTM"
  hidden_dim: 32
  dropout: 0.4
  num_layers: 2

training:
  batch_size: 16
  epochs: 150
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 20
  gradient_clip: 1.0

seed: 42
output_dir: "outputs"
log_level: "INFO"
```

Run with:

```bash
python main.py --config configs/my_experiment.yaml
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_length` | 20 | Number of days in each input window. The model sees this many consecutive days to predict the last day's target. |
| `nan_threshold` | 0.1 | Maximum fraction of NaN values allowed per feature per window (10%). Windows exceeding this are dropped; those below are linearly interpolated. |
| `same_year_constraint` | `true` | If true, sliding windows cannot span across year boundaries. |
| `month_range` | `(4, 10)` | Restrict data to these months (inclusive). Set to `null`/`None` to use all months. |
| `year_range` | `(2017, 2024)` | Restrict data to these years (inclusive). Set to `null`/`None` to use all years. |
| `model_type` | `"LSTM"` | `"LSTM"` for standard multi-layer LSTM, `"EALSTM"` for Entity-Aware LSTM. Use EA-LSTM when you have meaningful static features. |
| `hidden_dim` | 32 | LSTM hidden state dimension. |
| `num_layers` | 2 | Number of stacked LSTM layers (only for standard LSTM). |
| `dropout` | 0.5 | Dropout probability for regularization. |
| `learning_rate` | 0.001 | AdamW optimizer learning rate. |
| `early_stopping_patience` | 20 | Stop training if validation loss doesn't improve for this many epochs. |

---

## Running the Model

### Leave-One-Station-Out Cross-Validation

```bash
cd LSTM_EA
python run_cv.py
```

This trains on N-1 stations and tests on the held-out station for each fold. Results (scatter plots, metrics CSV, loss curves) are saved to the `output_dir` specified in `CVConfig`.

### Single Experiment via YAML Config

```bash
python main.py --config configs/baseline.yaml
```

### What Happens During a Run

1. **Data loading**: CSVs are read, dates parsed, and optional month/year filters applied.
2. **Feature engineering**: Accumulated precipitation (`precip_jan_apr`) is computed automatically. Irrigation data is loaded from `Irrigation-data.xlsx` if present (otherwise set to 0). If Presto embeddings are enabled, they are merged on `Date` + `Site`.
3. **Scaling**: `StandardScaler` (zero mean, unit variance) is fit on the training data for dynamic features, static features, and target separately.
4. **Sliding windows**: Sequences of `seq_length` consecutive days are created. Windows with too many NaNs are dropped; the rest are interpolated.
5. **Training**: AdamW optimizer with learning rate scheduling (ReduceLROnPlateau) and early stopping. Best model checkpoint is saved.
6. **Evaluation**: Predictions are inverse-transformed to original scale. R², RMSE, MAE, bias, and correlation are computed.

---

## Downloading Satellite Data from Google Earth Engine

This section describes how to obtain the satellite-derived features used by the model. All scripts require a [Google Earth Engine account](https://earthengine.google.com/) and the `earthengine-api` Python package:

```bash
pip install earthengine-api
earthengine authenticate
```

### Sentinel-2 Surface Reflectance (HLS)

The script `presto_utils/download_s2.py` downloads Harmonized Sentinel-2 Level-2A surface reflectance data from the `COPERNICUS/S2_SR_HARMONIZED` collection.

**To adapt for your sites:**

1. Open `presto_utils/download_s2.py`
2. Edit the `STATIONS` dictionary with your site coordinates (longitude, latitude):

```python
STATIONS = {
    'MySite1': [-105.1, 40.0],   # [longitude, latitude]
    'MySite2': [-105.2, 40.1],
}
```

3. Set the date range:

```python
START_DATE = '2017-01-01'
END_DATE = '2024-12-31'
```

4. Update the `OUTPUT_FILE` path and the GEE project ID in `ee.Initialize(project="your-project-id")`.

5. Run:

```bash
cd LSTM_EA
python presto_utils/download_s2.py
```

**Output**: A CSV with columns `Date`, `Site`, `B1`--`B12`, `QA60`, `latitude`, `longitude`. Cloud-masked using the QA60 band.

**Bands downloaded**: B1 (Aerosols), B2 (Blue), B3 (Green), B4 (Red), B5-B7 (Red Edge), B8 (NIR), B8A (Narrow NIR), B9 (Water Vapor), B11-B12 (SWIR), QA60 (Cloud Mask).

To use these as dynamic features in the LSTM, compute NDVI and rename bands to lowercase (`b2`, `b3`, ..., `b8a`, `b11`, `b12`, `ndvi`) and merge with your main CSV on `Date`.

### Sentinel-1 SAR Backscatter

The script `presto_utils/download_s1.py` downloads Sentinel-1 Ground Range Detected (GRD) data from the `COPERNICUS/S1_GRD` collection.

**To adapt for your sites:**

1. Open `presto_utils/download_s1.py`
2. Edit the `SITES` dictionary with your coordinates:

```python
SITES = {
    'MySite1': [-105.1, 40.0],
    'MySite2': [-105.2, 40.1],
}
```

3. Set the date range and output path, and update the GEE project ID.

4. Run:

```bash
python presto_utils/download_s1.py
```

**Output**: A CSV with columns `Date`, `Site`, `VV`, `VH` (backscatter in dB, daily averaged across ascending/descending orbits).

### Presto Embeddings

[Presto](https://github.com/nasaharvest/presto) is a lightweight pre-trained transformer from NASA Harvest that encodes Sentinel-2, Sentinel-1, ERA5 climate data, and elevation into **128-dimensional embeddings** per location and time. These can be used as static features in the EA-LSTM.

**Steps:**

1. Download Sentinel-2 and Sentinel-1 data using the scripts above.
2. Download the Presto model weights:

```bash
python presto_utils/download_weights.py
```

3. Generate embeddings:

```bash
python presto_utils/generate_embeddings.py
```

This script:
- Loads S2 and S1 CSVs, merges them on `Date` + `Site`
- Runs each observation through the Presto encoder
- Outputs a CSV with columns `Date`, `Site`, `emb_0`...`emb_127`
- Interpolates to daily values (filling gaps between satellite overpasses)

**Output files**:
- `presto_embeddings_fused.csv` -- raw embeddings at satellite overpass dates
- `presto_embeddings_fused_interpolated.csv` -- linearly interpolated to daily

To use in the model, set in `CVConfig` or YAML config:

```python
use_presto_static = True
presto_embeddings_path = "/path/to/presto_embeddings_fused_interpolated.csv"
```

The data loader automatically merges the embeddings into your main data on `Date` and `Site`.

### Alpha Earth Embeddings

[AlphaEarth Foundations Satellite Embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) is a Google Earth dataset providing **64-dimensional embeddings** (bands `A00`--`A63`) per 10 m pixel, computed annually from 2017 onward. These encode land surface characteristics derived from satellite imagery.

- **Collection**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- **Resolution**: 10 m
- **Coverage**: Global terrestrial land, 2017--present
- **License**: CC-BY 4.0

**Extracting Alpha Earth embeddings at your site locations:**

```python
import ee
import pandas as pd

ee.Initialize(project="your-gee-project-id")

# Define your sites
sites = {
    'MySite1': ee.Geometry.Point([-105.1, 40.0]),
    'MySite2': ee.Geometry.Point([-105.2, 40.1]),
}

# Years to extract
years = range(2017, 2025)

# Band names
bands = [f'A{i:02d}' for i in range(64)]

collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

rows = []
for year in years:
    # Filter to the specific year
    image = collection.filter(
        ee.Filter.calendarRange(year, year, 'year')
    ).first()

    if image is None:
        continue

    for site_name, point in sites.items():
        values = image.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()

        row = {'Site': site_name, 'Year': year}
        row.update(values)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('alpha_earth_embeddings.csv', index=False)
print(f"Saved {len(df)} rows to alpha_earth_embeddings.csv")
```

The output CSV will have columns `Site`, `Year`, `A00`--`A63`. To use these in the model, merge the 64 embedding columns into your main data CSV (matching on year), and include `A00`--`A63` in your `static_cols`:

```python
static_cols = [f'A{i:02d}' for i in range(64)]
```

---

## Repository Structure

```
LSTM_EA/
├── run_cv.py                  # Leave-one-station-out cross-validation (main entry point)
├── main.py                    # Single experiment entry point (YAML config)
├── requirements.txt           # Python dependencies
├── configs/                   # YAML experiment configurations
│   ├── baseline.yaml
│   ├── presto.yaml
│   └── ...
├── src/                       # Core model and data pipeline
│   ├── config.py              # Dataclass configs (DataConfig, FeatureConfig, ModelConfig, TrainingConfig)
│   ├── data_loader.py         # CSV loading, cleaning, sliding window creation, scaling
│   ├── models.py              # EALSTM and StandardLSTM architectures
│   ├── trainer.py             # Training loop with early stopping and checkpointing
│   ├── evaluator.py           # Prediction, metrics (R2, RMSE, MAE), and plotting
│   ├── features.py            # Temporal feature engineering (sin/cos day-of-year, etc.)
│   └── presto_embeddings.py   # Presto embedding loading and lookup
├── presto_utils/              # Scripts for satellite data download and embedding generation
│   ├── download_s2.py         # Download Sentinel-2 from Earth Engine
│   ├── download_s1.py         # Download Sentinel-1 from Earth Engine
│   ├── generate_embeddings.py # Generate Presto 128-d embeddings from S1+S2
│   └── download_weights.py    # Download Presto model weights
├── utils/
│   ├── logging_utils.py       # Logging setup
│   └── experiment_tracker.py  # Experiment CSV tracker
└── outputs/                   # Model outputs (created at runtime)
```

---

## Output

Each run produces the following in the output directory:

| File | Description |
|------|-------------|
| `cv_results.csv` | Per-fold metrics: R², RMSE, MAE, bias, correlation, sample counts |
| `cv_scatter_plots.png` | Combined scatter plot (observed vs. predicted) for all folds |
| `cv_r2_summary.png` | Bar chart of R² scores across folds |
| `<fold>/checkpoint_best.pt` | Best model checkpoint (PyTorch state dict) |
| `<fold>/*_training_history.png` | Training and validation loss curves |
| `experiment_*.log` | Full training log with timestamps |

Metrics reported:
- **R²** (coefficient of determination)
- **RMSE** (root mean squared error)
- **MAE** (mean absolute error)
- **Bias** (mean prediction - mean observed)
- **Correlation** (Pearson r)
- **NSE** (Nash-Sutcliffe Efficiency)

---

## Citation

If you use this code, please cite the associated publication (forthcoming).

## License

See the repository license file for details.
