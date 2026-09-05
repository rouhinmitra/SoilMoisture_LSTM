"""
Build the FIELD-AVERAGED dataset as a NEW dataset. Nothing existing is modified.

  Data/Base/          untouched (point-sampled A00-A63)
  Data/Base_field/    new: A00-A63 replaced by the field average
  Code/Data/field_avg/presto_embeddings_field_interpolated.csv   new Presto block

WHAT IS SWAPPED, AND WHAT IS NOT.
Swapped - the 194-dim STATIC block, both parts of which can be replaced exactly:
  A00-A63    field-averaged AlphaEarth, per site x year
  emb_0..127 Presto regenerated from field-averaged S1 + S2
Not swapped - the dynamic Sentinel-2 columns (ndvi, b2..b12) in the station CSVs.
They are 100% populated on every day of 2017-2023 while only 430 of those 2483
dates coincide with an actual S2 overpass, and they correlate 0.82-0.87 (not 1.0)
with s2_presto_inputs.csv.  So they were produced by a different extraction AND a
different gap-filling than anything in this repo.  Replacing them would change
spatial support and processing lineage simultaneously and confound the test.

2024: field AlphaEarth covers 2017-2023.  year_range is (2017, 2024), so the
2024 rows carry the 2023 field values forward.  Scoring is 2017-2023 throughout,
so this affects only window padding.
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/Users/rouhinmitra/SM_work")
SRC = BASE / "Data/Base"
DST = BASE / "Data/Base_field"
AE = BASE / "Code/Data/field_avg/alphaearth_field.csv"
FILES = {"ne1": "ne1_1_maize.csv", "ne2": "ne2_1_maize.csv", "ne3": "ne3_1_maize.csv"}
SITEMAP = {"ne1": "US-Ne1", "ne2": "US-Ne2", "ne3": "US-Ne3"}
A = [f"A{i:02d}" for i in range(64)]

DST.mkdir(parents=True, exist_ok=True)
ae = pd.read_csv(AE)
for key, fn in FILES.items():
    df = pd.read_csv(SRC / fn)
    df["_d"] = pd.to_datetime(df["Date"])
    yr = df["_d"].dt.year
    sub = ae[ae.Site == SITEMAP[key]].set_index("Year")
    before = df[A].iloc[0].values.astype(float).copy()
    n_set = 0
    for y in range(2017, 2025):
        src_y = min(y, 2023)
        if src_y not in sub.index:
            continue
        m = (yr == y).values
        if not m.any():
            continue
        df.loc[m, A] = sub.loc[src_y, A].values.astype(float)
        n_set += int(m.sum())
    after = df[A].iloc[(yr == 2018).values][A].iloc[0].values.astype(float)
    df = df.drop(columns=["_d"])
    df.to_csv(DST / fn, index=False)
    print(f"{key}: wrote {fn}  rows={len(df)}  A00-A63 replaced on {n_set} rows "
          f"(2017-2024; 2024 carries 2023)")

print(f"\nfield dataset -> {DST}")
print("original Data/Base left untouched:",
      all((SRC / f).exists() for f in FILES.values()))
