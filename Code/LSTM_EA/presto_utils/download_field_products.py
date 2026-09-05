"""
Field-AVERAGED extraction of all three satellite products: S1, S2, AlphaEarth.

Everything in this repo was previously sampled at ee.Geometry.Point(tower) with
scale=10 - a single 10 m pixel:
    download_s1.py       point
    download_s2.py       point
    README AlphaEarth    point
This replaces the point with the delineated field geometry from
build_field_geoms.py (../../Data/field_boundaries.geojson):

    US-Ne1  centre-pivot circle, r=380 m, 45.4 ha   CDL purity 0.990 (min 0.983)
    US-Ne2  centre-pivot circle, r=420 m, 55.4 ha   CDL purity 0.977 (min 0.973)
    US-Ne3  CDL polygon,                  63.8 ha   CDL purity 1.000

Pivot centres come from the CDL all-year intersection centroid, NOT the tower -
the towers sit 107 m (Ne1) and 104 m (Ne2) off centre, which is why a
tower-centred buffer was only 0.87-0.90 pure.

S1  : COPERNICUS/S1_GRD, IW, VV+VH.  Adds VV_VH_diff = VV - VH (dB).  Because
      VV/VH are already logarithmic, the dB DIFFERENCE is the linear ratio
      10*log10(sigma_VV/sigma_VH); dividing dB values would be meaningless.
      Ascending-only / relative orbit 136 / platform A is all that exists here
      (verified against the unfiltered archive), so no geometry mixing.
      Linearly interpolated to daily with provenance flags.
S2  : COPERNICUS/S2_SR_HARMONIZED, same QA60 cloud mask as download_s2.py,
      same 13 bands, so it is a drop-in replacement for s2_presto_inputs.csv.
      Cloud masking makes pixel count vary, so n_valid is recorded per date.
AE  : GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, A00-A63, annual, 10 m.

Outputs -> ../../Data/
    s1_pixels/s1_field_observations.csv, s1_pixels/s1_field_daily.csv
    s2_pixels/s2_field_inputs.csv
    alphaearth_field.csv
"""
import json, os, time
import ee
import numpy as np
import pandas as pd

PROJECT = "irriquate-backend"
GEOJSON = "../../Data/field_boundaries.geojson"
START, END = "2017-01-01", "2023-12-31"
S2_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','QA60']
AE_BANDS = [f"A{i:02d}" for i in range(64)]


def load_geoms():
    g = json.load(open(GEOJSON))
    return {f["properties"]["site"]: ee.Geometry(f["geometry"]) for f in g["features"]}


# ---------------------------------------------------------------- S1
def s1(geoms):
    rows = []
    for site, geom in geoms.items():
        col = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(geom).filterDate(START, END)
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
               .filter(ee.Filter.eq("instrumentMode", "IW")))

        def ex(img):
            i = img.select(["VV", "VH"])
            m = i.reduceRegion(ee.Reducer.mean(), geom, 10)
            c = i.select("VV").reduceRegion(ee.Reducer.count(), geom, 10)
            return ee.Feature(None, {"Site": site, "ms": img.date().millis(),
                                     "orbit": img.get("orbitProperties_pass"),
                                     "VV": m.get("VV"), "VH": m.get("VH"),
                                     "n": c.get("VV")})
        t0 = time.time()
        fs = col.map(ex).getInfo()["features"]
        print(f"  S1 {site}: {len(fs)} images ({time.time()-t0:.0f}s)")
        for f in fs:
            p = f["properties"]
            if p.get("VV") is None or p.get("VH") is None:
                continue
            rows.append(dict(Site=site, ms=p["ms"], orbit=p.get("orbit"),
                             VV=float(p["VV"]), VH=float(p["VH"]),
                             n_pixels=int(p["n"] or 0)))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df.ms, unit="ms").dt.normalize()
    df["VV_VH_diff"] = df.VV - df.VH
    return df.drop(columns=["ms"])


def interp_daily(obs):
    full = pd.date_range(START, END, freq="D")
    out = []
    for site, g in obs.groupby("Site"):
        d = g.groupby("Date")[["VV", "VH", "VV_VH_diff"]].mean().reindex(full)
        seen = d.VV.notna()
        it = d.interpolate(method="linear", limit_direction="both")
        idx = np.arange(len(full)); oi = idx[seen.values]
        gap = np.abs(idx[:, None] - oi[None, :]).min(axis=1)
        out.append(pd.DataFrame({"Site": site, "Date": full, "VV": it.VV.values,
                                 "VH": it.VH.values, "VV_VH_diff": it.VV_VH_diff.values,
                                 "is_observed": seen.values, "days_to_nearest_obs": gap}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- S2
def mask_clouds(img):
    qa = img.select("QA60")
    m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(m).select(S2_BANDS).copyProperties(img, ["system:time_start"])


def s2(geoms):
    rows = []
    for site, geom in geoms.items():
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(geom).filterDate(START, END).map(mask_clouds))

        def ex(img):
            m = img.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9)
            c = img.select("B4").reduceRegion(ee.Reducer.count(), geom, 10, maxPixels=1e9)
            return ee.Feature(None, {"Site": site, "ms": img.get("system:time_start"),
                                     "n": c.get("B4"),
                                     **{b: m.get(b) for b in S2_BANDS}})
        t0 = time.time()
        fs = col.map(ex).getInfo()["features"]
        print(f"  S2 {site}: {len(fs)} images ({time.time()-t0:.0f}s)")
        for f in fs:
            p = f["properties"]
            if p.get("B4") is None or not p.get("n"):
                continue
            rows.append(dict(Site=site, ms=p["ms"], n_valid=int(p["n"]),
                             **{b: (float(p[b]) if p.get(b) is not None else np.nan)
                                for b in S2_BANDS}))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df.ms, unit="ms").dt.normalize()
    return df.drop(columns=["ms"])


# ---------------------------------------------------------------- AlphaEarth
def alphaearth(geoms):
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    rows = []
    for yr in range(2017, 2024):
        for site, geom in geoms.items():
            # filterBounds BEFORE first(): without it, first() returns an arbitrary
            # global tile and reduceRegion blows the maxPixels budget.
            img = col.filter(ee.Filter.calendarRange(yr, yr, "year")).filterBounds(geom).first()
            v = img.select(AE_BANDS).reduceRegion(
                ee.Reducer.mean(), geom, 10, maxPixels=1e9).getInfo()
            n = img.select("A00").reduceRegion(
                ee.Reducer.count(), geom, 10, maxPixels=1e9).getInfo()["A00"]
            rows.append({"Site": site, "Year": yr, "n_pixels": int(n), **v})
        print(f"  AE {yr}: done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ee.Initialize(project=PROJECT)
    G = load_geoms()
    print(f"Field geometries: {list(G)}\n")

    o = s1(G)
    os.makedirs("../../Data/s1_pixels", exist_ok=True)
    o.to_csv("../../Data/s1_pixels/s1_field_observations.csv", index=False)
    d = interp_daily(o)
    d.to_csv("../../Data/s1_pixels/s1_field_daily.csv", index=False)
    print(f"  -> s1_field_observations.csv ({len(o)}), s1_field_daily.csv ({len(d)})\n")

    x = s2(G)
    os.makedirs("../../Data/s2_pixels", exist_ok=True)
    x.to_csv("../../Data/s2_pixels/s2_field_inputs.csv", index=False)
    print(f"  -> s2_field_inputs.csv ({len(x)})\n")

    a = alphaearth(G)
    a.to_csv("../../Data/alphaearth_field.csv", index=False)
    print(f"  -> alphaearth_field.csv ({len(a)})")
