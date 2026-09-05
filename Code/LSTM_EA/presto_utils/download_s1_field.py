"""
Sentinel-1 VV/VH averaged over the FIELD (not a single pixel), then linearly
interpolated to daily.

WHY THIS EXISTS.  download_s1.py reduces with ee.Geometry.Point(coords) at
scale=10, which returns the ONE 10 m pixel containing the tower - maximally
exposed to speckle, and not representative of the field the flux tower
integrates over.  This averages over a buffer instead, which suppresses speckle
by roughly sqrt(N_pixels).

BACKSCATTER, NOT REFLECTANCE.  S1 is active radar; VV/VH are backscatter
coefficients in dB.  (S2 is the reflectance product.)

FOUR RADII are extracted so the choice is auditable rather than assumed:
    30 m   ~ 28 pixels   - close to the existing point sample
    100 m  ~ 314 px
    250 m  ~ 1963 px
    450 m  ~ 6362 px     - ~64 ha, the nominal size of these Mead fields
No field boundary files exist in the repo, so a circular buffer on the tower
coordinate is the approximation used; 450 m is reported as the primary
"field average" and the others show sensitivity.

ORBIT PASS IS PRESERVED.  download_s1.py fetched orbitProperties_pass and then
dropped it, averaging ascending and descending passes together.  Those have
different incidence angles and systematically different backscatter, so they are
kept separate here and the combining is left explicit and visible.

Outputs (Code/Data/s1_pixels/):
    s1_field_observations.csv   one row per site x date x radius x orbit
    s1_field_daily.csv          linearly interpolated to daily, per site x radius
"""
import os, sys, time
import ee
import numpy as np
import pandas as pd

PROJECT = "irriquate-backend"       # the project these credentials are authorised for
OUT_DIR = "../../Data/s1_pixels"
START, END = "2017-01-01", "2023-12-31"
RADII = [30, 100, 250, 450]
PRIMARY_RADIUS = 450
SITES = {"US-Ne1": [-96.4766, 41.1651],
         "US-Ne2": [-96.4701, 41.1649],
         "US-Ne3": [-96.4397, 41.1797]}


def init():
    ee.Initialize(project=PROJECT)
    print(f"Earth Engine initialised (project={PROJECT})")


def fetch():
    rows = []
    for site, coords in SITES.items():
        pt = ee.Geometry.Point(coords)
        col = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(pt).filterDate(START, END)
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
               .filter(ee.Filter.eq("instrumentMode", "IW")))

        def extract(img):
            img2 = img.select(["VV", "VH"])
            props = {"Site": site,
                     "Date_Millis": img.date().millis(),
                     "orbit": img.get("orbitProperties_pass"),
                     "rel_orbit": img.get("relativeOrbitNumber_start")}
            for r in RADII:
                geom = pt.buffer(r)
                mean = img2.reduceRegion(ee.Reducer.mean(), geom, 10)
                cnt = img2.select("VV").reduceRegion(ee.Reducer.count(), geom, 10)
                props[f"VV_{r}"] = mean.get("VV")
                props[f"VH_{r}"] = mean.get("VH")
                props[f"n_{r}"] = cnt.get("VV")
            return ee.Feature(None, props)

        t0 = time.time()
        feats = col.map(extract).getInfo()["features"]
        print(f"  {site}: {len(feats)} images in {time.time()-t0:.0f}s")
        for f in feats:
            p = f["properties"]
            for r in RADII:
                if p.get(f"VV_{r}") is None or p.get(f"VH_{r}") is None:
                    continue
                rows.append(dict(Site=site, Date_Millis=p["Date_Millis"],
                                 orbit=p.get("orbit"), rel_orbit=p.get("rel_orbit"),
                                 radius_m=r, VV=float(p[f"VV_{r}"]),
                                 VH=float(p[f"VH_{r}"]), n_pixels=int(p[f"n_{r}"] or 0)))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df.Date_Millis, unit="ms").dt.normalize()
    return df.drop(columns=["Date_Millis"])


def interpolate(obs):
    """Linear interpolation to daily, per site x radius, with provenance flags."""
    full = pd.date_range(START, END, freq="D")
    out = []
    for (site, r), g in obs.groupby(["Site", "radius_m"]):
        # average any same-day passes AFTER the per-orbit record is preserved upstream
        daily = g.groupby("Date")[["VV", "VH"]].mean().reindex(full)
        obs_mask = daily.VV.notna()
        interp = daily.interpolate(method="linear", limit_direction="both")
        idx = np.arange(len(full))
        obs_idx = idx[obs_mask.values]
        gap = (np.abs(idx[:, None] - obs_idx[None, :]).min(axis=1)
               if len(obs_idx) else np.full(len(idx), -1))
        out.append(pd.DataFrame({"Site": site, "radius_m": r, "Date": full,
                                 "VV": interp.VV.values, "VH": interp.VH.values,
                                 "is_observed": obs_mask.values,
                                 "days_to_nearest_obs": gap}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    init()
    os.makedirs(OUT_DIR, exist_ok=True)
    obs = fetch()
    obs.to_csv(f"{OUT_DIR}/s1_field_observations.csv", index=False)
    print(f"\nobservations -> {OUT_DIR}/s1_field_observations.csv  ({len(obs)} rows)")
    daily = interpolate(obs)
    daily.to_csv(f"{OUT_DIR}/s1_field_daily.csv", index=False)
    print(f"daily        -> {OUT_DIR}/s1_field_daily.csv  ({len(daily)} rows)")
