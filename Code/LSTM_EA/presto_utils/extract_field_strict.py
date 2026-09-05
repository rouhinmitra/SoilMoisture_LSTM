"""
STRICT field-averaged extraction of S1, S2 and AlphaEarth.

Guarantees that no signal comes from outside the field:
  1. GEOMETRY ERODED BY 10 m.  GEE includes a pixel when its CENTRE falls in the
     region (verified: weighted and unweighted counts are identical, 4493 = 4493,
     and the two means differ by < 0.004 dB).  A centre-inside pixel can still
     overhang the outline by up to half a pixel, so the polygon is eroded by one
     full 10 m pixel before reduction.
  2. CDL CROP MASK APPLIED.  Even inside the outline, CDL says 0-2.3% of pixels
     carry a different class at Ne1/Ne2.  Every product image is updateMask()ed
     to the all-year CDL agreement mask - the same criterion that defined the
     field - so only pixels that are the tower's own crop in EVERY year
     2017-2023 contribute.
  3. CONTAINMENT VERIFIED, not assumed: pixel centroids are sampled back and
     tested for containment in the original polygon.

Outputs go to ../../Data/field_avg/ as a NEW dataset.  Nothing existing is
overwritten.
"""
import json, os, time
import ee
import numpy as np
import pandas as pd

PROJECT = "irriquate-backend"
GEOJSON = "../../Data/field_boundaries.geojson"
OUT = "../../Data/field_avg"
START, END = "2017-01-01", "2023-12-31"
YEARS = list(range(2017, 2024))
EROSION_M = 10
S2_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','QA60']
AE_BANDS = [f"A{i:02d}" for i in range(64)]
SITES = {"US-Ne1": [-96.4766, 41.1651],
         "US-Ne2": [-96.4701, 41.1649],
         "US-Ne3": [-96.4397, 41.1797]}


def load():
    g = json.load(open(GEOJSON))
    out = {}
    for f in g["features"]:
        s = f["properties"]["site"]
        poly = ee.Geometry(f["geometry"])
        out[s] = dict(poly=poly, eroded=poly.buffer(-EROSION_M),
                      pt=ee.Geometry.Point(SITES[s]))
    return out


def cdl_mask(pt):
    """1 where the pixel is the tower's own CDL class in EVERY year."""
    m = ee.Image(1)
    for yr in YEARS:
        cdl = ee.Image(f"USDA/NASS/CDL/{yr}").select("cropland")
        cls = cdl.reduceRegion(ee.Reducer.first(), pt, 30).get("cropland")
        m = m.And(cdl.eq(ee.Number(cls)))
    return m.selfMask()


def verify(geoms, masks):
    print("CONTAINMENT VERIFICATION (pixel centroids sampled back and tested)\n")
    rows = []
    for s, g in geoms.items():
        base = ee.Image.pixelLonLat().updateMask(masks[s])
        fc = base.sample(region=g["eroded"], scale=10, geometries=True, dropNulls=True)
        n = fc.size().getInfo()
        inside = fc.filterBounds(g["poly"]).size().getInfo()
        # how many sampled centroids fall outside the ORIGINAL (un-eroded) outline
        print(f"  {s}: {n} masked pixel centroids sampled | inside original outline: "
              f"{inside} ({100*inside/max(n,1):.2f}%)")
        rows.append((s, n, inside))
    ok = all(i == n for _, n, i in rows)
    print(f"\n  ALL CENTROIDS INSIDE THE FIELD OUTLINE: {ok}\n")
    return ok


def s1(geoms, masks):
    rows = []
    for site, g in geoms.items():
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(g["poly"])
               .filterDate(START, END)
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
               .filter(ee.Filter.eq("instrumentMode", "IW")))
        m = masks[site]

        def ex(img):
            i = img.select(["VV", "VH"]).updateMask(m)
            r = i.reduceRegion(ee.Reducer.mean(), g["eroded"], 10, maxPixels=1e9)
            c = i.select("VV").reduceRegion(ee.Reducer.count(), g["eroded"], 10, maxPixels=1e9)
            return ee.Feature(None, {"Site": site, "ms": img.date().millis(),
                                     "orbit": img.get("orbitProperties_pass"),
                                     "VV": r.get("VV"), "VH": r.get("VH"), "n": c.get("VV")})
        t0 = time.time()
        fs = col.map(ex).getInfo()["features"]
        print(f"  S1 {site}: {len(fs)} images ({time.time()-t0:.0f}s)")
        for f in fs:
            p = f["properties"]
            if p.get("VV") is None:
                continue
            rows.append(dict(Site=site, ms=p["ms"], orbit=p.get("orbit"),
                             VV=float(p["VV"]), VH=float(p["VH"]), n_pixels=int(p["n"] or 0)))
    d = pd.DataFrame(rows)
    d["Date"] = pd.to_datetime(d.ms, unit="ms").dt.normalize()
    d["VV_VH_diff"] = d.VV - d.VH
    return d.drop(columns=["ms"])


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


def s2(geoms, masks):
    def cloudmask(img):
        qa = img.select("QA60")
        k = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(k).select(S2_BANDS).copyProperties(img, ["system:time_start"])
    rows = []
    for site, g in geoms.items():
        m = masks[site]
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(g["poly"])
               .filterDate(START, END).map(cloudmask))

        def ex(img):
            i = ee.Image(img).updateMask(m)
            r = i.reduceRegion(ee.Reducer.mean(), g["eroded"], 10, maxPixels=1e9)
            c = i.select("B4").reduceRegion(ee.Reducer.count(), g["eroded"], 10, maxPixels=1e9)
            return ee.Feature(None, {"Site": site, "ms": img.get("system:time_start"),
                                     "n": c.get("B4"), **{b: r.get(b) for b in S2_BANDS}})
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
    d = pd.DataFrame(rows)
    d["Date"] = pd.to_datetime(d.ms, unit="ms").dt.normalize()
    return d.drop(columns=["ms"])


def alphaearth(geoms, masks):
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    rows = []
    for yr in YEARS:
        for site, g in geoms.items():
            img = (col.filter(ee.Filter.calendarRange(yr, yr, "year"))
                     .filterBounds(g["poly"]).first().updateMask(masks[site]))
            v = img.select(AE_BANDS).reduceRegion(
                ee.Reducer.mean(), g["eroded"], 10, maxPixels=1e9).getInfo()
            n = img.select("A00").reduceRegion(
                ee.Reducer.count(), g["eroded"], 10, maxPixels=1e9).getInfo()["A00"]
            rows.append({"Site": site, "Year": yr, "n_pixels": int(n), **v})
        print(f"  AE {yr}: done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ee.Initialize(project=PROJECT)
    os.makedirs(OUT, exist_ok=True)
    G = load()
    M = {s: cdl_mask(g["pt"]) for s, g in G.items()}
    for s, g in G.items():
        print(f"{s}: polygon {g['poly'].area(1).getInfo()/1e4:.1f} ha -> "
              f"eroded {g['eroded'].area(1).getInfo()/1e4:.1f} ha")
    print()
    if not verify(G, M):
        raise SystemExit("ABORT: some sampled pixel centroids fall outside the field outline.")

    o = s1(G, M); o.to_csv(f"{OUT}/s1_field_observations.csv", index=False)
    d = interp_daily(o); d.to_csv(f"{OUT}/s1_field_daily.csv", index=False)
    print(f"  -> s1 obs {len(o)}, daily {len(d)}\n")
    x = s2(G, M); x.to_csv(f"{OUT}/s2_field_inputs.csv", index=False)
    print(f"  -> s2 {len(x)}\n")
    a = alphaearth(G, M); a.to_csv(f"{OUT}/alphaearth_field.csv", index=False)
    print(f"  -> alphaearth {len(a)}")
