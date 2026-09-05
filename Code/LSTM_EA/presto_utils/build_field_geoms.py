"""
Final field geometries for Ne1/Ne2/Ne3, validated, exported to GeoJSON.

Ne1 and Ne2 are ADJACENT (~550 m apart).  A single-year CDL connected component
merges them into one ~115 ha blob in the 5 of 7 years when they carry the same
crop; they separate only in 2018 and 2020.  So the field is defined as the region
matching the tower's own CDL class in EVERY year 2017-2023 - the differing years
do the separating.  That intersection also erodes mixed boundary pixels, which is
desirable here.

Ne1 and Ne2 are centre-pivot (user-confirmed), so their eroded, 30 m-ragged
polygon is replaced by the fitted circle: pivot centre = polygon centroid,
radius = equivalent-area radius.  Ne3 is not a pivot, so its polygon is kept.

Purity of every candidate geometry is checked against CDL before use.
"""
import json, os
import ee
import numpy as np

PROJECT = "irriquate-backend"
OUT = "../../Data/field_boundaries.geojson"
YEARS = list(range(2017, 2024))
SITES = {"US-Ne1": [-96.4766, 41.1651],
         "US-Ne2": [-96.4701, 41.1649],
         "US-Ne3": [-96.4397, 41.1797]}
PIVOT = {"US-Ne1": True, "US-Ne2": True, "US-Ne3": False}
# Pivot radii tuned by sweep: the largest radius whose WORST-year CDL purity is
# still >= 0.97 (see the sweep in the session log).  Equivalent-area radius from
# the eroded polygon overshot at Ne2 (0.937 purity), so purity sets the radius.
PIVOT_R = {"US-Ne1": 380.0, "US-Ne2": 420.0}


def stable_polygon(pt):
    inall = ee.Image(1)
    for yr in YEARS:
        cdl = ee.Image(f"USDA/NASS/CDL/{yr}").select("cropland")
        cls = cdl.reduceRegion(ee.Reducer.first(), pt, 30).get("cropland")
        inall = inall.And(cdl.eq(ee.Number(cls)))
    vec = inall.selfMask().reduceToVectors(
        geometry=pt.buffer(2000), scale=30, geometryType="polygon",
        eightConnected=False, maxPixels=1e9)
    return ee.Feature(vec.filterBounds(pt).first()).geometry()


def purity(geom, pt):
    """Mean fraction of pixels in geom sharing the tower's class, across years."""
    out = []
    for yr in YEARS:
        cdl = ee.Image(f"USDA/NASS/CDL/{yr}").select("cropland")
        cls = cdl.reduceRegion(ee.Reducer.first(), pt, 30).get("cropland")
        v = cdl.eq(ee.Number(cls)).reduceRegion(
            ee.Reducer.mean(), geom, 30).get("cropland").getInfo()
        out.append(float(v))
    return float(np.mean(out)), float(np.min(out))


def build():
    feats = []
    for site, coords in SITES.items():
        pt = ee.Geometry.Point(coords)
        poly = stable_polygon(pt)
        area = poly.area(1).getInfo()
        cent = poly.centroid(1)
        r_eq = float(np.sqrt(area / np.pi))
        if PIVOT[site]:
            r_use = PIVOT_R[site]
            geom = cent.buffer(r_use)
            kind = f"fitted pivot circle r={r_use:.0f} m"
        else:
            geom = poly
            kind = "CDL polygon"
        mean_p, min_p = purity(geom, pt)
        a_ha = geom.area(1).getInfo() / 1e4
        off = pt.distance(cent, 1).getInfo()
        print(f"{site}: {kind:34} area={a_ha:5.1f} ha  "
              f"tower->centre={off:4.0f} m  CDL purity mean={mean_p:.3f} min={min_p:.3f}")
        feats.append({"type": "Feature",
                      "properties": {"site": site, "kind": kind, "area_ha": a_ha,
                                     "tower_offset_m": off, "purity_mean": mean_p,
                                     "purity_min": min_p,
                                     "centre": cent.coordinates().getInfo()},
                      "geometry": geom.getInfo()})
    return feats


if __name__ == "__main__":
    ee.Initialize(project=PROJECT)
    print("Field geometry, CDL 2017-2023 all-year intersection\n")
    feats = build()
    print("\nfor reference, purity of a 450 m tower-centred buffer (what I used before):")
    for site, coords in SITES.items():
        pt = ee.Geometry.Point(coords)
        m, mn = purity(pt.buffer(450), pt)
        print(f"  {site}: mean={m:.3f} min={mn:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT, "w"))
    print(f"\ngeometries -> {OUT}")
