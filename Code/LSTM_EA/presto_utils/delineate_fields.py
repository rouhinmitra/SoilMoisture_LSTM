"""
Delineate the actual field polygon for each tower from USDA CDL, and export as
GeoJSON for reuse by the S1 / S2 / AlphaEarth extractors.

WHY.  Every satellite product in this repo is currently sampled at a POINT:
  download_s1.py      reduceRegion(mean, ee.Geometry.Point, scale=10)
  download_s2.py      same
  README AlphaEarth   same
A point sample is one 10 m pixel - maximally speckle/noise exposed and not what
the flux tower integrates over.  A fixed circular buffer is no better if the
tower is off-centre in its field, which the CDL profile showed it is at Ne1
(only 89% same-class within 450 m).

METHOD.  For each year the tower's own CDL class is read, so crop rotation is
handled.  A pixel is "in the field" only if it shares the tower's class in EVERY
year - neighbouring fields on a different rotation drop out.  The connected
region containing the tower is then vectorised.  No shape is assumed, so this
works for the Ne1/Ne2 centre pivots and for Ne3 alike.

Reported per site: area, circularity 4*pi*A/P^2 (1.0 = perfect circle,
0.785 = square), and the tower's offset from the field centroid - the quantity
that makes a tower-centred buffer wrong.
"""
import json, os
import ee

PROJECT = "irriquate-backend"
OUT = "../../Data/field_boundaries.geojson"
YEARS = list(range(2017, 2024))
SITES = {"US-Ne1": [-96.4766, 41.1651],
         "US-Ne2": [-96.4701, 41.1649],
         "US-Ne3": [-96.4397, 41.1797]}


def field_for(site, coords):
    pt = ee.Geometry.Point(coords)
    inall = ee.Image(1)
    for yr in YEARS:
        cdl = ee.Image(f"USDA/NASS/CDL/{yr}").select("cropland")
        cls = cdl.reduceRegion(ee.Reducer.first(), pt, 30).get("cropland")
        inall = inall.And(cdl.eq(ee.Number(cls)))
    vec = inall.selfMask().reduceToVectors(
        geometry=pt.buffer(2000), scale=30, geometryType="polygon",
        eightConnected=False, maxPixels=1e9)
    poly = ee.Feature(vec.filterBounds(pt).first())
    g = poly.geometry()
    area = g.area(1)
    per = g.perimeter(1)
    cent = g.centroid(1)
    return dict(
        site=site,
        area_ha=area.divide(1e4).getInfo(),
        perimeter_m=per.getInfo(),
        circularity=area.multiply(4 * 3.141592653589793).divide(per.pow(2)).getInfo(),
        equiv_radius_m=area.divide(3.141592653589793).sqrt().getInfo(),
        tower_to_centroid_m=pt.distance(cent, 1).getInfo(),
        centroid=cent.coordinates().getInfo(),
        geometry=g.getInfo(),
    )


if __name__ == "__main__":
    ee.Initialize(project=PROJECT)
    print(f"Delineating fields from CDL {YEARS[0]}-{YEARS[-1]} "
          f"(pixel must match the tower's class in EVERY year)\n")
    feats, meta = [], []
    for s, c in SITES.items():
        r = field_for(s, c)
        meta.append(r)
        feats.append({"type": "Feature",
                      "properties": {k: v for k, v in r.items() if k != "geometry"},
                      "geometry": r["geometry"]})
        print(f"{s}:  area={r['area_ha']:6.1f} ha   circularity={r['circularity']:.3f}   "
              f"equiv_radius={r['equiv_radius_m']:5.0f} m   "
              f"tower offset from centroid={r['tower_to_centroid_m']:5.0f} m")
    print("\n  circularity: 1.000 = perfect circle (centre pivot), 0.785 = square")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT, "w"))
    print(f"\npolygons -> {OUT}")
