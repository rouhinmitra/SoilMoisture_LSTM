"""
Show the field geometries actually used, over a Sentinel-2 RGB composite.

Draws, per site: the S2 true-colour background, the field outline used for
averaging (fitted centre-pivot circle at Ne1/Ne2, CDL polygon at Ne3), the 10 m
eroded boundary that reduction is actually run over, the tower location, and the
fitted pivot centre.  A fourth panel shows all three sites together, which is
where the Ne1/Ne2 adjacency that forced the all-year CDL intersection is visible.
"""
import io, json, urllib.request
import ee
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from PIL import Image

ee.Initialize(project="irriquate-backend")
GEOJSON = "../../Data/field_boundaries.geojson"
OUT = "../outputs/model_variants/field_shapes_s2rgb.png"
SITES = {"US-Ne1": [-96.4766, 41.1651],
         "US-Ne2": [-96.4701, 41.1649],
         "US-Ne3": [-96.4397, 41.1797]}


def s2_rgb():
    """Cloud-free peak-season true-colour composite (July 2020)."""
    def mask(img):
        qa = img.select("QA60")
        k = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(k)
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2020-07-01", "2020-08-15")
            .filterBounds(ee.Geometry.Point(SITES["US-Ne1"]).buffer(6000))
            .map(mask).median().select(["B4", "B3", "B2"]))


def thumb(img, region, px=700):
    url = img.getThumbURL({"region": region, "dimensions": px,
                           "min": 200, "max": 2600, "format": "png"})
    return np.array(Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB"))


def rings(geom):
    """All exterior rings as lon/lat arrays."""
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return [np.array(c[0])]
    if t == "MultiPolygon":
        return [np.array(p[0]) for p in c]
    return []


G = json.load(open(GEOJSON))
feats = {f["properties"]["site"]: f for f in G["features"]}
rgb = s2_rgb()

fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
for ax, site in zip(axes[:3], SITES):
    f = feats[site]
    pt = ee.Geometry.Point(SITES[site])
    box = pt.buffer(800).bounds()
    coords = np.array(box.coordinates().getInfo()[0])
    lo, la = coords[:, 0], coords[:, 1]
    extent = [lo.min(), lo.max(), la.min(), la.max()]
    ax.imshow(thumb(rgb, box), extent=extent, origin="upper")

    geom = ee.Geometry(f["geometry"])
    for r in rings(f["geometry"]):
        ax.add_patch(MplPoly(r, closed=True, fill=False, ec="#ffd400", lw=2.6,
                             label="field used for averaging"))
    er = ee.Geometry(f["geometry"]).buffer(-10).getInfo()
    for r in rings(er):
        ax.add_patch(MplPoly(r, closed=True, fill=False, ec="#00e5ff", lw=1.4, ls="--",
                             label="eroded 10 m (reduction region)"))
    ax.plot(*SITES[site], marker="^", ms=13, mfc="#ff2d55", mec="w", mew=1.4,
            ls="none", label="flux tower")
    c = f["properties"].get("centre")
    if c:
        ax.plot(*c, marker="+", ms=15, mec="w", mew=2.4, ls="none", label="fitted pivot centre")
    p = f["properties"]
    ax.set_title(f"{site} — {p['kind']}\n{p['area_ha']:.1f} ha | CDL purity "
                 f"{p['purity_mean']:.3f} | tower {p['tower_offset_m']:.0f} m off-centre",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="lower left", fontsize=7.5, framealpha=0.85)

# overview
ax = axes[3]
allpts = ee.Geometry.MultiPoint(list(SITES.values()))
box = allpts.buffer(1400).bounds()
coords = np.array(box.coordinates().getInfo()[0])
extent = [coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max()]
ax.imshow(thumb(rgb, box, 900), extent=extent, origin="upper")
for site in SITES:
    for r in rings(feats[site]["geometry"]):
        ax.add_patch(MplPoly(r, closed=True, fill=False, ec="#ffd400", lw=2.2))
    ax.plot(*SITES[site], marker="^", ms=10, mfc="#ff2d55", mec="w", mew=1.2, ls="none")
    ax.annotate(site, SITES[site], textcoords="offset points", xytext=(9, 7),
                color="w", fontsize=10, weight="bold")
ax.set_title("All three sites — Ne1/Ne2 are adjacent (~550 m),\n"
             "which is why CDL merges them in same-crop years", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Field geometries used for spatial averaging, over Sentinel-2 true colour "
             "(median, 1 Jul – 15 Aug 2020, cloud-masked)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved ->", OUT)
