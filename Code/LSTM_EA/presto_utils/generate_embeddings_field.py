"""
Generate Presto embeddings from FIELD-AVERAGED S1 + S2 (or, with --orig, from the
original point-based inputs, to validate that this environment reproduces the
published embeddings before trusting the field version).

Mirrors generate_embeddings.py exactly - same model, same band lists, same daily
aggregation, same outer join, same per-row encoder call, same interpolation - so
the ONLY difference in the field run is the spatial support of the inputs.

The field S2 CSV carries no lat/lon (the point extractor got them for free), so
the Presto latlon input is taken from the field centroid in field_boundaries.geojson.

  python generate_embeddings_field.py --orig    # reproduce published embeddings
  python generate_embeddings_field.py           # field-averaged version
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from presto import Presto, construct_single_presto_input

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
S2_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
S1_BANDS = ['VV', 'VH']
GEOJSON = "../../Data/field_boundaries.geojson"


def centroids():
    g = json.load(open(GEOJSON))
    return {f["properties"]["site"]: f["properties"]["centre"] for f in g["features"]}


def load(s2_path, s1_path, latlon_from_geojson):
    s2 = pd.read_csv(s2_path)
    s2["Date"] = pd.to_datetime(s2["Date"]).dt.normalize()
    if latlon_from_geojson:
        c = centroids()
        s2["longitude"] = s2.Site.map(lambda s: c[s][0])
        s2["latitude"] = s2.Site.map(lambda s: c[s][1])
    s2 = s2.groupby(["Site", "Date"])[S2_BANDS + ["latitude", "longitude"]].mean().reset_index()
    print(f"  S2: {len(s2)} daily rows")

    s1 = pd.read_csv(s1_path)
    s1["Date"] = pd.to_datetime(s1["Date"]).dt.normalize()
    s1 = s1.groupby(["Site", "Date"])[S1_BANDS].mean().reset_index()
    print(f"  S1: {len(s1)} daily rows")

    m = pd.merge(s2, s1, on=["Site", "Date"], how="outer")
    for c in ("latitude", "longitude"):
        m[c] = m.groupby("Site")[c].transform(lambda x: x.ffill().bfill())
    m = m.dropna(subset=["latitude", "longitude"]).sort_values(["Site", "Date"])
    print(f"  fused timeline: {len(m)} rows")
    return m


def embed(df, model):
    out = []
    for site, sdf in df.groupby("Site"):
        embs = []
        for i in range(len(sdf)):
            row = sdf.iloc[[i]]
            s2t = (torch.tensor(row[S2_BANDS].values, dtype=torch.float32)
                   if pd.notna(row["B1"].values[0]) else None)
            s1t = (torch.tensor(row[S1_BANDS].values, dtype=torch.float32)
                   if pd.notna(row["VV"].values[0]) else None)
            x, mask, dw = construct_single_presto_input(
                s2=s2t, s2_bands=S2_BANDS if s2t is not None else None,
                s1=s1t, s1_bands=S1_BANDS if s1t is not None else None)
            with torch.no_grad():
                e = model.encoder(x.unsqueeze(0), dw.long().unsqueeze(0),
                                  torch.tensor([[row["latitude"].iloc[0],
                                                 row["longitude"].iloc[0]]]).float(),
                                  mask.unsqueeze(0),
                                  torch.tensor([row["Date"].iloc[0].month - 1]).long())
            embs.append(e.reshape(1, -1).numpy())
            if i % 200 == 0:
                sys.stdout.write(f"\r  {site}: {i}/{len(sdf)}"); sys.stdout.flush()
        print(f"\r  {site}: {len(sdf)}/{len(sdf)} done")
        a = np.concatenate(embs, axis=0)
        d = pd.DataFrame(a, columns=[f"emb_{k}" for k in range(a.shape[1])])
        d["Date"] = sdf["Date"].values; d["Site"] = site
        out.append(d)
    return pd.concat(out, ignore_index=True)


def interpolate(df):
    df["Date"] = pd.to_datetime(df["Date"])
    cols = [c for c in df.columns if c.startswith("emb_")]
    chunks = []
    for site in df.Site.unique():
        s = df[df.Site == site]
        for yr in sorted(s.Date.dt.year.unique()):
            days = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D")
            m = pd.merge(pd.DataFrame({"Date": days}), s, on="Date", how="left")
            m["Site"] = site
            m[cols] = m[cols].interpolate(method="linear", limit_direction="both")
            chunks.append(m)
    return pd.concat(chunks, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", action="store_true",
                    help="use the original point-based inputs (validation run)")
    a = ap.parse_args()

    if a.orig:
        s2p, s1p = "../../Data/s2_pixels/s2_presto_inputs.csv", "../../Data/s1_pixels/s1_presto_inputs.csv"
        outdir, tag, latlon = "../../Data/field_avg", "REPRO_point", False
    else:
        s2p, s1p = "../../Data/field_avg/s2_field_inputs.csv", "../../Data/field_avg/s1_field_observations.csv"
        outdir, tag, latlon = "../../Data/field_avg", "field", True
    os.makedirs(outdir, exist_ok=True)
    print(f"Loading Presto...")
    model = Presto.construct()
    model.load_state_dict(torch.load("default_model.pt", map_location="cpu", weights_only=False))
    model.eval()
    df = load(s2p, s1p, latlon)
    emb = embed(df, model)
    emb.to_csv(f"{outdir}/presto_embeddings_{tag}.csv", index=False)
    itp = interpolate(emb)
    itp.to_csv(f"{outdir}/presto_embeddings_{tag}_interpolated.csv", index=False)
    print(f"\n-> {outdir}/presto_embeddings_{tag}.csv ({len(emb)})")
    print(f"-> {outdir}/presto_embeddings_{tag}_interpolated.csv ({len(itp)})")
