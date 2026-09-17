"""Presence/absence sampling and background samples for both districts.
Usage: python3 03_sample.py
Outputs: data/processed/samples.csv, data/processed/background.csv, data/processed/inventory_summary.json
"""
import json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.ndimage import distance_transform_edt
from rasterio.features import rasterize
from config import *

LITHO = list(LITHO_CODES.keys())

def load_stack(key):
    d = PROC / key
    with rasterio.open(d / "mask.tif") as s:
        mask = s.read(1) == 1; T = s.transform
    arrs = {}
    for f in CONT:
        with rasterio.open(d / f"{f}.tif") as s: arrs[f] = s.read(1).astype("float32")
    with rasterio.open(d / "landcover.tif") as s: lc = s.read(1)
    with rasterio.open(d / "lithology.tif") as s: lit = s.read(1)
    # fill unmapped lithology cells (district-edge slivers) with nearest mapped class
    if (lit == 0).any():
        idx = distance_transform_edt(lit == 0, return_distances=False, return_indices=True)
        lit = lit[tuple(idx)]
    return mask, T, arrs, lc, lit

def features_at(rows, cols, arrs, lc, lit):
    df = pd.DataFrame({f: arrs[f][rows, cols] for f in CONT})
    lcv = pd.Series(lc[rows, cols]).map(LC_MAP).fillna("other")
    for c in LC_KEEP: df[f"lc_{c}"] = (lcv == c).astype(int)
    inv = {v: k for k, v in LITHO_CODES.items()}
    litv = pd.Series(lit[rows, cols]).map(inv)
    for c in LITHO: df[f"lith_{c}"] = (litv == c).astype(int)
    df["lc_class"] = lcv.values; df["lith_class"] = litv.values
    return df

adm = gpd.read_file(RAW / "boundaries" / "geoBoundaries-IND-ADM2.geojson")
inv_sum = {}
samples, background = [], []
for key, name in DISTRICTS.items():
    mask, T, arrs, lc, lit = load_stack(key)
    H, W = mask.shape
    poly = adm[adm.shapeName == name].to_crs(CRS)
    if key == "wayanad":
        k = gpd.read_file(RAW / "kerala_inventory" / "Kerela landslide.shp").to_crs(CRS)
        n_all = int(gpd.sjoin(k, poly[["geometry"]], predicate="within").shape[0])
        pts = gpd.sjoin(k, poly[["geometry"]], predicate="within")
        types = pts["Type_of_sl"].value_counts().to_dict()
        pts = pts[pts["Type_of_sl"] != "RF"]          # rockfalls excluded (absent from Kodagu inventory)
        excl_geoms = list(k.geometry)
    else:
        pts = gpd.read_file(RAW / "kodagu_inventory" / "Landslide_Points.shp").to_crs(CRS)
        n_all = len(pts); types = {"points": n_all}
        polys = gpd.read_file(RAW / "kodagu_inventory" / "Landslide_polygons.shp").to_crs(CRS)
        excl_geoms = list(pts.geometry) + list(polys.geometry)
    inv = ~T
    cols, rows = inv * (pts.geometry.x.values, pts.geometry.y.values)
    rows = np.floor(rows).astype(int); cols = np.floor(cols).astype(int)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols = rows[ok], cols[ok]
    cells = pd.DataFrame({"r": rows, "c": cols}).drop_duplicates()
    rows, cols = cells.r.values, cells.c.values
    in_mask = mask[rows, cols]
    rows, cols = rows[in_mask], cols[in_mask]
    finite = np.all([np.isfinite(arrs[f][rows, cols]) for f in CONT], axis=0)
    rows, cols = rows[finite], cols[finite]
    n_pos = len(rows)
    inv_sum[key] = {"points_in_district": n_all, "types": {str(a): int(b) for a, b in types.items()},
                    "unique_cells_used": int(n_pos)}
    print(key, inv_sum[key], flush=True)

    # exclusion zone around all mapped landslides
    ls = rasterize([(g, 1) for g in excl_geoms], out_shape=(H, W), transform=T, fill=0, dtype="uint8", all_touched=True)
    dist = distance_transform_edt(ls == 0) * RES
    valid = mask & (dist > NEG_BUFFER_M) & (lc != 80) & np.all([np.isfinite(arrs[f]) for f in CONT], axis=0)
    vr, vc = np.nonzero(valid)
    inv_sum[key]["candidate_absence_cells"] = int(len(vr))
    inv_sum[key]["district_cells"] = int(mask.sum())

    pos = features_at(rows, cols, arrs, lc, lit)
    xs, ys = T * (cols + 0.5, rows + 0.5)
    pos["x"], pos["y"], pos["label"] = xs, ys, 1
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        pick = rng.choice(len(vr), size=n_pos, replace=False)
        neg = features_at(vr[pick], vc[pick], arrs, lc, lit)
        nx, ny = T * (vc[pick] + 0.5, vr[pick] + 0.5)
        neg["x"], neg["y"], neg["label"] = nx, ny, 0
        both = pd.concat([pos, neg], ignore_index=True)
        both["district"], both["seed"] = key, seed
        samples.append(both)
    # background sample for covariate-shift analysis (random district cells)
    rng = np.random.default_rng(7)
    mr, mc = np.nonzero(mask & np.all([np.isfinite(arrs[f]) for f in CONT], axis=0))
    pick = rng.choice(len(mr), size=min(20000, len(mr)), replace=False)
    bg = features_at(mr[pick], mc[pick], arrs, lc, lit); bg["district"] = key
    background.append(bg)
    inv_sum[key]["lithology_share"] = pd.Series(lit[mask]).map({v: k for k, v in LITHO_CODES.items()}).value_counts(normalize=True).round(4).to_dict()
    inv_sum[key]["landcover_share"] = pd.Series(lc[mask]).map(LC_MAP).value_counts(normalize=True).round(4).to_dict()

pd.concat(samples).to_csv(SAMPLES_CSV, index=False)
pd.concat(background).to_csv(BACKGROUND_CSV, index=False)
OUT.mkdir(parents=True, exist_ok=True); json.dump(inv_sum, open(OUT / "inventory_summary.json", "w"), indent=2)
print("samples written")
