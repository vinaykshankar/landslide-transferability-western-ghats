"""Susceptibility maps (seed-ensemble XGBoost) and class-wise landslide statistics.
Usage: python3 05_maps.py
Outputs: outputs/maps/*.tif, outputs/tables/class_statistics.csv
"""
import json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.ndimage import distance_transform_edt
from xgboost import XGBClassifier
from config import *

S = pd.read_csv(SAMPLES_CSV)
BG = pd.read_csv(BACKGROUND_CSV)
shift = json.load(open(OUT / "tables" / "shift_summary.json"))
LITHO = list(LITHO_CODES.keys())
ONEHOT = [f"lc_{c}" for c in LC_KEEP] + [f"lith_{c}" for c in LITHO]
FEATS = CONT + ONEHOT
MAPS = OUT / "maps"; MAPS.mkdir(parents=True, exist_ok=True)


def xgb(seed):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=2, reg_lambda=1.0, n_jobs=2, random_state=seed, eval_metric="logloss")


def load_grid(key):
    d = PROC / key
    with rasterio.open(d / "mask.tif") as s:
        mask = s.read(1) == 1; prof = s.profile
    arrs = {f: rasterio.open(d / f"{f}.tif").read(1).astype("float32") for f in CONT}
    lc = rasterio.open(d / "landcover.tif").read(1)
    lit = rasterio.open(d / "lithology.tif").read(1)
    if (lit == 0).any():
        idx = distance_transform_edt(lit == 0, return_distances=False, return_indices=True)
        lit = lit[tuple(idx)]
    valid = mask & np.all([np.isfinite(arrs[f]) for f in CONT], axis=0)
    r, c = np.nonzero(valid)
    X = pd.DataFrame({f: arrs[f][r, c] for f in CONT})
    lcv = pd.Series(lc[r, c]).map(LC_MAP).fillna("other")
    for k in LC_KEEP: X[f"lc_{k}"] = (lcv == k).astype("int8").values
    inv = {v: k for k, v in LITHO_CODES.items()}
    litv = pd.Series(lit[r, c]).map(inv)
    for k in LITHO: X[f"lith_{k}"] = (litv == k).astype("int8").values
    return mask, prof, r, c, X


def zs(X, region, feats):
    X = X.copy()
    for f in [c for c in feats if c in CONT]:
        b = BG.loc[BG.district == region, f]
        X[f] = (X[f] - b.mean()) / b.std()
    return X


def predict_map(train_region, target_region, feats, zscore, tag):
    mask, prof, r, c, X = load_grid(target_region)
    Xt = zs(X, target_region, feats) if zscore else X
    prob = np.zeros(len(X), dtype="float64")
    for seed in SEEDS:
        d = S[(S.district == train_region) & (S.seed == seed)]
        Xtr = zs(d, train_region, feats) if zscore else d
        m = xgb(seed).fit(Xtr[feats], d.label)
        for i in range(0, len(X), 500000):
            prob[i:i + 500000] += m.predict_proba(Xt[feats].iloc[i:i + 500000])[:, 1]
    prob /= len(SEEDS)
    out = np.full(mask.shape, np.nan, dtype="float32"); out[r, c] = prob
    p = dict(prof); p.update(dtype="float32", nodata=np.nan, compress="deflate", tiled=True)
    with rasterio.open(MAPS / f"susceptibility_{tag}.tif", "w", **p) as dst:
        dst.write(out, 1)
    print("map", tag, "done", flush=True)
    return out


def class_stats(prob, target_region, tag):
    """Equal-area quintile classes; share of landslide cells per class."""
    pos = S[(S.district == target_region) & (S.seed == 0) & (S.label == 1)]
    with rasterio.open(MAPS / f"susceptibility_{tag}.tif") as s:
        T = s.transform
    cols, rows = (~T) * (pos.x.values, pos.y.values)
    pv = prob[rows.astype(int), cols.astype(int)]
    vals = prob[np.isfinite(prob)]
    q = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    names = ["Very low", "Low", "Moderate", "High", "Very high"]
    cls_all = np.digitize(vals, q); cls_pos = np.digitize(pv, q)
    rows_out = []
    for k, n in enumerate(names):
        area = (cls_all == k).mean() * 100; ls = (cls_pos == k).mean() * 100
        rows_out.append(dict(map=tag, target=target_region, cls=n, prob_lower=(0 if k == 0 else q[k - 1]),
                             area_pct=area, landslide_pct=ls, FR=ls / area))
    return rows_out


if __name__ == "__main__":
    shifted = shift["shifted_features"]
    perf = pd.read_csv(OUT / "tables" / "performance_summary.csv")
    t = perf[(perf.scheme == "transfer") & (perf.train == "wayanad") & (perf.model == "XGB") & (perf.strategy != "S1_all")]
    best = t.sort_values("AUC_mean", ascending=False).iloc[0]
    json.dump({"tag": f"kodagu_from_wayanad_{best.strategy[:2]}", "strategy": best.strategy, "AUC_mean": best.AUC_mean},
              open(OUT / "tables" / "best_transfer_strategy.json", "w"), indent=2)
    rows = []
    jobs = [("wayanad", "wayanad", FEATS, False, "wayanad_local"),
            ("kodagu", "kodagu", FEATS, False, "kodagu_local"),
            ("wayanad", "kodagu", FEATS, False, "kodagu_from_wayanad_S1"),
            ("wayanad", "kodagu", FEATS, True, "kodagu_from_wayanad_S2"),
            ("wayanad", "kodagu", [f for f in FEATS if f not in shifted], False, "kodagu_from_wayanad_S3"),
            ("wayanad", "kodagu", [f for f in FEATS if not f.startswith("rain_")], False, "kodagu_from_wayanad_S4"),
            ("wayanad", "wayanad", [f for f in FEATS if not f.startswith("rain_")], False, "wayanad_local_S4")]
    for tr, tg, feats, z, tag in jobs:
        prob = predict_map(tr, tg, feats, z, tag)
        rows += class_stats(prob, tg, tag)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "class_statistics.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
