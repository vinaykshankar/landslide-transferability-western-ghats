"""Clip/reproject raw inputs to the common 30 m grid (UTM 43N) for Wayanad and Kodagu.
Usage: python3 01_clip_inputs.py wayanad|kodagu [layer ...]
Each layer is skipped if its output already exists (restartable).
"""
import sys, json, gzip, shutil, glob, os
import pandas as pd
import numpy as np, geopandas as gpd, rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString
from config import *

key = sys.argv[1]; name = DISTRICTS[key]
WK = Path(os.environ["HOME"]) / "wk" / key; WK.mkdir(parents=True, exist_ok=True)
out_dir = PROC / key; out_dir.mkdir(parents=True, exist_ok=True)

# ---------- grid ----------
adm = gpd.read_file(RAW / "boundaries" / "geoBoundaries-IND-ADM2.geojson")
aoi = adm[adm.shapeName == name].to_crs(CRS)
minx, miny, maxx, maxy = aoi.total_bounds
minx = np.floor((minx - BUFFER_M) / RES) * RES; miny = np.floor((miny - BUFFER_M) / RES) * RES
maxx = np.ceil((maxx + BUFFER_M) / RES) * RES; maxy = np.ceil((maxy + BUFFER_M) / RES) * RES
W = int((maxx - minx) / RES); H = int((maxy - miny) / RES)
T = from_origin(minx, maxy, RES, RES)
prof = dict(driver="GTiff", width=W, height=H, count=1, crs=CRS, transform=T, dtype="float32",
            nodata=np.nan, compress="deflate", tiled=True)
geo_bounds = transform_bounds(CRS, "EPSG:4326", minx, miny, maxx, maxy, densify_pts=21)
gb = (geo_bounds[0] - 0.02, geo_bounds[1] - 0.02, geo_bounds[2] + 0.02, geo_bounds[3] + 0.02)
print(name, "grid", W, "x", H, "bounds4326", [round(v, 3) for v in gb], flush=True)

def to_grid(src_arr, src_transform, src_crs, resampling, dtype="float32", src_nodata=None):
    dst = np.full((H, W), np.nan if dtype == "float32" else 0, dtype=dtype)
    reproject(src_arr, dst, src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
              dst_transform=T, dst_crs=CRS, dst_nodata=(np.nan if dtype == "float32" else 0), resampling=resampling)
    return dst

def save(arr, fname, dtype="float32"):
    p = dict(prof); p["dtype"] = dtype
    if dtype != "float32": p["nodata"] = 0
    tmp = WK / f"{fname}.tif"                      # write locally, then copy (works on sync/mounted folders)
    with rasterio.open(tmp, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)
    shutil.copyfile(tmp, out_dir / f"{fname}.tif")

def layer_mask():
    if os.environ.get("FORCE") != "1" and (out_dir / "mask.tif").exists():
        print("skip mask"); return
    # ---------- district mask ----------
    mask = rasterize([(g, 1) for g in aoi.geometry], out_shape=(H, W), transform=T, fill=0, dtype="uint8")
    save(mask, "mask", "uint8")


def layer_dem():
    if os.environ.get("FORCE") != "1" and (out_dir / "elevation.tif").exists():
        print("skip dem"); return
    # ---------- DEM ----------
    dem_files = sorted(glob.glob(str(RAW / "dem" / "*.tif")))
    srcs = [rasterio.open(f) for f in dem_files]
    mosaic, mt = merge(srcs, bounds=gb, nodata=-32767)
    dem = to_grid(mosaic[0].astype("float32"), mt, srcs[0].crs, Resampling.bilinear, src_nodata=-32767)
    [s.close() for s in srcs]; del mosaic
    save(dem, "elevation")
    print("dem done", np.nanmin(dem), np.nanmax(dem), flush=True)


def layer_roads():
    if os.environ.get("FORCE") != "1" and (out_dir / "dist_roads.tif").exists():
        print("skip roads"); return
    # ---------- roads (OSM Overpass JSON) ----------
    EXCL = {"footway", "path", "steps", "pedestrian", "cycleway", "bridleway", "corridor", "proposed",
            "construction", "platform", "elevator", "via_ferrata"}
    elements = []
    for fj in sorted(glob.glob(str(RAW / "osm" / "roads_*.json"))):
        with open(fj, encoding="utf-8") as f:
            elements += json.load(f)["elements"]
    seen = set(); lines = []
    for el in elements:
        if el.get("type") != "way" or el["id"] in seen or len(el.get("geometry", [])) < 2:
            continue
        if el.get("tags", {}).get("highway") in EXCL:
            continue
        seen.add(el["id"])
        lines.append(LineString([(n["lon"], n["lat"]) for n in el["geometry"]]))
    roads = gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(CRS)
    rr = rasterize([(g, 1) for g in roads], out_shape=(H, W), transform=T, fill=0, dtype="uint8", all_touched=True)
    save(distance_transform_edt(rr == 0).astype("float32") * RES, "dist_roads")
    print("roads done", len(lines), flush=True)
    del elements, lines, roads, rr


def layer_landcover():
    if os.environ.get("FORCE") != "1" and (out_dir / "landcover.tif").exists():
        print("skip landcover"); return
    # ---------- land cover (ESA WorldCover 2021, 10 m -> 30 m majority) ----------
    lc_files = sorted(glob.glob(str(RAW / "worldcover" / "*.tif")))
    srcs = [rasterio.open(f) for f in lc_files]
    lcm, lt = merge(srcs, bounds=gb, nodata=0)
    lc = to_grid(lcm[0], lt, srcs[0].crs, Resampling.mode, dtype="uint8", src_nodata=0)
    [s.close() for s in srcs]; del lcm
    save(lc, "landcover", "uint8")
    print("landcover done", np.unique(lc), flush=True)


def layer_rain():
    if os.environ.get("FORCE") != "1" and (out_dir / "rain_aug2018.tif").exists():
        print("skip rain"); return
    # ---------- rainfall (CHIRPS v2.0) ----------
    def read_chirps(path):
        with rasterio.open(path) as s:
            cb = (gb[0] - 0.15, gb[1] - 0.15, gb[2] + 0.15, gb[3] + 0.15)  # wider margin for 0.05 deg bilinear
            win = rasterio.windows.from_bounds(*cb, transform=s.transform).round_offsets().round_lengths()
            a = s.read(1, window=win).astype("float32"); t = s.window_transform(win)
            a[a < 0] = np.nan
            return a, t, s.crs
    ann = [read_chirps(f) for f in sorted(glob.glob(str(RAW / "chirps" / "chirps-v2.0.20[12]?.tif")))]
    print("CHIRPS annual files:", len(ann), flush=True)
    mean_ann = np.nanmean(np.stack([a for a, _, _ in ann]), axis=0)
    # fill coarse NaN cells (coast/edge) before bilinear resampling
    save(to_grid(mean_ann, ann[0][1], ann[0][2], Resampling.bilinear), "rain_annual_mean")
    gz = RAW / "chirps" / "chirps-v2.0.2018.08.tif.gz"; aug = WK / "chirps-v2.0.2018.08.tif"
    if not aug.exists():
        with gzip.open(gz, "rb") as fi, open(aug, "wb") as fo: shutil.copyfileobj(fi, fo)
    a, t, c = read_chirps(aug)
    save(to_grid(a, t, c, Resampling.bilinear), "rain_aug2018")
    print("rainfall done", flush=True)


def layer_soil():
    if os.environ.get("FORCE") != "1" and (out_dir / "soil_bdod.tif").exists():
        print("skip soil"); return
    # ---------- soil (SoilGrids 250 m, depth-weighted 0-30 cm) ----------
    wts = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
    scale = {"clay": 10.0, "sand": 10.0, "bdod": 100.0}   # -> %, %, g/cm3
    for prop in ("clay", "sand", "bdod"):
        acc_s = None
        for d, w in wts.items():
            with rasterio.open(RAW / "soilgrids" / f"{prop}_{d}_mean.tif") as s:
                a = s.read(1).astype("float32"); nd = s.nodata
                if nd is not None: a[a == nd] = np.nan
                a[a <= 0] = np.nan
                g = to_grid(a, s.transform, s.crs, Resampling.bilinear)
            acc_s = g * w if acc_s is None else acc_s + g * w
        save(acc_s / 30.0 / scale[prop], f"soil_{prop}")
    print("soil done", flush=True)


def layer_lithology():
    if os.environ.get("FORCE") != "1" and (out_dir / "lithology.tif").exists():
        print("skip lithology"); return
    # ---------- lithology (GSI 1:2M district geology, harmonised) ----------
    parts = []
    for f in glob.glob(str(RAW / "lithology" / "GEOLOGY_2M_DIST_*" / "**" / "*.shp"), recursive=True):
        parts.append(gpd.read_file(f))
    geo = gpd.GeoDataFrame(pd.concat(parts), crs=parts[0].crs).to_crs(CRS)
    geo["litho"] = geo["index_"].map(LITHO_MAP)
    unmapped = geo.loc[geo.litho.isna(), "index_"].unique()
    if len(unmapped): print("WARNING unmapped lithology:", unmapped)
    geo = geo.dropna(subset=["litho"])
    lit = rasterize([(g, LITHO_CODES[c]) for g, c in zip(geo.geometry, geo.litho)], out_shape=(H, W),
                    transform=T, fill=0, dtype="uint8")
    save(lit, "lithology", "uint8")
    m = rasterio.open(out_dir / "mask.tif").read(1)
    print("lithology done", np.unique(lit[m == 1], return_counts=True), flush=True)


LAYERS = ["mask", "dem", "roads", "landcover", "rain", "soil", "lithology"]
if __name__ == "__main__":
    todo = sys.argv[2:] or LAYERS
    for l in todo:
        globals()[f"layer_{l}"]()
    print("DONE", key, todo, flush=True)
