"""Terrain derivatives (Zevenbergen & Thorne 1987) and hydrological indices from the 30 m DEM.
Usage: python3 02_terrain_hydrology.py wayanad|kodagu
"""
import sys, os
import numpy as np, rasterio
from scipy.ndimage import distance_transform_edt
from config import *

key = sys.argv[1]
out_dir = PROC / key
WK = Path(os.environ.get("WK", Path.home() / "wk")) / key; WK.mkdir(parents=True, exist_ok=True)
with rasterio.open(out_dir / "elevation.tif") as s:
    dem = s.read(1).astype("float32"); prof = s.profile
prof.update(dtype="float32", nodata=np.nan, compress="deflate", tiled=True)
H, W = dem.shape

def save(arr, fname, dtype="float32"):
    p = dict(prof); p["dtype"] = dtype
    if dtype != "float32": p["nodata"] = 0
    with rasterio.open(out_dir / f"{fname}.tif", "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)

# ---------- terrain derivatives (Zevenbergen & Thorne 1987) ----------
L = RES
Zp = np.pad(dem, 1, mode="edge")
Z1, Z2, Z3 = Zp[:-2, :-2], Zp[:-2, 1:-1], Zp[:-2, 2:]
Z4, Z5, Z6 = Zp[1:-1, :-2], Zp[1:-1, 1:-1], Zp[1:-1, 2:]
Z7, Z8, Z9 = Zp[2:, :-2], Zp[2:, 1:-1], Zp[2:, 2:]
D = ((Z4 + Z6) / 2 - Z5) / L**2
E = ((Z2 + Z8) / 2 - Z5) / L**2
F = (-Z1 + Z3 + Z7 - Z9) / (4 * L**2)
G = (Z6 - Z4) / (2 * L)
Hh = (Z2 - Z8) / (2 * L)
p2 = G**2 + Hh**2
slope_rad = np.arctan(np.sqrt(p2))
save(np.degrees(slope_rad), "slope")
az = np.arctan2(-G, -Hh)
save(np.cos(az), "northness"); save(np.sin(az), "eastness")
with np.errstate(invalid="ignore", divide="ignore"):
    prof_c = np.where(p2 > 1e-10, -2 * (D * G**2 + E * Hh**2 + F * G * Hh) / p2, 0) * 100
    plan_c = np.where(p2 > 1e-10, 2 * (D * Hh**2 + E * G**2 - F * G * Hh) / p2, 0) * 100
save(prof_c, "profile_curvature"); save(plan_c, "plan_curvature")
tri = np.sqrt(sum((z - Z5) ** 2 for z in (Z1, Z2, Z3, Z4, Z6, Z7, Z8, Z9)))
save(tri, "tri")
del Zp, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, D, E, F, G, Hh, prof_c, plan_c, tri, az
print("terrain done", flush=True)

# ---------- hydrology (pysheds) ----------
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # compatibility: pysheds with NumPy >= 2.0
from pysheds.grid import Grid
dem_path = WK / "dem_utm.tif"
p = dict(prof); p["nodata"] = -9999
with rasterio.open(dem_path, "w", **p) as dst:
    dst.write(np.nan_to_num(dem, nan=-9999).astype("float32"), 1)
grid = Grid.from_raster(str(dem_path))
rd = grid.read_raster(str(dem_path))
rd = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(rd)))
fdir = grid.flowdir(rd)
acc = np.asarray(grid.accumulation(fdir), dtype="float32")
del rd, fdir, grid
sca = (acc + 1) * L
tanb = np.maximum(np.tan(slope_rad), 0.001)
save(np.log(sca / tanb), "twi")
save(np.log(sca * tanb + 1), "spi")
streams = acc * L * L / 1e6 >= STREAM_KM2
save(distance_transform_edt(~streams).astype("float32") * L, "dist_streams")
save(streams.astype("uint8") + 0, "streams", "uint8")
del acc, sca, tanb, streams
print("hydrology done", flush=True)

print("ALL DONE", key, flush=True)
