"""Shared configuration for the Western Ghats explainable-ML transferability study."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
CRS = "EPSG:32643"          # WGS 84 / UTM zone 43N
RES = 30.0                  # m
BUFFER_M = 1000             # grid margin around each district
STREAM_KM2 = 0.5            # contributing area threshold for stream initiation (km2)
DISTRICTS = {"wayanad": "Wayanad", "kodagu": "Kodagu"}
SEEDS = list(range(10))     # repeated negative sampling
NEG_BUFFER_M = 100          # exclusion radius around landslides for absence sampling

# harmonised lithology classes from GSI 1:2M district geology (field 'index_')
LITHO_MAP = {
    "CHARNOCKITE GNEISSIC COMPLEX (SOUTHERN GRANULITE TERRAIN)": "charnockite",
    "KHONDALITE GNEISSIC COMPLEX (SOUTHERN GRANULITE TERRAIN)": "khondalite_migmatite",
    "MIGMATITE GNEISSIC COMPLEX (SOUTHERN GRANULITE TERRAIN)": "khondalite_migmatite",
    "PENINSULAR GNEISSIC COMPLEX-I": "peninsular_gneiss",
    "WYANAD Gp.": "schist_belt",
    "SARGUR Gp.": "schist_belt",
    "SATYAMANGALAM Gp.": "schist_belt",
    "ACID  INTRUSIVE / GRANITE / GRANODIORITE": "intrusive",
    "INTERMEDIATE  INTRUSIVE": "intrusive",
    "CHAMUNDI GRANITE": "intrusive",
    "GABBRO-ANORTHOSITE COMPLEX": "intrusive",
}
LITHO_CODES = {"charnockite": 1, "khondalite_migmatite": 2, "peninsular_gneiss": 3, "schist_belt": 4, "intrusive": 5}

# ESA WorldCover 2021 classes aggregated
LC_MAP = {10: "tree", 20: "shrub_grass", 30: "shrub_grass", 40: "cropland", 50: "built_up",
          60: "bare_sparse", 70: "bare_sparse", 80: "water", 90: "wetland", 95: "wetland", 100: "bare_sparse"}

# ---- factor set / run variant ----
import os
USE_ROADS = os.environ.get("USE_ROADS", "0") == "1"
CONT = ["elevation", "slope", "northness", "eastness", "plan_curvature", "profile_curvature", "tri",
        "twi", "spi", "dist_streams", "rain_annual_mean", "rain_aug2018",
        "soil_clay", "soil_sand", "soil_bdod"] + (["dist_roads"] if USE_ROADS else [])
LC_KEEP = ["tree", "shrub_grass", "cropland", "built_up"]   # bare/sparse and wetland < 0.1% of both districts
TAG = "with_roads" if USE_ROADS else "no_roads"
OUT = ROOT / "outputs" / TAG
SAMPLES_CSV = PROC / f"samples_{TAG}.csv"
BACKGROUND_CSV = PROC / f"background_{TAG}.csv"
