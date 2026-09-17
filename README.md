# Landslide susceptibility transferability in the Western Ghats (Wayanad ↔ Kodagu)

Code and outputs for the manuscript *"What transfers and what does not: covariate shift, SHAP explanation stability and cross-district transferability of machine-learning landslide susceptibility models in the Western Ghats, India"* (prepared for the Frontiers Research Topic *Applications of Geocomputation Methods in Geohazard Studies*).

## Folder layout

```
code/            analysis pipeline (run in numeric order)
data/raw/        downloaded open data (not redistributed; see download_data.ps1 / MANUAL_DOWNLOADS.txt)
data/processed/  30 m factor rasters per district, sample tables
outputs/no_roads/    main analysis (15 continuous factors + land cover + lithology)
outputs/with_roads/  sensitivity analysis with distance to OpenStreetMap roads
manuscript/      manuscript .docx, supplementary .docx, text sources and table JSON
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_extract_osm_roads.py` | Extracts motorable roads for the study box from the Geofabrik South India `.osm.pbf` |
| 1 | `01_clip_inputs.py <district> [layers]` | Builds the common 30 m UTM 43N grid; clips/reprojects DEM, land cover, CHIRPS, SoilGrids, GSI lithology, roads |
| 2 | `02_terrain_hydrology.py <district>` | Slope, aspect, curvature, TRI, TWI, SPI, distance to streams |
| 3 | `03_sample.py` | Landslide cells, 10 repeated 1:1 absence samples (>100 m from landslides), background samples |
| 4 | `04_models.py` | LR/RF/XGBoost/LightGBM; random CV, 5 km spatial block CV, transfer both directions; covariate shift; SHAP; strategies S1–S4 |
| 5 | `05_maps.py` | Ten-model XGBoost ensemble susceptibility maps; quintile class statistics |
| 6 | `06_figures.py` | Figures 1–8 |
| 7 | `07_supplementary_stats.py` | Spatial CV without rainfall, paired Wilcoxon tests, VIF, SHAP effect agreement |
| 8 | `08_manuscript_tables.py` | Builds manuscript tables/numbers directly from outputs |
| – | `manuscript/build_docx.js` | Assembles the Word manuscript (Node.js, `docx` package) |

Set `USE_ROADS=1` before steps 3–7 to run the roads sensitivity variant (outputs go to `outputs/with_roads`).

Example (Linux/macOS/WSL):

```bash
pip install -r requirements.txt
cd code
python 00_extract_osm_roads.py
for d in wayanad kodagu; do python 01_clip_inputs.py $d; python 02_terrain_hydrology.py $d; done
python 03_sample.py && python 04_models.py && python 05_maps.py && python 06_figures.py && python 07_supplementary_stats.py
USE_ROADS=1 python 03_sample.py && USE_ROADS=1 python 04_models.py && USE_ROADS=1 python 05_maps.py && USE_ROADS=1 python 06_figures.py
python 08_manuscript_tables.py
```

## Data sources (all open)

- Kerala 2018 landslide inventory — van Westen (2020), DANS, doi:10.17026/dans-x6c-y7x2 (CC BY 4.0)
- Kodagu 2018 landslide inventory — Arpitha et al. (2024), GitHub (CC BY-SA 4.0)
- Copernicus GLO-30 DEM — doi:10.5270/ESA-c5d3d65
- CHIRPS v2.0 — Funk et al. (2015)
- SoilGrids 250 m (WCS) — Poggio et al. (2021)
- ESA WorldCover 2021 v200 — doi:10.5281/zenodo.7254221
- GSI district geology 1:2M — Bhukosh portal
- geoBoundaries ADM2 — Runfola et al. (2020)
- OpenStreetMap (Geofabrik southern-zone extract)

## Notes

- Hyperparameters are fixed (no tuning) so that all validation designs are directly comparable.
- Susceptibility maps of the training district are fitted maps, not independent tests; only transferred maps are validation.
- `pysheds` with NumPy ≥ 2 needs the `np.in1d = np.isin` shim included in `02_terrain_hydrology.py`.
