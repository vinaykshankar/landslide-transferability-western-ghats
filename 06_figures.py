"""Publication figures (300 dpi PNG) for the manuscript.
Usage: python3 06_figures.py [fig1 fig2 ...]
"""
import sys, json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from config import *

FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C_W, C_K = "#2a78d6", "#eb6834"                       # Wayanad, Kodagu (categorical slots 1-2)
MODEL_C = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a", "LGBM": "#eda100"}
INK2 = "#52514e"
LABELS = {"elevation": "Elevation", "slope": "Slope", "northness": "Northness", "eastness": "Eastness",
          "plan_curvature": "Plan curvature", "profile_curvature": "Profile curvature", "tri": "TRI", "twi": "TWI",
          "spi": "SPI", "dist_streams": "Distance to streams", "dist_roads": "Distance to roads",
          "rain_annual_mean": "Mean annual rainfall", "rain_aug2018": "August 2018 rainfall",
          "soil_clay": "Soil clay", "soil_sand": "Soil sand", "soil_bdod": "Soil bulk density",
          "landcover": "Land cover", "lithology": "Lithology"}
UNITS = {"elevation": "m", "slope": "°", "dist_streams": "m", "dist_roads": "m", "rain_annual_mean": "mm yr⁻¹",
         "rain_aug2018": "mm", "soil_clay": "%", "soil_sand": "%", "soil_bdod": "g cm⁻³", "tri": "m"}


def read(key, name):
    with rasterio.open(PROC / key / f"{name}.tif") as s:
        return s.read(1), s.transform, s.bounds


def fig1():
    adm = gpd.read_file(RAW / "boundaries" / "geoBoundaries-IND-ADM2.geojson")
    dist = adm[adm.shapeName.isin(["Wayanad", "Kodagu"])]
    S = pd.read_csv(SAMPLES_CSV); pos = S[(S.seed == 0) & (S.label == 1)]
    fig = plt.figure(figsize=(7.4, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.15, 1.05], wspace=0.38)
    ax0 = fig.add_subplot(gs[0])
    south = adm.cx[73.5:80.5, 8:15.5]          # southern peninsula districts only (context map)
    south.boundary.plot(ax=ax0, color="#b9b8b2", lw=0.25)
    dist.plot(ax=ax0, color=[C_K if n == "Kodagu" else C_W for n in dist.shapeName], edgecolor="none")
    ax0.set_xlim(74, 80); ax0.set_ylim(8, 15.2); ax0.set_aspect("equal")
    ax0.text(75.45, 12.95, "Kodagu", fontsize=7, color=C_K, ha="right", weight="bold")
    ax0.text(76.55, 11.45, "Wayanad", fontsize=7, color=C_W, ha="left", weight="bold")
    ax0.set_title("(a) Location, southern India", loc="left", fontsize=8.5); ax0.tick_params(labelsize=7)
    ax0.set_xlabel("Longitude (°E)"); ax0.set_ylabel("Latitude (°N)")
    ls = LightSource(azdeg=315, altdeg=45)
    for i, (key, name, col, lab) in enumerate([("wayanad", "Wayanad", C_W, "(b) Wayanad, Kerala"),
                                                ("kodagu", "Kodagu", C_K, "(c) Kodagu, Karnataka")]):
        ax = fig.add_subplot(gs[i + 1])
        dem, T, b = read(key, "elevation"); m, _, _ = read(key, "mask")
        dem_m = np.where(m == 1, dem, np.nan)
        hs = ls.hillshade(np.nan_to_num(dem, nan=0), vert_exag=2, dx=RES, dy=RES)
        ext = (b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000)
        p = pos[pos.district == key]
        ax.imshow(np.where(m == 1, hs, np.nan), cmap="Greys_r", extent=ext, alpha=0.55)
        im = ax.imshow(dem_m, cmap="terrain", vmin=0, vmax=2300, extent=ext, alpha=0.65)
        g = dist[dist.shapeName == name].to_crs(CRS)
        g = g.set_geometry(g.geometry.affine_transform([1e-3, 0, 0, 1e-3, 0, 0]))
        g.plot(ax=ax, facecolor="none", edgecolor="#0b0b0b", lw=0.6)
        ax.scatter(p.x / 1000, p.y / 1000, s=5, c="#e34948", edgecolors="white", linewidths=0.25)
        ax.set_title(f"{lab}\n(n = {len(p)} landslide cells)", loc="left", fontsize=8.5)
        ax.set_xlabel("Easting (km, UTM 43N)"); ax.tick_params(labelsize=7)
        if i == 0: ax.set_ylabel("Northing (km)")
    cax = fig.add_axes([0.925, 0.22, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Elevation (m)"); cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#e34948", mec="white", ms=5, label="Landslide cell (2018 inventory)")],
               loc="lower left", ncol=1, bbox_to_anchor=(0.12, 0.0))
    fig.savefig(FIG / "Figure1_study_area.png"); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 56)
    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#52514e", lw=0.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, weight="bold", color="#0b0b0b")
        ax.text(x + w / 2, y + h - 6.3, "\n".join(lines), ha="center", va="top", fontsize=6.8, color="#0b0b0b", linespacing=1.35)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.8))
    light = "#eef4fc"; warm = "#fdf0ea"; green = "#e8f7f1"
    box(1, 30, 22, 24, "1 Open data", ["2018 inventories", "(Kerala; Kodagu)", "Copernicus DEM 30 m", "ESA WorldCover 2021",
                                        "CHIRPS rainfall", "SoilGrids 250 m", "GSI geology 1:2M", "OSM roads"], light)
    box(27, 30, 22, 24, "2 Factor stack", ["30 m grid, UTM 43N", "16 continuous factors", "(terrain, hydrology,", "rainfall, soil, roads)",
                                            "+ land cover, lithology", "(one-hot)"], light)
    box(53, 30, 22, 24, "3 Sampling", ["Landslide cells", "Absence cells > 100 m", "from landslides", "1:1 ratio,", "10 random draws"], light)
    box(79, 30, 20, 24, "4 Classifiers", ["Logistic regression", "Random forest", "XGBoost", "LightGBM"], light)
    box(1, 2, 30, 23, "5 Validation designs", ["Random 10-fold CV", "Spatial 5-fold CV (5 km blocks)", "Transfer Wayanad → Kodagu", "Transfer Kodagu → Wayanad",
                                                "AUC, accuracy, F1, kappa, Brier"], warm)
    box(35, 2, 30, 23, "6 Shift diagnostics", ["Background samples (20,000 cells", "per district)", "Standardised mean difference,", "Wasserstein distance,",
                                                "adversarial validation AUC"], green)
    box(69, 2, 30, 23, "7 Explain & adapt", ["TreeSHAP (XGBoost)", "Rank agreement across districts", "S2: regional z-scores", "S3: drop shifted factors",
                                              "Ensemble susceptibility maps"], green)
    arrow(23.8, 42, 26.2, 42); arrow(49.8, 42, 52.2, 42); arrow(75.8, 42, 78.2, 42)
    arrow(89, 29.2, 89, 27); arrow(89, 27, 16, 27); arrow(16, 27, 16, 25.8)
    arrow(31.8, 13.5, 34.2, 13.5); arrow(65.8, 13.5, 68.2, 13.5)
    fig.savefig(FIG / "Figure2_workflow.png"); plt.close(fig)


def fig3():
    shift = pd.read_csv(OUT / "tables" / "covariate_shift.csv")
    BG = pd.read_csv(BACKGROUND_CSV)
    top = shift.sort_values("adv_AUC", ascending=False).feature.tolist()[:8]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    for ax, f in zip(axes.ravel(), top):
        w = BG.loc[BG.district == "wayanad", f]; k = BG.loc[BG.district == "kodagu", f]
        lo, hi = np.nanpercentile(pd.concat([w, k]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(w.clip(lo, hi), bins=bins, density=True, histtype="step", lw=1.6, color=C_W)
        ax.hist(k.clip(lo, hi), bins=bins, density=True, histtype="step", lw=1.6, color=C_K)
        r = shift.set_index("feature").loc[f]
        ax.set_title(f"{LABELS[f]}", loc="left", fontsize=8)
        ax.text(0.98, 0.95, f"AUC {r.adv_AUC:.2f}\nSMD {r.SMD:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK2)
        ax.set_yticks([]); ax.tick_params(labelsize=6.5)
        ax.set_xlabel(UNITS.get(f, ""), fontsize=7)
    fig.legend(handles=[Line2D([], [], color=C_W, lw=2, label="Wayanad"), Line2D([], [], color=C_K, lw=2, label="Kodagu")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); fig.savefig(FIG / "Figure3_covariate_shift.png"); plt.close(fig)


def fig4():
    P = pd.read_csv(OUT / "tables" / "performance_all_runs.csv")
    P = P[P.strategy == "S1_all"]
    models = ["LR", "RF", "XGB", "LGBM"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (src, tgt, title) in zip(axes, [("wayanad", "kodagu", "(a) Trained in Wayanad"), ("kodagu", "wayanad", "(b) Trained in Kodagu")]):
        designs = [("random_10fold", src, "Random CV"), ("spatial_block5km", src, "Spatial CV"), ("transfer", tgt, f"Transfer → {tgt.capitalize()}")]
        xw = 0.19
        for j, mname in enumerate(models):
            means, sds = [], []
            for sch, test, _ in designs:
                v = P[(P.scheme == sch) & (P.train == src) & (P.test == test) & (P.model == mname)].AUC
                means.append(v.mean()); sds.append(v.std())
            x = np.arange(3) + (j - 1.5) * xw
            ax.bar(x, np.array(means) - 0.4, bottom=0.4, width=xw - 0.03, color=MODEL_C[mname], label=mname)
            ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="#0b0b0b", elinewidth=0.7, capsize=1.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([d[2] for d in designs])
        ax.set_title(title, loc="left"); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("AUC (mean ± SD, 10 samplings)")
    fig.legend(handles=[Patch(color=MODEL_C[m], label=m) for m in models], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig(FIG / "Figure4_validation_designs.png"); plt.close(fig)


def fig5():
    P = pd.read_csv(OUT / "tables" / "performance_all_runs.csv")
    T = P[P.scheme == "transfer"]
    models = ["LR", "RF", "XGB", "LGBM"]
    strat = [("S1_all", "S1\nall factors"), ("S2_regional_z", "S2\nregional z"), ("S3_drop_shifted", "S3\nshift removed"), ("S4_no_rainfall", "S4\nno rainfall")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (src, tgt, title) in zip(axes, [("wayanad", "kodagu", "(a) Wayanad → Kodagu"), ("kodagu", "wayanad", "(b) Kodagu → Wayanad")]):
        xw = 0.19
        for j, mname in enumerate(models):
            v = [T[(T.train == src) & (T.strategy == s) & (T.model == mname)].AUC for s, _ in strat]
            x = np.arange(4) + (j - 1.5) * xw
            ax.bar(x, np.array([a.mean() for a in v]) - 0.4, bottom=0.4, width=xw - 0.03, color=MODEL_C[mname], label=mname)
            ax.errorbar(x, [a.mean() for a in v], yerr=[a.std() for a in v], fmt="none", ecolor="#0b0b0b", elinewidth=0.7, capsize=1.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([s[1] for s in strat], fontsize=7)
        ax.set_title(title, loc="left"); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Transfer AUC (mean ± SD)")
    fig.legend(handles=[Patch(color=MODEL_C[m], label=m) for m in models], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig(FIG / "Figure7_transfer_strategies.png"); plt.close(fig)


def fig6():
    I = pd.read_csv(OUT / "shap" / "mean_abs_shap_grouped.csv", index_col=0)
    I = I.sort_values("wayanad_model_on_wayanad")
    y = np.arange(len(I)); h = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.barh(y + h / 2, I.wayanad_model_on_wayanad, height=h - 0.04, color=C_W, xerr=I.wayanad_model_on_wayanad_sd,
            error_kw=dict(elinewidth=0.6, ecolor="#0b0b0b"), label="Wayanad model, Wayanad samples")
    ax.barh(y - h / 2, I.kodagu_model_on_kodagu, height=h - 0.04, color=C_K, xerr=I.kodagu_model_on_kodagu_sd,
            error_kw=dict(elinewidth=0.6, ecolor="#0b0b0b"), label="Kodagu model, Kodagu samples")
    ax.set_yticks(y); ax.set_yticklabels([LABELS.get(i, i) for i in I.index])
    ax.set_xlabel("Mean |SHAP| (log-odds), XGBoost, 10 samplings")
    rho = json.load(open(OUT / "shap" / "shap_rank_agreement.json"))["spearman_local_models"]
    ax.set_title(f"Factor importance by district (Spearman ρ = {rho[0]:.2f}, p = {rho[1]:.3f})", loc="left", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1); ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(FIG / "Figure5_shap_importance.png"); plt.close(fig)


def fig7():
    D = pd.read_csv(OUT / "shap" / "shap_values_seed0.csv")
    I = pd.read_csv(OUT / "shap" / "mean_abs_shap_grouped.csv", index_col=0)
    cont = [f for f in (I.wayanad_model_on_wayanad + I.kodagu_model_on_kodagu).sort_values(ascending=False).index
            if f not in ("landcover", "lithology")][:6]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ax, f in zip(axes.ravel(), cont):
        for reg, col in (("wayanad", C_W), ("kodagu", C_K)):
            d = D[(D.model_region == reg) & (D.eval_region == reg)]
            ax.scatter(d[f], d[f"shap_{f}"], s=6, color=col, alpha=0.45, edgecolors="none", label=reg.capitalize())
        ax.axhline(0, color=INK2, lw=0.6, ls=":")
        ax.set_title(LABELS[f], loc="left", fontsize=8); ax.set_xlabel(UNITS.get(f, ""), fontsize=7); ax.tick_params(labelsize=6.5)
        if f in ("dist_roads", "dist_streams", "spi"):
            ax.set_xlim(right=np.nanpercentile(D[f], 99))
    for ax in axes[:, 0]: ax.set_ylabel("SHAP value (log-odds)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_W, label="Wayanad model on Wayanad"),
                        Line2D([], [], marker="o", ls="", color=C_K, label="Kodagu model on Kodagu")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); fig.savefig(FIG / "Figure6_shap_dependence.png"); plt.close(fig)


def fig8():
    S = pd.read_csv(SAMPLES_CSV); pos = S[(S.seed == 0) & (S.label == 1)]
    CS = pd.read_csv(OUT / "tables" / "class_statistics.csv")
    panels = [("wayanad_local", "wayanad", "(a) Wayanad, local, all factors"), ("wayanad_local_S4", "wayanad", "(b) Wayanad, local, no rainfall"),
              ("kodagu_local", "kodagu", "(c) Kodagu, local, all factors"), ("kodagu_from_wayanad_S1", "kodagu", "(d) Kodagu ← Wayanad, S1"),
              ("kodagu_from_wayanad_S2", "kodagu", "(e) Kodagu ← Wayanad, S2"), ("kodagu_from_wayanad_S4", "kodagu", "(f) Kodagu ← Wayanad, S4")]
    cmap = ListedColormap(["#fde0c5", "#facba6", "#f59e72", "#e2603f", "#a8321d"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.2))
    for ax, (tag, reg, title) in zip(axes.ravel(), panels):
        with rasterio.open(OUT / "maps" / f"susceptibility_{tag}.tif") as s:
            a = s.read(1); b = s.bounds
        q = np.nanquantile(a, [0.2, 0.4, 0.6, 0.8]); cls = np.where(np.isfinite(a), np.digitize(a, q), np.nan)
        ext = (b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000)
        ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
        p = pos[pos.district == reg]
        ax.scatter(p.x / 1000, p.y / 1000, s=2.2, c="#0b0b0b", edgecolors="none")
        ax.set_title(title, loc="left", fontsize=7.8)
        if "from_wayanad" in tag:
            vh = CS[(CS["map"] == tag) & (CS.cls.isin(["High", "Very high"]))].landslide_pct.sum()
            ax.text(0.02, 0.02, f"{vh:.0f}% of landslides in High +\nVery high (40% of area)", transform=ax.transAxes, fontsize=6.5, va="bottom")
        else:
            ax.text(0.02, 0.02, "fitted to this inventory\n(not an independent test)", transform=ax.transAxes, fontsize=6.5, va="bottom", color=INK2)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(["Very low", "Low", "Moderate", "High", "Very high"])] +
               [Line2D([], [], marker="o", ls="", color="#0b0b0b", ms=3, label="2018 landslide")],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "Figure8_susceptibility_maps.png"); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8"]
    for f in todo:
        globals()[f](); print(f, "saved", flush=True)
