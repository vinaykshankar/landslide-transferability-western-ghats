"""Assemble manuscript tables (JSON) directly from analysis outputs, so no number is transcribed by hand.
Usage: python3 08_manuscript_tables.py  -> manuscript/tables.json, manuscript/numbers.json
"""
import json
import numpy as np, pandas as pd
from config import ROOT

M = ROOT / "manuscript"; M.mkdir(exist_ok=True)
NR = ROOT / "outputs" / "no_roads"; WR = ROOT / "outputs" / "with_roads"
MODELS = ["LR", "RF", "XGB", "LGBM"]
f3 = lambda m, s: f"{m:.3f} ± {s:.3f}"


def perf_table(out):
    P = pd.read_csv(out / "tables" / "performance_all_runs.csv")
    g = P.groupby(["scheme", "train", "test", "strategy", "model"])
    stats = g.agg(AUC_m=("AUC", "mean"), AUC_s=("AUC", "std"), F1_m=("F1", "mean"), K_m=("Kappa", "mean"), B_m=("Brier", "mean")).reset_index()
    return P, stats


tables, numbers = {}, {}
# ---------------- Table 2: validation designs (S1, no roads) -----------------
P, st = perf_table(NR)
rows = [["Training district", "Validation design", "Test samples"] + MODELS]
designs = [("wayanad", "random_10fold", "wayanad", "Random 10-fold CV", "Wayanad"),
           ("wayanad", "spatial_block5km", "wayanad", "Spatial block CV (5 km)", "Wayanad"),
           ("wayanad", "transfer", "kodagu", "Cross-district transfer", "Kodagu"),
           ("kodagu", "random_10fold", "kodagu", "Random 10-fold CV", "Kodagu"),
           ("kodagu", "spatial_block5km", "kodagu", "Spatial block CV (5 km)", "Kodagu"),
           ("kodagu", "transfer", "wayanad", "Cross-district transfer", "Wayanad")]
for tr, sch, te, lab, tel in designs:
    r = [tr.capitalize(), lab, tel]
    for m in MODELS:
        x = st[(st.train == tr) & (st.scheme == sch) & (st.test == te) & (st.strategy == "S1_all") & (st.model == m)].iloc[0]
        r.append(f3(x.AUC_m, x.AUC_s))
        numbers[f"auc_{tr}_{sch}_{te}_S1_{m}"] = round(float(x.AUC_m), 3)
        numbers[f"auc_{tr}_{sch}_{te}_S1_{m}_sd"] = round(float(x.AUC_s), 3)
    rows.append(r)
tables["table2"] = rows

# secondary metrics (supplementary table S1)
rows = [["Training", "Design", "Model", "AUC", "Accuracy", "F1", "Kappa", "Brier"]]
g = P[P.strategy == "S1_all"].groupby(["train", "scheme", "model"])[["AUC", "ACC", "F1", "Kappa", "Brier"]].mean().reset_index()
for _, x in g.iterrows():
    rows.append([x.train.capitalize(), x.scheme, x.model] + [f"{x[c]:.3f}" for c in ["AUC", "ACC", "F1", "Kappa", "Brier"]])
tables["tableS1"] = rows

# ---------------- Table 3: covariate shift -----------------
sh = pd.read_csv(NR / "tables" / "covariate_shift.csv")
lab = {"elevation": "Elevation (m)", "slope": "Slope (°)", "northness": "Northness", "eastness": "Eastness",
       "plan_curvature": "Plan curvature (100 m⁻¹)", "profile_curvature": "Profile curvature (100 m⁻¹)", "tri": "TRI (m)",
       "twi": "TWI", "spi": "SPI (ln)", "dist_streams": "Distance to streams (m)", "dist_roads": "Distance to roads (m)",
       "rain_annual_mean": "Mean annual rainfall (mm)", "rain_aug2018": "August 2018 rainfall (mm)",
       "soil_clay": "Clay 0–30 cm (%)", "soil_sand": "Sand 0–30 cm (%)", "soil_bdod": "Bulk density 0–30 cm (g cm⁻³)"}
rows = [["Factor", "Mean Wayanad", "Mean Kodagu", "SMD", "W₁ (std.)", "Adversarial AUC"]]
for _, x in sh.iterrows():
    dec = 2 if abs(x.mean_wayanad) < 10 else 0
    rows.append([lab[x.feature], f"{x.mean_wayanad:.{dec}f}", f"{x.mean_kodagu:.{dec}f}", f"{x.SMD:+.2f}", f"{x.W1_std:.2f}", f"{x.adv_AUC:.2f}"])
    numbers[f"shift_{x.feature}_auc"] = round(float(x.adv_AUC), 2); numbers[f"shift_{x.feature}_smd"] = round(float(x.SMD), 2)
tables["table3"] = rows
numbers["adv_auc_multivariate"] = round(json.load(open(NR / "tables" / "shift_summary.json"))["adversarial_auc_multivariate"], 3)
sh_r = pd.read_csv(WR / "tables" / "covariate_shift.csv").set_index("feature").loc["dist_roads"]
numbers["shift_dist_roads"] = dict(mean_w=round(float(sh_r.mean_wayanad)), mean_k=round(float(sh_r.mean_kodagu)), smd=round(float(sh_r.SMD), 2), auc=round(float(sh_r.adv_AUC), 2))

# ---------------- Table 4: transfer strategies with paired tests -----------------
W = pd.read_csv(NR / "tables" / "paired_wilcoxon_tests.csv")
rows = [["Direction", "Strategy"] + MODELS]
names = {"S1_all": "S1 all factors", "S2_regional_z": "S2 regional z-scores", "S3_drop_shifted": "S3 shifted factor removed",
         "S4_no_rainfall": "S4 rainfall removed"}
for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
    for s, sname in names.items():
        r = [f"{src.capitalize()} → {tgt.capitalize()}", sname]
        for m in MODELS:
            x = st[(st.scheme == "transfer") & (st.train == src) & (st.strategy == s) & (st.model == m)].iloc[0]
            cell = f"{x.AUC_m:.3f} ± {x.AUC_s:.3f}"
            numbers[f"tr_{src}_{s}_{m}"] = round(float(x.AUC_m), 3); numbers[f"tr_{src}_{s}_{m}_sd"] = round(float(x.AUC_s), 3)
            if s != "S1_all":
                t = W[W.comparison == f"{src}->{tgt}: {s} - S1 | {m}"].iloc[0]
                star = "**" if t.p_value < 0.01 else ("*" if t.p_value < 0.05 else "")
                cell += star
                numbers[f"p_{src}_{s}_{m}"] = round(float(t.p_value), 4); numbers[f"d_{src}_{s}_{m}"] = round(float(t.mean_diff), 3)
            r.append(cell)
        rows.append(r)
tables["table4"] = rows
for _, t in W.iterrows():
    numbers["wilcoxon::" + t.comparison] = {"diff": round(float(t.mean_diff), 3), "p": round(float(t.p_value), 4), "wins": int(t.wins)}

# spatial CV without rainfall
cv4 = pd.read_csv(NR / "tables" / "spatial_cv_no_rainfall_runs.csv").groupby(["train", "model"]).AUC.agg(["mean", "std"])
for (tr, m), x in cv4.iterrows():
    numbers[f"auc_{tr}_spatialCV_S4_{m}"] = round(float(x["mean"]), 3)

# ---------------- Table 5: SHAP importance + effect agreement -----------------
I = pd.read_csv(NR / "shap" / "mean_abs_shap_grouped.csv", index_col=0)
A = pd.read_csv(NR / "tables" / "shap_effect_agreement.csv", index_col=0)
lab2 = {**{k: v.split(" (")[0] for k, v in lab.items()}, "landcover": "Land cover", "lithology": "Lithology"}
I["rank_w"] = I.wayanad_model_on_wayanad.rank(ascending=False).astype(int)
I["rank_k"] = I.kodagu_model_on_kodagu.rank(ascending=False).astype(int)
rows = [["Factor", "Mean |SHAP| Wayanad model (rank)", "Mean |SHAP| Kodagu model (rank)", "Effect agreement r (mean ± SD)"]]
for f, x in I.sort_values("wayanad_model_on_wayanad", ascending=False).iterrows():
    rows.append([lab2[f], f"{x.wayanad_model_on_wayanad:.2f} ({int(x.rank_w)})", f"{x.kodagu_model_on_kodagu:.2f} ({int(x.rank_k)})",
                 f"{A.loc[f, 'mean']:.2f} ± {A.loc[f, 'std']:.2f}"])
    numbers[f"shap_{f}"] = dict(w=round(float(x.wayanad_model_on_wayanad), 2), k=round(float(x.kodagu_model_on_kodagu), 2),
                                rank_w=int(x.rank_w), rank_k=int(x.rank_k), agree=round(float(A.loc[f, "mean"]), 2))
tables["table5"] = rows
numbers["shap_rank_agreement"] = json.load(open(NR / "shap" / "shap_rank_agreement.json"))

# ---------------- class statistics -----------------
CS = pd.read_csv(NR / "tables" / "class_statistics.csv")
for tag in ["kodagu_from_wayanad_S1", "kodagu_from_wayanad_S2", "kodagu_from_wayanad_S3", "kodagu_from_wayanad_S4"]:
    c = CS[CS["map"] == tag].set_index("cls")
    numbers[f"cls_{tag}"] = dict(vh_pct=round(float(c.loc["Very high", "landslide_pct"]), 1),
                                 hvh_pct=round(float(c.loc[["High", "Very high"], "landslide_pct"].sum()), 1),
                                 vl_l_pct=round(float(c.loc[["Very low", "Low"], "landslide_pct"].sum()), 1),
                                 fr_vh=round(float(c.loc["Very high", "FR"]), 2))
rows = [["Map", "Very low", "Low", "Moderate", "High", "Very high"]]
mapname = {"kodagu_from_wayanad_S1": "S1 all factors", "kodagu_from_wayanad_S2": "S2 regional z-scores",
           "kodagu_from_wayanad_S3": "S3 shifted factor removed", "kodagu_from_wayanad_S4": "S4 rainfall removed"}
for tag, nm in mapname.items():
    c = CS[CS["map"] == tag].set_index("cls")
    rows.append([nm] + [f"{c.loc[k, 'landslide_pct']:.1f} ({c.loc[k, 'FR']:.2f})" for k in ["Very low", "Low", "Moderate", "High", "Very high"]])
tables["table6"] = rows

# ---------------- Table 7: roads sensitivity -----------------
P2, st2 = perf_table(WR)
rows = [["Design", "Strategy", "Model", "AUC without roads", "AUC with roads", "Δ"]]
for tr, sch, te, strat, lab_ in [("wayanad", "spatial_block5km", "wayanad", "S1_all", "Spatial CV, Wayanad"),
                                 ("kodagu", "spatial_block5km", "kodagu", "S1_all", "Spatial CV, Kodagu"),
                                 ("wayanad", "transfer", "kodagu", "S1_all", "Wayanad → Kodagu"),
                                 ("wayanad", "transfer", "kodagu", "S2_regional_z", "Wayanad → Kodagu"),
                                 ("wayanad", "transfer", "kodagu", "S4_no_rainfall", "Wayanad → Kodagu"),
                                 ("kodagu", "transfer", "wayanad", "S1_all", "Kodagu → Wayanad"),
                                 ("kodagu", "transfer", "wayanad", "S4_no_rainfall", "Kodagu → Wayanad")]:
    for m in ("RF", "XGB"):
        a = st[(st.train == tr) & (st.scheme == sch) & (st.test == te) & (st.strategy == strat) & (st.model == m)].AUC_m.iloc[0]
        b = st2[(st2.train == tr) & (st2.scheme == sch) & (st2.test == te) & (st2.strategy == strat) & (st2.model == m)].AUC_m.iloc[0]
        rows.append([lab_, strat.split("_")[0], m, f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"])
tables["table7"] = rows
I2 = pd.read_csv(WR / "shap" / "mean_abs_shap_grouped.csv", index_col=0)
numbers["roads_shap"] = dict(w=round(float(I2.loc["dist_roads", "wayanad_model_on_wayanad"]), 2),
                             k=round(float(I2.loc["dist_roads", "kodagu_model_on_kodagu"]), 2),
                             rank_w=int(I2.wayanad_model_on_wayanad.rank(ascending=False)["dist_roads"]),
                             rank_k=int(I2.kodagu_model_on_kodagu.rank(ascending=False)["dist_roads"]))
numbers["roads_rank_agreement"] = json.load(open(WR / "shap" / "shap_rank_agreement.json"))
numbers["inventory"] = json.load(open(NR / "inventory_summary.json"))
numbers["vif"] = pd.read_csv(NR / "tables" / "vif.csv").round(2).set_index("factor").VIF.to_dict()
json.dump(tables, open(M / "tables.json", "w"), indent=1, ensure_ascii=False)
json.dump(numbers, open(M / "numbers.json", "w"), indent=1, ensure_ascii=False)
print("tables:", list(tables)); print(json.dumps({k: v for k, v in numbers.items() if not k.startswith("wilcoxon")}, ensure_ascii=False)[:3000])
