"""Model benchmarking: random vs spatial CV, cross-region transfer, covariate shift, SHAP.
Usage: python3 04_models.py
Outputs in outputs/tables and outputs/shap
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from config import *
warnings.filterwarnings("ignore")

TAB = OUT / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SH = OUT / "shap"; SH.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(SAMPLES_CSV)
BG = pd.read_csv(BACKGROUND_CSV)
ONEHOT = [c for c in S.columns if c.startswith("lc_") and c != "lc_class"] + \
         [c for c in S.columns if c.startswith("lith_") and c != "lith_class"]
FEATS = CONT + ONEHOT
GROUP = {**{c: c for c in CONT}, **{c: "landcover" for c in ONEHOT if c.startswith("lc_")},
         **{c: "lithology" for c in ONEHOT if c.startswith("lith_")}}
BLOCK_M = 5000

def models(seed):
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "RF": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=seed),
        "XGB": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=2, reg_lambda=1.0, n_jobs=2, random_state=seed, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, min_child_samples=10, n_jobs=2, random_state=seed, verbose=-1),
    }

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(AUC=roc_auc_score(y, p), ACC=accuracy_score(y, yhat), F1=f1_score(y, yhat),
                Kappa=cohen_kappa_score(y, yhat), Brier=brier_score_loss(y, p))

def blocks(df):
    return (np.floor(df.x / BLOCK_M).astype(int).astype(str) + "_" + np.floor(df.y / BLOCK_M).astype(int).astype(str))

def zscore_by_region(train, test, feats):
    """Unsupervised region-wise standardisation of continuous features using each region's background sample."""
    tr, te = train.copy(), test.copy()
    for f in [c for c in feats if c in CONT]:
        for df, reg in ((tr, train.district.iloc[0]), (te, test.district.iloc[0])):
            b = BG.loc[BG.district == reg, f]
            df[f] = (df[f] - b.mean()) / b.std()
    return tr, te

# ------------------------------------------------------------------ 1. covariate shift
shift = []
bw, bk = BG[BG.district == "wayanad"], BG[BG.district == "kodagu"]
for f in CONT:
    pooled_sd = np.sqrt((bw[f].var() + bk[f].var()) / 2)
    y = np.r_[np.zeros(len(bw)), np.ones(len(bk))]; x = np.r_[bw[f].values, bk[f].values]
    auc = roc_auc_score(y, x); auc = max(auc, 1 - auc)
    shift.append(dict(feature=f, mean_wayanad=bw[f].mean(), mean_kodagu=bk[f].mean(),
                      SMD=(bk[f].mean() - bw[f].mean()) / pooled_sd,
                      W1_std=wasserstein_distance(bw[f] / pooled_sd, bk[f] / pooled_sd), adv_AUC=auc))
shift = pd.DataFrame(shift).sort_values("adv_AUC", ascending=False)
# multivariate adversarial validation
Xa = pd.concat([bw[FEATS], bk[FEATS]]); ya = np.r_[np.zeros(len(bw)), np.ones(len(bk))]
pa = cross_val_predict(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=2, random_state=0),
                       Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
adv_multi = roc_auc_score(ya, pa)
shift.to_csv(TAB / "covariate_shift.csv", index=False)
SHIFTED = shift.loc[shift.adv_AUC > 0.80, "feature"].tolist()
print("multivariate adversarial AUC", round(adv_multi, 3), "| shifted features (single-feature AUC>0.80):", SHIFTED, flush=True)
json.dump({"adversarial_auc_multivariate": adv_multi, "shifted_features": SHIFTED}, open(TAB / "shift_summary.json", "w"), indent=2)

STRATS = {"S1_all": (FEATS, False), "S2_regional_z": (FEATS, True), "S3_drop_shifted": ([f for f in FEATS if f not in SHIFTED], False),
          "S4_no_rainfall": ([f for f in FEATS if not f.startswith("rain_")], False)}

# ------------------------------------------------------------------ 2. within-region CV and transfer
rows = []
for seed in SEEDS:
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)].reset_index(drop=True)
        X, y, g = d[FEATS], d.label.values, blocks(d)
        for mname, m in models(seed).items():
            p_rand = cross_val_predict(m, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
            rows.append(dict(seed=seed, scheme="random_10fold", train=reg, test=reg, strategy="S1_all", model=mname, **metrics(y, p_rand)))
            p_sp = cross_val_predict(models(seed)[mname], X, y, cv=GroupKFold(5), groups=g, method="predict_proba")[:, 1]
            rows.append(dict(seed=seed, scheme="spatial_block5km", train=reg, test=reg, strategy="S1_all", model=mname, **metrics(y, p_sp)))
    for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
        tr = S[(S.district == src) & (S.seed == seed)].reset_index(drop=True)
        te = S[(S.district == tgt) & (S.seed == seed)].reset_index(drop=True)
        for sname, (feats, z) in STRATS.items():
            trX, teX = (zscore_by_region(tr, te, feats) if z else (tr, te))
            for mname, m in models(seed).items():
                m.fit(trX[feats], tr.label)
                p = m.predict_proba(teX[feats])[:, 1]
                rows.append(dict(seed=seed, scheme="transfer", train=src, test=tgt, strategy=sname, model=mname, **metrics(te.label.values, p)))
    print("seed", seed, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(TAB / "performance_all_runs.csv", index=False)
summ = R.groupby(["scheme", "train", "test", "strategy", "model"])[["AUC", "ACC", "F1", "Kappa", "Brier"]].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
summ.to_csv(TAB / "performance_summary.csv", index=False)
print(summ[["scheme", "train", "test", "strategy", "model", "AUC_mean", "AUC_std"]].to_string(), flush=True)

# ------------------------------------------------------------------ 3. SHAP (XGBoost, all seeds pooled)
def grouped_shap(sv, cols):
    df = pd.DataFrame(sv, columns=cols)
    return df.T.groupby(pd.Series(GROUP)[cols].values).sum().T

imp = {}
dep_rows = []
for seed in SEEDS:
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)].reset_index(drop=True)
        m = models(seed)["XGB"].fit(d[FEATS], d.label)
        ex = shap.TreeExplainer(m)
        for evreg in ("wayanad", "kodagu"):
            e = S[(S.district == evreg) & (S.seed == seed)].reset_index(drop=True)
            sv = ex.shap_values(e[FEATS])
            gs = grouped_shap(sv, FEATS)
            imp.setdefault((reg, evreg), []).append(gs.abs().mean())
            if seed == 0:
                out = gs.add_prefix("shap_"); out[CONT] = e[CONT]; out["lith_class"] = e.lith_class; out["lc_class"] = e.lc_class
                out["label"] = e.label; out["model_region"] = reg; out["eval_region"] = evreg
                dep_rows.append(out)
I = pd.DataFrame({f"{a}_model_on_{b}": pd.concat(v, axis=1).mean(axis=1) for (a, b), v in imp.items()})
I_sd = pd.DataFrame({f"{a}_model_on_{b}_sd": pd.concat(v, axis=1).std(axis=1) for (a, b), v in imp.items()})
I = I.join(I_sd).sort_values("wayanad_model_on_wayanad", ascending=False)
I.to_csv(SH / "mean_abs_shap_grouped.csv")
pd.concat(dep_rows).to_csv(SH / "shap_values_seed0.csv", index=False)
rho_models, pval = spearmanr(I["wayanad_model_on_wayanad"], I["kodagu_model_on_kodagu"])
rho_transfer, pval2 = spearmanr(I["wayanad_model_on_kodagu"], I["kodagu_model_on_kodagu"])
json.dump({"spearman_local_models": [rho_models, pval], "spearman_transferred_vs_local_on_kodagu": [rho_transfer, pval2]},
          open(SH / "shap_rank_agreement.json", "w"), indent=2)
print(I.round(4).to_string()); print("rho local models", rho_models, pval, "| rho transferred vs local", rho_transfer, pval2)
