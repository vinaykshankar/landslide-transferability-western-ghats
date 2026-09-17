"""(i) Spatial-block CV of the no-rainfall factor set (S4) within each district;
(ii) paired Wilcoxon signed-rank tests across the 10 samplings;
(iii) collinearity diagnostics (Spearman correlation, VIF) of the continuous factors.
Usage: python3 07_supplementary_stats.py
"""
import json, importlib.util, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import *
warnings.filterwarnings("ignore")

TAB = OUT / "tables"
S = pd.read_csv(SAMPLES_CSV)
ONEHOT = [c for c in S.columns if (c.startswith("lc_") and c != "lc_class") or (c.startswith("lith_") and c != "lith_class")]
FEATS = CONT + ONEHOT
NORAIN = [f for f in FEATS if not f.startswith("rain_")]

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def models(seed):
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "RF": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=seed),
        "XGB": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=2, reg_lambda=1.0, n_jobs=2, random_state=seed, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, min_child_samples=10, n_jobs=2, random_state=seed, verbose=-1),
    }


def blocks(df, m=5000):
    return np.floor(df.x / m).astype(int).astype(str) + "_" + np.floor(df.y / m).astype(int).astype(str)


# (i) spatial CV without rainfall
rows = []
for seed in SEEDS:
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)].reset_index(drop=True)
        for mname, m in models(seed).items():
            p = cross_val_predict(m, d[NORAIN], d.label, cv=GroupKFold(5), groups=blocks(d), method="predict_proba")[:, 1]
            rows.append(dict(seed=seed, scheme="spatial_block5km", train=reg, test=reg, strategy="S4_no_rainfall", model=mname,
                             AUC=roc_auc_score(d.label, p)))
cv4 = pd.DataFrame(rows)
cv4.to_csv(TAB / "spatial_cv_no_rainfall_runs.csv", index=False)
print(cv4.groupby(["train", "model"]).AUC.agg(["mean", "std"]).round(3))

# (ii) paired tests
P = pd.read_csv(TAB / "performance_all_runs.csv")
tests = []
def paired(a, b, label):
    a = a.sort_values("seed").AUC.values; b = b.sort_values("seed").AUC.values
    stat, p = wilcoxon(a, b)
    tests.append(dict(comparison=label, mean_diff=float(np.mean(b - a)), wins=int(np.sum(b > a)), n=len(a), p_value=float(p)))
for mname in ("LR", "RF", "XGB", "LGBM"):
    for reg in ("wayanad", "kodagu"):
        r = P[(P.scheme == "random_10fold") & (P.train == reg) & (P.model == mname)]
        s = P[(P.scheme == "spatial_block5km") & (P.train == reg) & (P.model == mname)]
        paired(r, s, f"{reg}: spatial CV - random CV | {mname}")
        s4 = cv4[(cv4.train == reg) & (cv4.model == mname)]
        paired(s, s4, f"{reg}: spatial CV S4 - S1 | {mname}")
    for src, tgt in (("wayanad", "kodagu"), ("kodagu", "wayanad")):
        s1 = P[(P.scheme == "transfer") & (P.train == src) & (P.strategy == "S1_all") & (P.model == mname)]
        loc = P[(P.scheme == "spatial_block5km") & (P.train == tgt) & (P.model == mname)]
        paired(loc, s1, f"{src}->{tgt}: transfer S1 - target spatial CV | {mname}")
        for st in ("S2_regional_z", "S3_drop_shifted", "S4_no_rainfall"):
            x = P[(P.scheme == "transfer") & (P.train == src) & (P.strategy == st) & (P.model == mname)]
            paired(s1, x, f"{src}->{tgt}: {st} - S1 | {mname}")
T = pd.DataFrame(tests); T.to_csv(TAB / "paired_wilcoxon_tests.csv", index=False)
print(T.round(4).to_string())

# (iii) collinearity on pooled background sample
BG = pd.read_csv(BACKGROUND_CSV)
X = BG[CONT].dropna()
corr = X.corr(method="spearman"); corr.to_csv(TAB / "spearman_correlation_factors.csv")
Xs = (X - X.mean()) / X.std()
Xs.insert(0, "const", 1.0)
vif = pd.DataFrame({"factor": CONT, "VIF": [variance_inflation_factor(Xs.values, i + 1) for i in range(len(CONT))]})
vif.to_csv(TAB / "vif.csv", index=False)
print(vif.round(2).to_string())
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(CONT) for b in CONT[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
print("pairs |rho|>=0.7:", pairs)

# (iv) agreement of learned factor effects: SHAP values of the Wayanad- and Kodagu-trained XGBoost models
#      evaluated on the same pooled samples (Pearson r per factor, mean over seeds)
import shap
GROUP = {**{c: c for c in CONT}, **{c: "landcover" for c in ONEHOT if c.startswith("lc_")},
         **{c: "lithology" for c in ONEHOT if c.startswith("lith_")}}
agree = []
for seed in SEEDS:
    pooled = S[S.seed == seed].reset_index(drop=True)
    sv = {}
    for reg in ("wayanad", "kodagu"):
        d = S[(S.district == reg) & (S.seed == seed)]
        m = models(seed)["XGB"].fit(d[FEATS], d.label)
        v = pd.DataFrame(shap.TreeExplainer(m).shap_values(pooled[FEATS]), columns=FEATS)
        sv[reg] = v.T.groupby(pd.Series(GROUP)[FEATS].values).sum().T
    for f in sv["wayanad"].columns:
        r = np.corrcoef(sv["wayanad"][f], sv["kodagu"][f])[0, 1]
        agree.append(dict(seed=seed, factor=f, pearson_r=r))
A = pd.DataFrame(agree).groupby("factor").pearson_r.agg(["mean", "std"]).sort_values("mean", ascending=False)
A.to_csv(TAB / "shap_effect_agreement.csv")
print(A.round(3).to_string())
