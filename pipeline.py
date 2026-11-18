"""
Beyond Clicks - AMLBT Project, Team 43C  (Version 2 pipeline)

Addresses the first-review feedback:
  * explicit scoring timestamps and a feature-timing audit (FEATURE_TIMING below);
  * three feature branches evaluated end-to-end (all four models, out-of-fold
    threshold, calibration, explanations, tiers and scenario analysis):
        START  - information available when the session starts (t0)
        NOPV   - START + in-session navigation aggregates, no PageValues (t_end)
        FULL   - NOPV + PageValues (t_end), the V1 setting
  * probability calibration fitted on training data only (isotonic, 5-fold);
  * uncertainty: 5-fold CV spread, test-set bootstrap CIs, paired bootstrap
    differences, and bootstrap bands for the scenario comparison;
  * month-blocked forward check (train Feb-Oct, test Nov-Dec);
  * environment, data hash and every reported number written to disk.

USAGE
    pip install -r requirements.txt
    python pipeline.py online_shoppers_intention.csv  [output_dir]

Outputs (default folder: overleaf_files/): figures (*.png), LaTeX tables
(table_*.tex), LaTeX macros (results_macros.tex), results.json,
environment.txt.  Runtime is about 8 minutes on two CPU cores.
"""
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, brier_score_loss,
                             confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
T_START = time.time()
SEED = 42
B_BOOT = 2000
rng_global = np.random.default_rng(SEED)
DATA = sys.argv[1] if len(sys.argv) > 1 else "online_shoppers_intention.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "overleaf_files"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
MACROS, RESULTS = {}, {}


def setm(name, value):
    if isinstance(value, str) and len(value) > 1 and value[0] == "-" and value[1].isdigit():
        value = "$-$" + value[1:]          # typographic minus in LaTeX
    MACROS[name] = value


def tex(s):
    return str(s).replace("_", r"\_")


def log(*a):
    print(f"[{time.time() - T_START:6.0f}s]", *a, flush=True)


# ----------------------------------------------------------------------------
# 0. Feature-timing audit: when is each feature known?
#    t0   = session start (first page request)
#    tend = session end (all pages of the session observed)
# ----------------------------------------------------------------------------
FEATURE_TIMING = {
    # feature: (group, earliest time it is fully known, note)
    "Month": ("context", "t0", "calendar"),
    "SpecialDay": ("context", "t0", "calendar (closeness to a holiday)"),
    "Weekend": ("context", "t0", "calendar"),
    "VisitorType": ("context", "t0", "cookie / login"),
    "OperatingSystems": ("context", "t0", "user agent"),
    "Browser": ("context", "t0", "user agent"),
    "Region": ("context", "t0", "IP geolocation"),
    "TrafficType": ("context", "t0", "referrer / campaign"),
    "Administrative": ("navigation", "tend", "running count"),
    "Administrative_Duration": ("navigation", "tend", "running sum"),
    "Informational": ("navigation", "tend", "running count"),
    "Informational_Duration": ("navigation", "tend", "running sum"),
    "ProductRelated": ("navigation", "tend", "running count"),
    "ProductRelated_Duration": ("navigation", "tend", "running sum"),
    "BounceRates": ("navigation", "tend", "mean over pages visited"),
    "ExitRates": ("navigation", "tend", "mean over pages visited"),
    "PageValues": ("outcome-proximal", "tend", "mean value of pages visited; built from transactions"),
}
CONTEXT = [f for f, v in FEATURE_TIMING.items() if v[0] == "context"]
NAV = [f for f, v in FEATURE_TIMING.items() if v[0] == "navigation"]
CAT_ALL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BRANCHES = {
    "START": CONTEXT,
    "NOPV": CONTEXT + NAV,
    "FULL": CONTEXT + NAV + ["PageValues"],
}
BR_LABEL = {"START": "Session-start ($t_0$)", "NOPV": "No-PageValues ($t_{end}$)",
            "FULL": "Full ($t_{end}$)"}
BR_SHORT = {"START": "Start", "NOPV": "No-PV", "FULL": "Full"}

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
with open(DATA, "rb") as fh:
    data_sha = hashlib.sha256(fh.read()).hexdigest()
df = pd.read_csv(DATA)
n_raw, n_missing, n_dup = len(df), int(df.isna().sum().sum()), int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)
for c in CAT_ALL:
    df[c] = df[c].astype(str)
y = df["Revenue"].values
setm("NRaw", f"{n_raw:,}"); setm("NMissing", n_missing); setm("NDup", n_dup)
setm("NClean", f"{len(df):,}"); setm("NPos", f"{int(y.sum()):,}")
setm("PosRate", f"{100 * y.mean():.1f}")

# feature-timing table
lines = [r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.3cm}c>{\raggedright\arraybackslash}p{1.9cm}}", r"\toprule",
         r"Feature group & Known at & Used in branch \\", r"\midrule",
         r"Context: Month, SpecialDay, Weekend, VisitorType, OS, Browser, Region, TrafficType"
         r" & $t_0$ & Start, No-PV, Full \\",
         r"Navigation: page counts and durations (3 page types), BounceRates, ExitRates"
         r" & $t_{end}$ & No-PV, Full \\",
         r"Outcome-proximal: PageValues & $t_{end}$ & Full only \\",
         r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_timing.tex", "w").write("\n".join(lines))
pd.DataFrame([(k,) + v for k, v in FEATURE_TIMING.items()],
             columns=["feature", "group", "known_at", "note"]).to_csv(
    f"{OUT}/feature_timing.csv", index=False)

# EDA numbers (kept for the text)
vt = df.groupby("VisitorType")["Revenue"].mean().mul(100)
pv = df.groupby(df["PageValues"] > 0)["Revenue"].mean().mul(100)
setm("RateNew", f"{vt['New_Visitor']:.1f}"); setm("RateRet", f"{vt['Returning_Visitor']:.1f}")
setm("RatePVzero", f"{pv[False]:.1f}"); setm("RatePVpos", f"{pv[True]:.1f}")
setm("RateNov", f"{100 * df.loc[df.Month == 'Nov', 'Revenue'].mean():.1f}")
setm("SharePVpos", f"{100 * (df.PageValues > 0).mean():.1f}")

# ----------------------------------------------------------------------------
# 2. Split
# ----------------------------------------------------------------------------
idx = np.arange(len(df))
itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
ytr, yte = y[itr], y[ite]
setm("NTrain", f"{len(itr):,}"); setm("NTest", f"{len(ite):,}")
setm("BaseAP", f"{yte.mean():.3f}")
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)


def make_pipe(name, feats):
    cat = [f for f in feats if f in CAT_ALL]
    num = [f for f in feats if f not in CAT_ALL]
    pre = ColumnTransformer([
        ("num", StandardScaler() if name == "Logistic Regression" else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return Pipeline([("pre", pre), ("clf", clone(MODELS[name]))])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                            class_weight="balanced_subsample", random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, subsample=0.8, random_state=SEED),
}
MSHORT = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF",
          "Gradient Boosting": "GB"}


def calibrated(name, feats):
    return CalibratedClassifierCV(make_pipe(name, feats), method="isotonic",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.nanargmax(f1[:-1])])


def ece(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y_true[b == k].mean() - p[b == k].mean()) * (b == k).mean()
                     for k in range(bins) if (b == k).any()))


# ----------------------------------------------------------------------------
# 3. Fit every model on every branch
#    raw: uncalibrated pipeline fit on all training data (for SHAP / "before")
#    cal: isotonic-calibrated model (5-fold, training data only)
#    oof: out-of-fold calibrated probabilities on training -> threshold
# ----------------------------------------------------------------------------
R = {}
for br, feats in BRANCHES.items():
    Xtr, Xte = df.loc[itr, feats], df.loc[ite, feats]
    for name in MODELS:
        log("fit", br, name)
        oof = np.zeros(len(itr)); fold_auc = []
        for f_tr, f_va in skf.split(Xtr, ytr):
            m = calibrated(name, feats).fit(Xtr.iloc[f_tr], ytr[f_tr])
            oof[f_va] = m.predict_proba(Xtr.iloc[f_va])[:, 1]
            fold_auc.append(roc_auc_score(ytr[f_va], oof[f_va]))
        thr = best_f1_threshold(ytr, oof)
        cal = calibrated(name, feats).fit(Xtr, ytr)
        raw = make_pipe(name, feats).fit(Xtr, ytr)
        p = cal.predict_proba(Xte)[:, 1]
        p_raw = raw.predict_proba(Xte)[:, 1]
        yh = (p >= thr).astype(int)
        R[(br, name)] = dict(
            p=p, p_raw=p_raw, thr=thr, raw=raw, feats=feats,
            cv_auc_mean=float(np.mean(fold_auc)), cv_auc_sd=float(np.std(fold_auc, ddof=1)),
            auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
            prec=precision_score(yte, yh, zero_division=0), rec=recall_score(yte, yh),
            f1=f1_score(yte, yh), brier=brier_score_loss(yte, p),
            brier_raw=brier_score_loss(yte, p_raw), ece=ece(yte, p), ece_raw=ece(yte, p_raw),
            auc_raw=roc_auc_score(yte, p_raw), cm=confusion_matrix(yte, yh).ravel())

# ----------------------------------------------------------------------------
# 4. Bootstrap uncertainty on the test set (same resamples for every model -> paired)
# ----------------------------------------------------------------------------
log("bootstrap")
nte = len(ite)
BOOT_IDX = [rng_global.integers(0, nte, nte) for _ in range(B_BOOT)]
BOOT_IDX = [b for b in BOOT_IDX if 0 < yte[b].sum() < len(b)]


def boot_metric(key):
    r = R[key]
    yh = (r["p"] >= r["thr"]).astype(int)
    out = {"auc": [], "ap": [], "f1": []}
    for b in BOOT_IDX:
        out["auc"].append(roc_auc_score(yte[b], r["p"][b]))
        out["ap"].append(average_precision_score(yte[b], r["p"][b]))
        out["f1"].append(f1_score(yte[b], yh[b]))
    return {k: np.array(v) for k, v in out.items()}


BOOT = {k: boot_metric(k) for k in R}


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def paired(k1, k2, metric):
    d = BOOT[k1][metric] - BOOT[k2][metric]
    lo, hi = ci(d)
    return float(d.mean()), lo, hi, float((d <= 0).mean())


# ----------------------------------------------------------------------------
# 5. Main results table (branch x model)
# ----------------------------------------------------------------------------
lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Branch & Model & CV AUC & Test ROC-AUC [95\% CI] & PR-AUC [95\% CI] & F1 & ECE \\",
         r"\midrule"]
for bi, br in enumerate(BRANCHES):
    best_auc = max(R[(br, n)]["auc"] for n in MODELS)
    for j, name in enumerate(MODELS):
        r = R[(br, name)]; a = ci(BOOT[(br, name)]["auc"]); p_ = ci(BOOT[(br, name)]["ap"])
        lab = BR_SHORT[br] if j == 0 else ""
        aucs = f"{r['auc']:.3f}"
        if abs(r["auc"] - best_auc) < 1e-12:
            aucs = rf"\textbf{{{aucs}}}"
        lines.append(f"{lab} & {MSHORT[name]} & {r['cv_auc_mean']:.3f}$\\pm${r['cv_auc_sd']:.3f} & "
                     f"{aucs} [{a[0]:.3f}, {a[1]:.3f}] & {r['ap']:.3f} [{p_[0]:.3f}, {p_[1]:.3f}] & "
                     f"{r['f1']:.3f} & {r['ece']:.3f} \\\\")
    if bi < 2:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_models.tex", "w").write("\n".join(lines))

for br in BRANCHES:
    for name in MODELS:
        r = R[(br, name)]; k = BR_SHORT[br].replace("-", "") + MSHORT[name]
        a = ci(BOOT[(br, name)]["auc"]); p_ = ci(BOOT[(br, name)]["ap"]); f = ci(BOOT[(br, name)]["f1"])
        setm(f"AUC{k}", f"{r['auc']:.3f}"); setm(f"AUClo{k}", f"{a[0]:.3f}"); setm(f"AUChi{k}", f"{a[1]:.3f}")
        setm(f"AP{k}", f"{r['ap']:.3f}"); setm(f"APlo{k}", f"{p_[0]:.3f}"); setm(f"APhi{k}", f"{p_[1]:.3f}")
        setm(f"F{k}", f"{r['f1']:.3f}"); setm(f"Flo{k}", f"{f[0]:.3f}"); setm(f"Fhi{k}", f"{f[1]:.3f}")
        setm(f"Prec{k}", f"{r['prec']:.3f}"); setm(f"Rec{k}", f"{r['rec']:.3f}")
        setm(f"Thr{k}", f"{r['thr']:.2f}"); setm(f"ECE{k}", f"{r['ece']:.3f}")
        setm(f"ECEraw{k}", f"{r['ece_raw']:.3f}"); setm(f"Brier{k}", f"{r['brier']:.3f}")
        setm(f"Brierraw{k}", f"{r['brier_raw']:.3f}"); setm(f"CVAUC{k}", f"{r['cv_auc_mean']:.3f}")
        setm(f"CVsd{k}", f"{r['cv_auc_sd']:.3f}")
        tn, fp, fn, tp = r["cm"]
        setm(f"TP{k}", int(tp)); setm(f"FP{k}", int(fp)); setm(f"FN{k}", int(fn)); setm(f"TN{k}", int(tn))
        RESULTS[f"{br}|{name}"] = {kk: (float(v) if np.isscalar(v) else None) for kk, v in r.items()
                                   if kk in ["auc", "ap", "prec", "rec", "f1", "brier", "brier_raw",
                                             "ece", "ece_raw", "thr", "cv_auc_mean", "cv_auc_sd",
                                             "auc_raw"]}
        RESULTS[f"{br}|{name}"].update(auc_ci=a, ap_ci=p_, f1_ci=f, cm=[int(x) for x in r["cm"]])

# paired comparisons
GB, RF, LR = "Gradient Boosting", "Random Forest", "Logistic Regression"
PAIRS = {
    "GBvsRF_FULL": (("FULL", GB), ("FULL", RF)), "GBvsLR_FULL": (("FULL", GB), ("FULL", LR)),
    "GBvsRF_NOPV": (("NOPV", GB), ("NOPV", RF)), "GBvsLR_NOPV": (("NOPV", GB), ("NOPV", LR)),
    "GBvsRF_START": (("START", GB), ("START", RF)),
    "FULLvsNOPV_GB": (("FULL", GB), ("NOPV", GB)), "NOPVvsSTART_GB": (("NOPV", GB), ("START", GB)),
}
for nm, (k1, k2) in PAIRS.items():
    for met in ["auc", "ap"]:
        m, lo, hi, pv_ = paired(k1, k2, met)
        RESULTS[f"paired|{nm}|{met}"] = dict(diff=m, lo=lo, hi=hi, p_le0=pv_)
        mk = nm.replace("_", "") + met.upper()
        setm(f"D{mk}", f"{m:+.3f}"); setm(f"Dlo{mk}", f"{lo:+.3f}"); setm(f"Dhi{mk}", f"{hi:+.3f}")
        setm(f"Dp{mk}", f"{pv_:.3f}")

# ----------------------------------------------------------------------------
# 6. Month-blocked forward check: train Feb-Oct, test Nov-Dec (GB, calibrated)
# ----------------------------------------------------------------------------
log("temporal check")
early = df.Month.isin(["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct"]).values
late = ~early
setm("NEarly", f"{early.sum():,}"); setm("NLate", f"{late.sum():,}")
setm("RateEarly", f"{100 * y[early].mean():.1f}"); setm("RateLate", f"{100 * y[late].mean():.1f}")
for br, feats in BRANCHES.items():
    f2 = [f for f in feats if f != "Month"]   # month levels in the test window are unseen
    m = calibrated(GB, f2).fit(df.loc[early, f2], y[early])
    pt = m.predict_proba(df.loc[late, f2])[:, 1]
    k = BR_SHORT[br].replace("-", "")
    setm(f"TAUC{k}", f"{roc_auc_score(y[late], pt):.3f}")
    setm(f"TAP{k}", f"{average_precision_score(y[late], pt):.3f}")
    setm(f"TECE{k}", f"{ece(y[late], pt):.3f}")
    RESULTS[f"temporal|{br}"] = dict(auc=roc_auc_score(y[late], pt),
                                     ap=average_precision_score(y[late], pt), ece=ece(y[late], pt))

# ----------------------------------------------------------------------------
# 7. Figures: (a) ROC-AUC with CI per branch/model, (b) reliability before/after (GB, No-PV & Full)
# ----------------------------------------------------------------------------
log("figures")
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"width_ratios": [1.25, 1, 1]})
cols = {"Logistic Regression": "#7f7f7f", "Decision Tree": "#d98e04",
        "Random Forest": "#2e8b57", "Gradient Boosting": "#b22222"}
for bi, br in enumerate(BRANCHES):
    for j, name in enumerate(MODELS):
        r = R[(br, name)]; lo, hi = ci(BOOT[(br, name)]["ap"])
        x = bi + (j - 1.5) * 0.18
        ax[0].errorbar(x, r["ap"], yerr=[[r["ap"] - lo], [hi - r["ap"]]], fmt="o", ms=3,
                       color=cols[name], capsize=2, lw=0.8, label=MSHORT[name] if bi == 0 else None)
ax[0].axhline(yte.mean(), color="k", ls=":", lw=0.7)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([BR_SHORT[b] for b in BRANCHES])
ax[0].set_ylabel("Test PR-AUC (95% CI)"); ax[0].set_title("(a) PR-AUC by feature branch")
ax[0].legend(ncol=4, loc="upper left", fontsize=5.5, handletextpad=0.2, columnspacing=0.6)
ax[0].set_ylim(0, 1.0); ax[0].grid(axis="y", alpha=0.25)
for a, br in zip(ax[1:], ["NOPV", "FULL"]):
    r = R[(br, GB)]
    for pp, lab, st in [(r["p_raw"], f"raw (ECE {r['ece_raw']:.3f})", "--"),
                        (r["p"], f"calibrated (ECE {r['ece']:.3f})", "-")]:
        edges = np.linspace(0, 1, 11); b = np.clip(np.digitize(pp, edges[1:-1]), 0, 9)
        xs = [pp[b == k].mean() for k in range(10) if (b == k).sum() >= 10]
        ys = [yte[b == k].mean() for k in range(10) if (b == k).sum() >= 10]
        a.plot(xs, ys, st, marker="o", ms=2.5, lw=1, label=lab)
    a.plot([0, 1], [0, 1], "k:", lw=0.7)
    a.set_xlabel("Predicted probability"); a.set_ylabel("Observed purchase rate")
    a.set_title(f"({'b' if br == 'NOPV' else 'c'}) Reliability, GB {BR_SHORT[br]}")
    a.legend(fontsize=5.5, loc="upper left"); a.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_models.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# 8. Explanations (TreeSHAP on the uncalibrated GB of each t_end branch)
# ----------------------------------------------------------------------------
log("shap")
import shap  # noqa: E402

SH = {}
for br in ["FULL", "NOPV", "START"]:
    raw = R[(br, GB)]["raw"]
    pre, clf = raw.named_steps["pre"], raw.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    Xt = pre.transform(df.loc[ite, R[(br, GB)]["feats"]])
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    SH[br] = (names, sv, np.abs(sv).mean(0))
    top = np.argsort(SH[br][2])[::-1]
    for i, kk in enumerate("ABCDE"):
        setm(f"Top{kk}{BR_SHORT[br].replace('-', '')}", tex(names[top[i]]))

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
for a, br, ttl in [(ax[0], "FULL", "(a) Full: global"), (ax[1], "NOPV", "(b) No-PV: global")]:
    names, sv, g = SH[br]; top = np.argsort(g)[::-1][:8]
    a.barh([names[i] for i in top][::-1], g[top][::-1], color="#3b6ea5")
    a.set_xlabel("mean |SHAP| (log-odds)"); a.set_title(ttl); a.tick_params(axis="y", labelsize=5.5)
names, sv, g = SH["NOPV"]
pN = R[("NOPV", GB)]["p"]
idx_mid = int(np.argmin(np.abs(pN - 0.45)))
c = sv[idx_mid]; o = np.argsort(np.abs(c))[::-1][:7][::-1]
ax[2].barh([names[j] for j in o], c[o], color=["#b22222" if v > 0 else "#3b6ea5" for v in c[o]])
ax[2].axvline(0, color="k", lw=0.6); ax[2].tick_params(axis="y", labelsize=5.5)
ax[2].set_title(f"(c) No-PV local, $\\hat p$={pN[idx_mid]:.2f}"); ax[2].set_xlabel("SHAP value")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# 9. Propensity tiers on calibrated probabilities (GB, each branch)
# ----------------------------------------------------------------------------
HI, LO = 0.50, 0.15
lines = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Branch & Tier ($\hat p$, calibrated) & Sessions (\%) & Buy rate (\%) & Captured (\%) \\",
         r"\midrule"]
labels = {"High": rf"High ($\geq${HI:.2f})", "Medium": rf"Medium ({LO:.2f}--{HI:.2f})",
          "Low": rf"Low ($<${LO:.2f})"}
for bi, br in enumerate(["FULL", "NOPV", "START"]):
    p = R[(br, GB)]["p"]
    tier = np.where(p >= HI, "High", np.where(p >= LO, "Medium", "Low"))
    for j, t in enumerate(["High", "Medium", "Low"]):
        m = tier == t
        share = 100 * m.mean(); rate = 100 * yte[m].mean() if m.any() else float("nan")
        cap = 100 * yte[m].sum() / yte.sum()
        rate_s = f"{rate:.1f}" if m.any() else "--"
        lines.append(f"{BR_SHORT[br] if j == 0 else ''} & {labels[t]} & {share:.1f} & {rate_s} & {cap:.1f} \\\\")
        k = BR_SHORT[br].replace("-", "")
        setm(f"Share{t}{k}", f"{share:.1f}"); setm(f"Rate{t}{k}", rate_s); setm(f"Cap{t}{k}", f"{cap:.1f}")
        RESULTS[f"tier|{br}|{t}"] = dict(share=share, rate=rate, captured=cap)
    if bi < 2:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_tiers.tex", "w").write("\n".join(lines))

# ----------------------------------------------------------------------------
# 10. Scenario analysis (ASSUMED effects; no intervention is observed in the data)
#     Strategy A: top-K by p;  Strategy B: top-K by p(1-p);  K = 20% of sessions
#     Assumed gain  dp(p; w) = tau * [ (1-w)(1-p) + w p ]   (capped at 1-p)
#         w = 0 -> headroom model, w = 1 -> proportional model (V1's two cases)
# ----------------------------------------------------------------------------
log("scenario")
V, BUDGET, TAU, COST = 50.0, 0.20, 0.10, 1.0
W = np.linspace(0, 1, 21)
setm("OrderValue", f"{V:.0f}"); setm("Budget", f"{int(BUDGET * 100)}")
setm("TauVal", f"{TAU:.2f}"); setm("CostVal", f"{COST:.0f}")


def dp(p, w):
    return np.minimum(TAU * ((1 - w) * (1 - p) + w * p), 1 - p)


def strat_values(p, w, cost=COST):
    n = len(p); K = int(BUDGET * n)
    sA = np.argsort(-p, kind="stable")[:K]; sB = np.argsort(-(p * (1 - p)), kind="stable")[:K]
    out = {}
    for S, s in [("A", sA), ("B", sB)]:
        inc = dp(p[s], w).sum()
        out[S] = dict(inc=inc * 1000 / n, net=(inc * V - cost * K) * 1000 / n,
                      be=inc * V / K)
    out["overlap"] = len(set(sA) & set(sB)) / K
    return out


fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.0))
for br, colr in [("FULL", "#b22222"), ("NOPV", "#2e8b57"), ("START", "#3b6ea5")]:
    p = R[(br, GB)]["p"]; k = BR_SHORT[br].replace("-", "")
    diff = np.array([[strat_values(p[b], w)["B"]["net"] - strat_values(p[b], w)["A"]["net"]
                      for w in W] for b in BOOT_IDX[:500]])
    point = np.array([strat_values(p, w)["B"]["net"] - strat_values(p, w)["A"]["net"] for w in W])
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    ax[1].plot(W, point, color=colr, lw=1, label=BR_SHORT[br])
    ax[1].fill_between(W, lo, hi, color=colr, alpha=0.15, lw=0)
    # crossing weight w* where B and A tie
    s = np.sign(point)
    cross = [W[i] - point[i] * (W[i + 1] - W[i]) / (point[i + 1] - point[i])
             for i in range(len(W) - 1) if s[i] != s[i + 1]]
    same_sets = strat_values(p, 0)["overlap"] == 1.0
    setm(f"Wstar{k}", "n/a" if same_sets else (f"{cross[0]:.2f}" if cross else "none"))
    RESULTS[f"scenario|{br}|wstar"] = None if (same_sets or not cross) else float(cross[0])
    # capture of buyers / precision within the top-20% budget (ranking quality at the budget)
    K20 = int(BUDGET * len(p)); top20 = np.argsort(-p, kind="stable")[:K20]
    setm(f"CapTop{k}", f"{100 * yte[top20].sum() / yte.sum():.1f}")
    setm(f"PrecTop{k}", f"{100 * yte[top20].mean():.1f}")
    RESULTS[f"budget|{br}"] = dict(capture=float(yte[top20].sum() / yte.sum()),
                                   precision=float(yte[top20].mean()))
    for w, wn in [(0.0, "H"), (1.0, "P")]:
        sv_ = strat_values(p, w)
        bd = np.array([strat_values(p[b], w)["B"]["net"] - strat_values(p[b], w)["A"]["net"]
                       for b in BOOT_IDX[:1000]])
        for S in "AB":
            setm(f"Inc{S}{wn}{k}", f"{sv_[S]['inc']:.1f}")
            setm(f"Net{S}{wn}{k}", f"{sv_[S]['net']:.0f}")
            setm(f"BE{S}{wn}{k}", f"{sv_[S]['be']:.2f}")
        dlo, dhi = np.percentile(bd, [2.5, 97.5])
        # random-targeting reference: expected value of contacting K random sessions
        Kn = int(BUDGET * len(p)); inc_r = dp(p, w).mean() * Kn
        setm(f"IncR{wn}{k}", f"{inc_r * 1000 / len(p):.1f}")
        setm(f"NetR{wn}{k}", f"{(inc_r * V - COST * Kn) * 1000 / len(p):.0f}")
        fmt0 = lambda v: f"{v:+.0f}" if abs(v) >= 0.5 else "0"
        setm(f"DNet{wn}{k}", fmt0(sv_['B']['net'] - sv_['A']['net']))
        setm(f"DNetlo{wn}{k}", fmt0(dlo)); setm(f"DNethi{wn}{k}", fmt0(dhi))
        setm(f"PBgtA{wn}{k}", f"{(bd > 0).mean():.2f}")
        RESULTS[f"scenario|{br}|w={w}"] = dict(A=sv_["A"], B=sv_["B"], diff_ci=[dlo, dhi],
                                              random_net=(inc_r * V - COST * Kn) * 1000 / len(p),
                                              p_B_gt_A=float((bd > 0).mean()))
    setm(f"Overlap{k}", f"{100 * strat_values(p, 0)['overlap']:.0f}")
    # net value vs cost for the Full branch (A, B and random reference) -> panel (a)
    if br == "FULL":
        costs = np.linspace(0, 5, 51); Kn = int(BUDGET * len(p))
        for w, ls in [(0.0, "-"), (1.0, "--")]:
            for S, c2 in [("A", "#b22222"), ("B", "#2e8b57")]:
                ys = [strat_values(p, w, c)[S]["net"] for c in costs]
                ax[0].plot(costs, ys, color=c2, ls=ls, lw=1,
                           label=f"{S}, {'headroom' if w == 0 else 'proportional'}")
            ys = [(dp(p, w).mean() * Kn * V - c * Kn) * 1000 / len(p) for c in costs]
            ax[0].plot(costs, ys, color="#7f7f7f", ls=ls, lw=1,
                       label=f"Random, {'headroom' if w == 0 else 'proportional'}")
ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.25)
ax[0].set_xlabel("Cost per intervention (USD, assumed)"); ax[0].set_ylabel("Net value / 1,000 sessions")
ax[0].set_title("(a) Full branch: net value vs cost (assumed)"); ax[0].set_ylim(-950, 1500)
ax[0].legend(fontsize=5, ncol=2, loc="upper right")
ax[1].axhline(0, color="k", lw=0.6); ax[1].grid(alpha=0.25)
ax[1].set_xlabel("Effect mix $w$ (0 = headroom, 1 = proportional)")
ax[1].set_ylabel("Net(B) $-$ Net(A) / 1,000 sessions")
ax[1].set_title(f"(b) B minus A at \\${COST:.0f}/contact, 95% bootstrap band")
ax[1].legend(fontsize=5.5, title="No-PV/Start: A = B (same sessions)", title_fontsize=5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_strategy.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# 11. Environment, macros, results
# ----------------------------------------------------------------------------
env = {"python": platform.python_version(), "platform": platform.platform(),
       "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
       "matplotlib": matplotlib.__version__, "shap": shap.__version__,
       "data_sha256": data_sha, "seed": SEED, "bootstrap_resamples": len(BOOT_IDX),
       "runtime_seconds": round(time.time() - T_START)}
with open(f"{OUT}/environment.txt", "w") as f:
    for k, v in env.items():
        f.write(f"{k}: {v}\n")
setm("PyVer", env["python"]); setm("SkVer", env["scikit-learn"]); setm("ShapVer", env["shap"])
setm("NpVer", env["numpy"]); setm("PdVer", env["pandas"]); setm("MplVer", env["matplotlib"])
setm("DataSHA", data_sha[:12]); setm("NBoot", f"{len(BOOT_IDX):,}")
RESULTS["environment"] = env
with open(f"{OUT}/results_macros.tex", "w") as f:
    f.write("% AUTO-GENERATED by pipeline.py - do not edit by hand\n")
    for k, v in MACROS.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(RESULTS, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("done ->", OUT)
summary = pd.DataFrame([dict(branch=b, model=n, auc=R[(b, n)]["auc"], ap=R[(b, n)]["ap"],
                             f1=R[(b, n)]["f1"], ece_raw=R[(b, n)]["ece_raw"], ece=R[(b, n)]["ece"],
                             thr=R[(b, n)]["thr"]) for (b, n) in R]).round(3)
print(summary.to_string(index=False))
