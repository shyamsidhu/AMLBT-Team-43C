"""
Beyond Clicks - AMLBT Project (Team 43, Section C)
Full pipeline: EDA -> models -> evaluation -> SHAP -> prioritisation -> simulation.

USAGE
    1. Download online_shoppers_intention.csv from the UCI repository
       (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
    2. pip install numpy pandas scikit-learn matplotlib shap
    3. python pipeline.py online_shoppers_intention.csv
    4. Upload EVERYTHING inside the generated folder  overleaf_files/  to Overleaf
       (next to main.tex). The paper picks up all numbers/figures automatically.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             roc_curve, precision_score, recall_score, f1_score,
                             confusion_matrix, brier_score_loss)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
DATA = sys.argv[1] if len(sys.argv) > 1 else "online_shoppers_intention.csv"
OUT = "overleaf_files"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})

MACROS = {}


def tex(s):
    return str(s).replace("_", r"\_")


def setm(name, value):
    MACROS[name] = value


# ----------------------------------------------------------------------------
# 1. Data loading and cleaning
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA)
n_raw = len(df)
n_missing = int(df.isna().sum().sum())
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)

CAT = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
for c in CAT:
    df[c] = df[c].astype(str)
NUM = [c for c in df.columns if c not in CAT + ["Revenue"]]
y = df["Revenue"].values
X = df[NUM + CAT]

setm("NRaw", f"{n_raw:,}")
setm("NMissing", n_missing)
setm("NDup", n_dup)
setm("NClean", f"{len(df):,}")
setm("NPos", f"{int(y.sum()):,}")
setm("PosRate", f"{100 * y.mean():.1f}")

# ----------------------------------------------------------------------------
# 2. Exploratory analysis figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.1))
vt = df.groupby("VisitorType")["Revenue"].mean().mul(100)
ax[0].bar([v.replace("_", "\n") for v in vt.index], vt.values, color="#3b6ea5")
ax[0].set_ylabel("Purchase rate (%)")
ax[0].set_title("(a) By visitor type")
order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
mm = df.groupby("Month")["Revenue"].mean().mul(100)
mm = mm.reindex([m for m in order if m in mm.index])
ax[1].bar(mm.index, mm.values, color="#3b6ea5")
ax[1].set_title("(b) By month")
ax[1].tick_params(axis="x", rotation=60)
pv = df.groupby(df["PageValues"] > 0)["Revenue"].mean().mul(100)
ax[2].bar(["PageValues = 0", "PageValues > 0"], pv.reindex([False, True]).values, color="#3b6ea5")
ax[2].set_title("(c) By page value")
for a in ax:
    a.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_eda.png", dpi=300)
plt.close()
setm("RateNew", f"{vt.get('New_Visitor', np.nan):.1f}")
setm("RateRet", f"{vt.get('Returning_Visitor', np.nan):.1f}")
setm("RatePVzero", f"{pv.get(False, np.nan):.1f}")
setm("RatePVpos", f"{pv.get(True, np.nan):.1f}")
if "Nov" in mm.index:
    setm("RateNov", f"{mm['Nov']:.1f}")

# ----------------------------------------------------------------------------
# 3. Train/test split, models, out-of-fold threshold selection
# ----------------------------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
setm("NTrain", f"{len(Xtr):,}")
setm("NTest", f"{len(Xte):,}")


def make_pre(num, cat, scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                            class_weight="balanced_subsample", random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, subsample=0.8, random_state=SEED),
}


def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.nanargmax(f1[:-1])])


def run(name, est, num, cat, keep=None):
    scale = name == "Logistic Regression"
    pipe = Pipeline([("pre", make_pre(num, cat, scale)), ("clf", est)])
    cols = num + cat
    oof = cross_val_predict(pipe, Xtr[cols], ytr, cv=skf, method="predict_proba")[:, 1]
    thr = best_f1_threshold(ytr, oof)
    pipe.fit(Xtr[cols], ytr)
    p = pipe.predict_proba(Xte[cols])[:, 1]
    yh = (p >= thr).astype(int)
    return dict(name=name, pipe=pipe, p=p, thr=thr, cols=cols,
                auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                prec=precision_score(yte, yh, zero_division=0), rec=recall_score(yte, yh),
                f1=f1_score(yte, yh), brier=brier_score_loss(yte, p),
                cm=confusion_matrix(yte, yh), cv_ap=average_precision_score(ytr, oof))


results = {}
for n, m in MODELS.items():
    print("Training", n)
    results[n] = run(n, m, NUM, CAT)

# Ablation: gradient boosting without PageValues (possible target-proximal feature)
num_nopv = [c for c in NUM if c != "PageValues"]
abl = run("GB w/o PageValues", GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED),
    num_nopv, CAT)

# ----- model comparison table (LaTeX) ---------------------------------------
rows = list(results.values()) + [abl]
best = {k: max(r[k] for r in rows) for k in ["auc", "ap", "prec", "rec", "f1"]}


def fmt(r, k):
    s = f"{r[k]:.3f}"
    return rf"\textbf{{{s}}}" if abs(r[k] - best[k]) < 1e-12 else s


lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & ROC-AUC & PR-AUC & Prec. & Recall & F1 \\", r"\midrule"]
for r in rows:
    if r["name"].startswith("GB w/o"):
        lines.append(r"\midrule")
    lines.append(f"{tex(r['name'])} & " + " & ".join(fmt(r, k) for k in
                 ["auc", "ap", "prec", "rec", "f1"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_models.tex", "w").write("\n".join(lines))

gb = results["Gradient Boosting"]
rf = results["Random Forest"]
lr = results["Logistic Regression"]
for key, r in [("GB", gb), ("RF", rf), ("LR", lr)]:
    setm(f"AUC{key}", f"{r['auc']:.3f}")
    setm(f"AP{key}", f"{r['ap']:.3f}")
    setm(f"F{key}", f"{r['f1']:.3f}")
    setm(f"Prec{key}", f"{r['prec']:.3f}")
    setm(f"Rec{key}", f"{r['rec']:.3f}")
setm("BrierGB", f"{gb['brier']:.3f}")
setm("ThrGB", f"{gb['thr']:.2f}")
setm("AUCnoPV", f"{abl['auc']:.3f}")
setm("APnoPV", f"{abl['ap']:.3f}")
setm("FnoPV", f"{abl['f1']:.3f}")
tn, fp, fn, tp = gb["cm"].ravel()
setm("TP", int(tp)); setm("FP", int(fp)); setm("FN", int(fn)); setm("TN", int(tn))
setm("BaseAP", f"{yte.mean():.3f}")

# ----------------------------------------------------------------------------
# 4. ROC and precision-recall curves
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
colors = ["#7f7f7f", "#d98e04", "#2e8b57", "#b22222"]
for (n, r), c in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(yte, r["p"])
    ax[0].plot(fpr, tpr, color=c, label=f"{n} ({r['auc']:.3f})")
    pr, rc, _ = precision_recall_curve(yte, r["p"])
    ax[1].plot(rc, pr, color=c, label=f"{n} ({r['ap']:.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=0.7)
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("(a) ROC curves"); ax[0].legend(loc="lower right", fontsize=5.5)
ax[1].axhline(yte.mean(), color="k", ls=":", lw=0.7)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("(b) Precision-recall curves"); ax[1].legend(loc="upper right", fontsize=5.5)
# threshold sweep for the primary model
ths = np.linspace(0.05, 0.9, 60)
P = [precision_score(yte, gb["p"] >= t, zero_division=0) for t in ths]
R = [recall_score(yte, gb["p"] >= t) for t in ths]
F = [f1_score(yte, gb["p"] >= t) for t in ths]
ax[2].plot(ths, P, label="Precision"); ax[2].plot(ths, R, label="Recall")
ax[2].plot(ths, F, label="F1")
ax[2].axvline(gb["thr"], color="k", ls="--", lw=0.7)
ax[2].set_xlabel("Decision threshold"); ax[2].set_title("(c) Threshold trade-off (GB)")
ax[2].legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_curves.png", dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# 5. Explainability: SHAP (fallback: permutation importance)
# ----------------------------------------------------------------------------
pre = gb["pipe"].named_steps["pre"]
clf = gb["pipe"].named_steps["clf"]
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
Xte_t = pre.transform(Xte[gb["cols"]])
p_te = gb["p"]
try:
    import shap
    sv = shap.TreeExplainer(clf).shap_values(Xte_t)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    glob = np.abs(sv).mean(0)
    method = "SHAP"
except Exception as e:  # pragma: no cover
    print("SHAP unavailable, using permutation importance:", e)
    pi = permutation_importance(gb["pipe"], Xte[gb["cols"]], yte, scoring="average_precision",
                                n_repeats=10, random_state=SEED)
    glob = pi.importances_mean
    sv = None
    method = "permutation"

top = np.argsort(glob)[::-1][:10]
setm("ImpMethod", method)
for i, k in enumerate(["A", "B", "C", "D", "E"]):
    setm(f"Top{k}", tex(names[top[i]]))

fig, ax = plt.subplots(1, 3 if sv is not None else 1, figsize=(7.1, 2.5),
                       gridspec_kw={"width_ratios": [1.5, 1, 1]} if sv is not None else None)
ax = np.atleast_1d(ax)
ax[0].barh([names[i] for i in top][::-1], glob[top][::-1], color="#3b6ea5")
ax[0].set_xlabel("mean |SHAP| (log-odds)" if sv is not None else "Permutation importance (AP drop)")
ax[0].set_title("(a) Global importance (top 10)")
if sv is not None:
    ax[0].tick_params(axis="y", labelsize=6)
if sv is not None:
    picks = [("(b) Highest-prob. session", int(np.argmax(p_te))),
             ("(c) Mid-prob. session", int(np.argmin(np.abs(p_te - 0.4))))]
    for a, (title, idx) in zip(ax[1:], picks):
        contrib = sv[idx]
        o = np.argsort(np.abs(contrib))[::-1][:6][::-1]
        a.tick_params(axis="y", labelsize=6)
        a.barh([names[j] for j in o], contrib[o],
               color=["#b22222" if v > 0 else "#3b6ea5" for v in contrib[o]])
        a.axvline(0, color="k", lw=0.6)
        a.set_title(f"{title}\n$p$={p_te[idx]:.2f}", fontsize=7)
        a.set_xlabel("SHAP value")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_shap.png", dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# 6. Propensity tiers
# ----------------------------------------------------------------------------
HI, LO = 0.50, 0.15
tier = np.where(p_te >= HI, "High", np.where(p_te >= LO, "Medium", "Low"))
lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Tier ($\hat p$) & Sessions (\%) & Buy rate (\%) & Captured (\%) \\",
         r"\midrule"]
labels = {"High": rf"High ($\geq${HI:.2f})", "Medium": rf"Medium ({LO:.2f}--{HI:.2f})",
          "Low": rf"Low ($<${LO:.2f})"}
for t in ["High", "Medium", "Low"]:
    m = tier == t
    share = 100 * m.mean()
    rate = 100 * yte[m].mean() if m.any() else 0
    cap = 100 * yte[m].sum() / yte.sum()
    lines.append(f"{labels[t]} & {share:.1f} & {rate:.1f} & {cap:.1f} \\\\")
    setm(f"Share{t}", f"{share:.1f}"); setm(f"Rate{t}", f"{rate:.1f}"); setm(f"Cap{t}", f"{cap:.1f}")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_tiers.tex", "w").write("\n".join(lines))

# ----------------------------------------------------------------------------
# 7. Business simulation: Strategy A vs Strategy B
# ----------------------------------------------------------------------------
V = 50.0          # assumed average order value (hypothetical, USD)
BUDGET = 0.20     # intervene on top 20 % of sessions
TAU = 0.10        # headroom model: intervention closes 10 % of the remaining gap
LAM = 0.10        # proportional model: intervention lifts p by 10 % (relative)
n = len(p_te)
K = int(BUDGET * n)
selA = np.argsort(-p_te)[:K]
selB = np.argsort(-(p_te * (1 - p_te)))[:K]


def delta(p, model):
    return TAU * (1 - p) if model == "headroom" else np.minimum(LAM * p, 1 - p)


costs = np.linspace(0, 8, 81)
fig, ax = plt.subplots(figsize=(3.45, 2.4))
sty = {("A", "headroom"): ("#b22222", "-"), ("B", "headroom"): ("#2e8b57", "-"),
       ("A", "proportional"): ("#b22222", "--"), ("B", "proportional"): ("#2e8b57", "--")}
for S, sel in [("A", selA), ("B", selB)]:
    for mod in ["headroom", "proportional"]:
        inc = delta(p_te[sel], mod).sum()
        net = (inc * V - costs * K) * 1000 / n
        c, ls = sty[(S, mod)]
        ax.plot(costs, net, color=c, ls=ls, label=f"Strategy {S}, {mod}")
        key = S + ("H" if mod == "headroom" else "P")
        setm(f"Inc{key}", f"{inc * 1000 / n:.1f}")
        setm(f"BE{key}", f"{inc * V / K:.2f}")
        setm(f"Net{key}", f"{(inc * V - 1.0 * K) * 1000 / n:.0f}")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Cost per intervention (USD)")
ax.set_ylabel("Net value per 1,000 sessions (USD)")
ax.legend(fontsize=5.5)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_strategy.png", dpi=300)
plt.close()
setm("OrderValue", f"{V:.0f}"); setm("Budget", f"{int(BUDGET * 100)}")
setm("TauVal", f"{TAU:.2f}"); setm("LamVal", f"{LAM:.2f}")
setm("OverlapAB", f"{100 * len(set(selA) & set(selB)) / K:.0f}")

# ----------------------------------------------------------------------------
# 8. Write LaTeX macro file
# ----------------------------------------------------------------------------
with open(f"{OUT}/results_macros.tex", "w") as f:
    f.write("% AUTO-GENERATED by pipeline.py - do not edit by hand\n")
    for k, v in MACROS.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\nDone. Upload the contents of", OUT, "to Overleaf.")
print(pd.DataFrame([{k: r[k] for k in ["name", "auc", "ap", "prec", "rec", "f1", "thr"]}
                    for r in rows]).round(3).to_string(index=False))
