# Beyond Clicks — AMLBT Project, Team 43C (Version 2)

Explainable purchase-intent prediction on the UCI Online Shoppers Purchasing Intention dataset,
with an explicit scoring-time audit, calibrated probabilities, bootstrap uncertainty and a
scenario analysis that keeps assumed intervention effects separate from measured outcomes.

Team: Samarth Narang (B2026146), Shyam Sitharrth E (B2026125), Yash Bansal (B2026160),
Goa Institute of Management, Big Data Analytics, Section C.

## Reproduce every number in the paper

```bash
git clone https://github.com/shyamsidhu/AMLBT-Team-43C.git
cd AMLBT-Team-43C
pip install -r requirements.txt          # Python 3.13; exact versions pinned
python pipeline.py online_shoppers_intention.csv
```

Runtime: about 8 minutes on two CPU cores. Seed: 42.
All outputs go to `overleaf_files/`:

| File | Content |
|---|---|
| `results_macros.tex` | every number quoted in the paper, as LaTeX macros |
| `table_timing.tex`, `table_models.tex`, `table_tiers.tex` | Tables I–III |
| `fig_models.png`, `fig_shap.png`, `fig_strategy.png` | Figures 1–3 |
| `results.json` | all metrics, confidence intervals and scenario values |
| `feature_timing.csv` | the feature-timing audit |
| `environment.txt` | Python/package versions, platform, data SHA-256, runtime |

Copy the contents of `overleaf_files/` next to `main.tex` in Overleaf and recompile.

## Data

`online_shoppers_intention.csv` is the unmodified UCI file
(<https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>),
SHA-256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
The script removes the 125 exact duplicate rows (12,205 sessions remain).

## Feature timing (scoring timestamp)

`t0` = first page request of the session, `tend` = last page request.
The public file has one aggregated row per session and no page timestamps, so no
intermediate cutoff can be rebuilt. The `FEATURE_TIMING` dictionary in `pipeline.py`
defines three branches:

| Branch | Features | Scored at | Valid claim |
|---|---|---|---|
| Start | Month, SpecialDay, Weekend, VisitorType, OperatingSystems, Browser, Region, TrafficType | t0 | prospective (session start) |
| No-PV | Start + page counts/durations, BounceRates, ExitRates | tend | retrospective session classification |
| Full | No-PV + PageValues | tend | retrospective; PageValues is outcome-proximal |

## Method summary

* 80/20 stratified split; all fitting, calibration and thresholds use training data only.
* Four models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) on every branch.
* Isotonic calibration (5-fold, training only); F1-optimal threshold from nested out-of-fold
  calibrated predictions.
* 5-fold CV spread, 2,000-resample test bootstrap (paired across models), month-blocked forward
  check (train Feb–Oct, test Nov–Dec).
* TreeSHAP for Gradient Boosting; propensity tiers on calibrated probabilities.
* Scenario analysis with *assumed* effects `dp = tau*[(1-w)(1-p) + w*p]`, Strategies A (top p),
  B (top p(1-p)) and random targeting, with bootstrap bands. No uplift is observed in the data.

## Version history

* **v1** (first review): Full-feature models, PageValues ablation, SHAP, two-strategy simulation.
* **v2** (second review): feature-timing audit and three fully evaluated branches, calibration,
  confidence intervals, forward check, random-targeting reference, environment logging.
