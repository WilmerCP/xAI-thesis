# An Analysis of Explainable AI Methods in Terms of Consistency, Faithfulness, and Vulnerability to Noise

Graduation project (BIL497), Department of Computer Engineering, İstanbul Medeniyet University.

**Team:** Emre Eriş, Wilmer Jesús Cuevas Perdomo
**Advisor:** Asst. Prof. Dr. Tamer Uçar

This project compares two explanation methods, **SHAP** and **LIME**, on two real datasets and one synthetic dataset. The explanations are evaluated with three criteria:

- **Consistency:** do SHAP and LIME find the same important features for the same prediction?
- **Stability:** does the explanation stay the same when LIME is run again or when small noise is added to the input?
- **Fidelity:** does the explanation show the information that the model really uses?

## Repository structure

```
data/       German Credit data file and its description (UCI)
src/        Python scripts
results/    Tables and figures produced by the scripts
```

## Datasets

| Dataset | Source | Size |
|---|---|---|
| German Credit | UCI Statlog (German Credit Data), file `data/german.data` | 1000 rows, 20 attributes |
| Breast Cancer Wisconsin (Diagnostic) | Loaded from scikit-learn (`load_breast_cancer`) | 569 rows, 30 features |
| Synthetic | Generated in the code: 10 features from N(0, 1), label = 1 if 3·x1 + 2·x2 − x3 > 0 (seed 42) | 1000 rows, 10 features |

## Setup

Python 3 is required.

```
pip install -r requirements.txt
```

## How to run

Run the commands from the main folder of the repository.

**Exploratory data analysis (week 9):**

```
python src/eda_report_figures.py data/german.data
```

This creates the figures and summary tables used in the progress report in `results/eda/`.

## Status

- [x] Datasets and exploratory data analysis
- [ ] Baseline setup (train/test split, scikit-learn Pipeline, Logistic Regression and Random Forest)
- [ ] Experiment script (5 random seeds, mean ± standard deviation, results saved to CSV)
- [ ] SHAP and LIME experiments: consistency, stability and fidelity tests

## Data sources and licenses

- Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0)
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B (CC BY 4.0)
