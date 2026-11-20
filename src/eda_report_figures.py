# -*- coding: utf-8 -*-
"""
BIL497 - Week 9: Exploratory data analysis for the progress report.
Datasets: UCI German Credit (german.data), Breast Cancer Wisconsin (scikit-learn), synthetic dataset.

Usage:
    python src/eda_report_figures.py data/german.data
Outputs (folder ./results/eda):
    fig1_class_distribution.png, fig2_breast_corr.png, fig3_german_status.png, fig4_synthetic_corr.png
    summary_table.csv, *_numeric_summary.csv, breast_high_corr_pairs.csv
Note: EDA is descriptive only; no preprocessing parameter is learned here (no leakage).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

OUT = os.path.join("results", "eda"); os.makedirs(OUT, exist_ok=True)
SEED = 42
C1, C2 = "#4f81bd", "#c0504d"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

GERMAN_COLS = ["status_checking", "duration_month", "credit_history", "purpose", "credit_amount",
               "savings", "employment_since", "installment_rate", "personal_status_sex",
               "other_debtors", "residence_since", "property", "age", "other_installment_plans",
               "housing", "existing_credits", "job", "num_liable", "telephone", "foreign_worker", "class"]


def load_german(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLS)
    df["class"] = df["class"].map({1: 0, 2: 1})          # 0 = good, 1 = bad (UCI: 1 good, 2 bad)
    return df, "class"


def load_breast():
    return load_breast_cancer(as_frame=True).frame, "target"   # 0 = malignant, 1 = benign


def make_synthetic(n=1000, d=10, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(d)]); df["y"] = y
    return df, "y"


def summary_row(name, df, t):
    feats = df.drop(columns=t)
    num = feats.select_dtypes("number").columns
    vc = df[t].value_counts().sort_index()
    return {"dataset": name, "rows": len(df), "features": feats.shape[1], "numerical": len(num),
            "categorical": feats.shape[1] - len(num), "missing": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "class_0": int(vc.iloc[0]), "class_1": int(vc.iloc[1]),
            "imbalance_ratio": round(vc.max() / vc.min(), 2)}


def main(german_path):
    gc, gt = load_german(german_path); bc, bt = load_breast(); sy, st = make_synthetic()

    # tables
    pd.DataFrame([summary_row("Breast Cancer Wisconsin", bc, bt), summary_row("German Credit", gc, gt),
                  summary_row("Synthetic", sy, st)]).to_csv(f"{OUT}/summary_table.csv", index=False)
    for name, df, t in [("breast", bc, bt), ("german", gc, gt), ("synthetic", sy, st)]:
        num = df.drop(columns=t).select_dtypes("number")
        s = num.describe().T; s["zeros"] = (num == 0).sum(); s["skewness"] = num.skew()
        s.round(3).to_csv(f"{OUT}/{name}_numeric_summary.csv")
    cm = bc.drop(columns=bt).corr().abs(); cols = cm.columns
    pairs = [(a, b, round(cm.loc[a, b], 3)) for i, a in enumerate(cols) for b in cols[i + 1:] if cm.loc[a, b] >= 0.9]
    pd.DataFrame(pairs, columns=["feature_1", "feature_2", "abs_r"]).sort_values("abs_r", ascending=False)\
        .to_csv(f"{OUT}/breast_high_corr_pairs.csv", index=False)

    # Figure 1: class distributions
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4))
    for ax, (df, t, labs, title) in zip(axs, [(bc, bt, ["malignant", "benign"], "Breast Cancer Wisconsin"),
                                               (gc, gt, ["good", "bad"], "German Credit"),
                                               (sy, st, ["label 0", "label 1"], "Synthetic")]):
        vc = df[t].value_counts().sort_index(); n = len(df)
        b = ax.bar(labs, vc.values, color=[C1, C2])
        ax.bar_label(b, labels=[f"{v}\n({v / n * 100:.1f}%)" for v in vc.values], fontsize=7.5)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, vc.max() * 1.35)
        ax.set_ylabel("samples" if ax is axs[0] else "")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_class_distribution.png", dpi=200); plt.close(fig)

    # Figure 2: Breast Cancer correlation matrix
    c = bc.drop(columns=bt).corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=5.5); ax.set_yticklabels(c.columns, fontsize=5.5)
    fig.colorbar(im, shrink=0.8).set_label("Pearson correlation", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_breast_corr.png", dpi=200); plt.close(fig)

    # Figure 3: German Credit bad-credit rate by checking / savings status
    spec = [("status_checking", {"A11": "A11\n< 0 DM", "A12": "A12\n0-200 DM", "A13": "A13\n>= 200 DM",
                                 "A14": "A14\nno account"}, "Checking account status"),
            ("savings", {"A61": "A61\n< 100", "A62": "A62\n100-500", "A63": "A63\n500-1000",
                         "A64": "A64\n>= 1000", "A65": "A65\nunknown/none"}, "Savings account (DM)")]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for ax, (col, names, title) in zip(axs, spec):
        g = gc.groupby(col)[gt].agg(["mean", "size"]).reindex(list(names))
        b = ax.bar([names[k] for k in g.index], g["mean"] * 100, color=C1)
        for r, (m, n) in zip(b, zip(g["mean"], g["size"])):
            ax.text(r.get_x() + r.get_width() / 2, m * 100 + 1, f"{m * 100:.1f}%\nn={n}", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 65); ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylabel("bad credit rate (%)" if ax is axs[0] else "")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_german_status.png", dpi=200); plt.close(fig)

    # Figure 4: synthetic data, correlation of each feature with the label
    r = sy.drop(columns=st).corrwith(sy[st])
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.bar(r.index, r.values, color=[C2 if k in ("x1", "x2", "x3") else C1 for k in r.index])
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("correlation with label")
    for i, v in enumerate(r.values):
        ax.text(i, v + (0.02 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylim(-0.35, 0.8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_synthetic_corr.png", dpi=200); plt.close(fig)

    # key numbers used in the report text
    print(pd.read_csv(f"{OUT}/summary_table.csv").to_string(index=False))
    print("Breast |r|>=0.9 pairs:", len(pairs))
    print("Breast zero counts:", {k: v for k, v in (bc.drop(columns=bt) == 0).sum().items() if v})
    print("German bad rate by checking:", gc.groupby("status_checking")[gt].mean().round(3).to_dict())
    print("German skew:", gc[["credit_amount", "duration_month"]].skew().round(2).to_dict(),
          "r =", round(gc["credit_amount"].corr(gc["duration_month"]), 3))
    print("German means by class:\n", gc.groupby(gt)[["duration_month", "credit_amount"]].mean().round(1))
    print("Synthetic r:", r.round(3).to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "german.data"))
