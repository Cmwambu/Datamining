"""Supervised ML model: predicting perception of university leaders (UEAB).
CRISP-DM: preparation -> modelling -> evaluation. Run: python train_model.py"""
import json, warnings, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
SEED = 42
DATA = "data/survey_responses.csv"
LIKERT = ["perception_trust", "perception_credibility",
          "perception_competence", "perception_approachability"]
CLASSES = ["negative", "neutral", "positive"]

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA)
print(f"Raw records: {len(df)}")
df = df.drop_duplicates()
df = df.dropna(subset=LIKERT)                       # target cannot be imputed
print(f"After removing duplicates / missing target: {len(df)}")

# ---------- 2. Target: Likert perception index ----------
# Mean of the 4 items (1-5): <2.5 negative, 2.5-3.5 neutral, >3.5 positive.
# The four items are DROPPED from the features to avoid target leakage.
index = df[LIKERT].mean(axis=1)
df["perception"] = pd.cut(index, [0, 2.5, 3.5, 5.01], labels=CLASSES, right=False)
df["perception"] = df["perception"].astype(str)
X, y = df.drop(columns=LIKERT + ["perception"]), df["perception"]
print("Class distribution:\n", y.value_counts().to_string(), "\n")

# ---------- 3. Outlier check (IQR) on numeric columns ----------
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
for c in num_cols:
    q1, q3 = X[c].quantile([.25, .75]); iqr = q3 - q1
    n_out = ((X[c] < q1 - 1.5 * iqr) | (X[c] > q3 + 1.5 * iqr)).sum()
    print(f"IQR outliers in {c}: {n_out}")
X[num_cols].plot(kind="box", subplots=True, layout=(1, len(num_cols)), figsize=(4 * len(num_cols), 4))
plt.tight_layout(); plt.savefig("results/boxplots.png", dpi=150); plt.close()

# ---------- 4. Split 80/20 ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

pre = ColumnTransformer([
    ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# SMOTE only if imbalanced (smallest class < 70% of largest); applied inside the
# pipeline so it only ever touches training folds / training data.
counts = y_tr.value_counts()
use_smote = counts.min() < 0.7 * counts.max()
k = max(1, min(5, counts.min() - 1))
print(f"SMOTE applied: {use_smote}\n")

models = {
    "Logistic Regression (baseline)": (LogisticRegression(max_iter=2000, random_state=SEED),
                                       {"clf__C": [0.1, 1, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [3, 5, 10, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]}),
    "SVM": (SVC(probability=True, random_state=SEED),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
}

cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, (clf, grid) in models.items():
    steps = [("pre", pre)] + ([("smote", SMOTE(k_neighbors=k, random_state=SEED))] if use_smote else []) + [("clf", clf)]
    gs = GridSearchCV(Pipeline(steps), grid, cv=cv, scoring="f1_weighted", n_jobs=-1).fit(X_tr, y_tr)
    pred, proba = gs.predict(X_te), gs.predict_proba(X_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    rows.append({"Model": name, "CV F1 (train)": gs.best_score_, "Accuracy": accuracy_score(y_te, pred),
                 "Precision": p, "Recall": r, "F1-score": f,
                 "ROC-AUC": roc_auc_score(y_te, proba, multi_class="ovr", average="weighted",
                                          labels=gs.classes_),
                 "Best params": json.dumps({k_.replace("clf__", ""): v for k_, v in gs.best_params_.items()})})
    fitted[name] = gs
    print(f"Done: {name}")

res = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
res.to_csv("results/model_comparison.csv", index=False)
print("\n", res.drop(columns="Best params").round(3).to_string(index=False))

# ---------- 5. Best model: confusion matrix, importance ----------
best_name = res.iloc[0]["Model"]; best = fitted[best_name]
print(f"\nBest model: {best_name}")
ConfusionMatrixDisplay(confusion_matrix(y_te, best.predict(X_te), labels=best.classes_),
                       display_labels=best.classes_).plot(cmap="Blues")
plt.title(f"Confusion matrix: {best_name}"); plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

imp = permutation_importance(best, X_te, y_te, n_repeats=20, random_state=SEED, scoring="f1_weighted")
fi = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
fi.to_csv("results/feature_importance.csv", header=["importance"])
fi.head(10)[::-1].plot(kind="barh", figsize=(7, 5)); plt.title("Top factors (permutation importance)")
plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=150); plt.close()
print("\nTop 10 factors:\n", fi.head(10).round(4).to_string())

joblib.dump(best, "results/best_model.joblib")
print("\nSaved outputs to results/")

# ---------- 6. Human-readable results file (upload this to GitHub) ----------
def md_table(d):
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)

with open("results/results_summary.md", "w") as f:
    f.write("# Results Summary\n\n")
    f.write(f"- Valid records after cleaning: {len(df)} (train {len(X_tr)}, test {len(X_te)})\n")
    f.write(f"- Class distribution: {y.value_counts().to_dict()}\n")
    f.write(f"- SMOTE applied: {use_smote}\n- Best model: **{best_name}**\n\n")
    f.write("## Model comparison (test set)\n\n")
    f.write(md_table(res.drop(columns="Best params").round(3)) + "\n\n")
    f.write("## Top 10 influencing factors (permutation importance)\n\n")
    f.write(md_table(fi.head(10).round(4).rename("importance").reset_index().rename(columns={"index": "factor"})) + "\n\n")
    f.write("![Confusion matrix](confusion_matrix.png)\n\n![Feature importance](feature_importance.png)\n")
print("Wrote results/results_summary.md")
