"""
Network Intrusion Detection System (NIDS) - UNSW-NB15 Dataset
=============================================================
Files expected:
  - UNSW_NB15_train_40k.csv
  - UNSW_NB15_test_10k.csv

Steps:
  i   — Exploratory Data Analysis
  ii  — Preprocessing (Label Encoding + Standard Scaling)
  iii — Modelling (Logistic Regression, Random Forest, Naive Bayes,
                   KNN, SVM, Gradient Boosting, DNN/MLP)
  iv  — Advanced Evaluation (Accuracy, Precision, Recall, F2, F2-Macro,
                              Confusion Matrix, Precision-Recall Curve)
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, confusion_matrix, precision_recall_curve,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\hp\Desktop\AI in cyber\UNSW_NB15_train_40k.csv"
TEST_PATH  = r"C:\Users\hp\Desktop\AI in cyber\UNSW_NB15_test_10k.csv"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Standardise column names to lowercase and strip whitespace
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns  = test_df.columns.str.strip().str.lower()

# Drop non-predictive columns if present
for col in ["id", "attack_cat"]:
    if col in train_df.columns:
        train_df.drop(columns=col, inplace=True)
    if col in test_df.columns:
        test_df.drop(columns=col, inplace=True)

print(f"  train_df : {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
print(f"  test_df  : {test_df.shape[0]:,} rows  x {test_df.shape[1]} cols")
print(f"  Label distribution (train):\n{train_df['label'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["proto", "state", "service"]
TARGET_COL       = "label"
RANDOM_STATE     = 42

# Numerical columns — auto-detected from dataframe
NUMERICAL_COLS = [
    c for c in train_df.columns
    if c not in CATEGORICAL_COLS + [TARGET_COL]
    and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP i — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP i — Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("UNSW-NB15 — EDA Overview", fontsize=16, fontweight="bold")

palette = {0: "#4C9BE8", 1: "#E8694C"}  # blue = normal, red = attack

# ── Plot 1: Label distribution countplot ──────────────────────────────────
ax1 = axes[0, 0]
sns.countplot(
    data=train_df, x=TARGET_COL, hue=TARGET_COL,
    palette={0: "#4C9BE8", 1: "#E8694C"},
    order=[0, 1], legend=False, ax=ax1
)
ax1.set_title("Label Distribution (Train Set)", fontweight="bold")
ax1.set_xlabel("Label  (0 = Normal | 1 = Attack)")
ax1.set_ylabel("Count")
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Normal (0)", "Attack (1)"])
for bar in ax1.patches:
    ax1.annotate(
        f"{int(bar.get_height()):,}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", fontsize=10
    )

# ── Plot 2: Histogram — dur (duration) by label ───────────────────────────
ax2 = axes[0, 1]
for lbl, color in palette.items():
    subset = train_df[train_df[TARGET_COL] == lbl]["dur"]
    ax2.hist(
        subset.clip(upper=subset.quantile(0.99)),
        bins=50, alpha=0.6, color=color,
        label=f"{'Normal' if lbl == 0 else 'Attack'} ({lbl})"
    )
ax2.set_title("Duration (dur) by Label", fontweight="bold")
ax2.set_xlabel("Duration (s)  [clipped at 99th pct]")
ax2.set_ylabel("Frequency")
ax2.legend()

# ── Plot 3: Histogram — sbytes (source bytes) by label ────────────────────
ax3 = axes[1, 0]
for lbl, color in palette.items():
    subset = train_df[train_df[TARGET_COL] == lbl]["sbytes"]
    ax3.hist(
        subset.clip(upper=subset.quantile(0.99)),
        bins=50, alpha=0.6, color=color,
        label=f"{'Normal' if lbl == 0 else 'Attack'} ({lbl})"
    )
ax3.set_title("Source Bytes (sbytes) by Label", fontweight="bold")
ax3.set_xlabel("sbytes  [clipped at 99th pct]")
ax3.set_ylabel("Frequency")
ax3.legend()

# ── Plot 4: Correlation heatmap of numerical features ─────────────────────
ax4 = axes[1, 1]
corr_cols   = NUMERICAL_COLS + [TARGET_COL]
corr_matrix = train_df[corr_cols].corr()
sns.heatmap(
    corr_matrix, ax=ax4, cmap="coolwarm", center=0,
    linewidths=0.3, annot=False, square=True,
    cbar_kws={"shrink": 0.75}
)
ax4.set_title("Correlation Heatmap — Numerical Features", fontweight="bold")
ax4.tick_params(axis="x", rotation=45, labelsize=6)
ax4.tick_params(axis="y", rotation=0,  labelsize=6)

plt.tight_layout()
plt.savefig(r"C:\Users\hp\Desktop\AI in cyber\eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plot saved -> eda_overview.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP ii — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP ii — Preprocessing")
print("=" * 60)

# ── 1. Label-encode categorical columns ───────────────────────────────────
# Encoders are fit ONLY on training data to prevent data leakage.
# Unseen categories in the test set are mapped to -1 instead of crashing.
encoders = {}
for col in CATEGORICAL_COLS:
    if col not in train_df.columns:
        print(f"  WARNING: '{col}' not found in dataframe, skipping.")
        continue
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    le_classes = list(le.classes_)
    test_df[col] = test_df[col].astype(str).apply(
        lambda x: le_classes.index(x) if x in le_classes else -1
    )
    encoders[col] = le
    print(f"  Encoded '{col}'  ->  {len(le_classes)} unique classes in train")

# ── 2. Separate features (X) and target (y) ───────────────────────────────
feature_cols = [c for c in train_df.columns if c != TARGET_COL]

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET_COL]

print(f"\n  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")

# ── 3. Feature scaling with StandardScaler ────────────────────────────────
#
# WHY SCALE?
#   Distance-based models (KNN) and margin-based models (SVM) are highly
#   sensitive to feature magnitude — a feature ranging in the thousands will
#   dominate the distance metric over one ranging 0-1, producing biased
#   predictions. StandardScaler normalises each feature to zero mean and unit
#   variance, putting all features on equal footing. Tree-based models like
#   Random Forest are scale-invariant, but scaling keeps the pipeline
#   consistent and ready for KNN/SVM experiments.
#
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)        # apply same params to test

print("\n  StandardScaler applied — features normalised to mean=0, std=1")
print(f"  Verification — first feature: mean={X_train_scaled[:, 0].mean():.4f}, "
      f"std={X_train_scaled[:, 0].std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP iii — ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP iii — Training All Models")
print("=" * 60)

all_models = {
    # ── Baseline ──────────────────────────────────────────────────────────
    "Logistic Regression":  LogisticRegression(
                                max_iter=1000,
                                random_state=RANDOM_STATE,
                                n_jobs=-1),
    "Random Forest":        RandomForestClassifier(
                                n_estimators=100,
                                random_state=RANDOM_STATE,
                                n_jobs=-1),
    # ── Non-tree ──────────────────────────────────────────────────────────
    "Naive Bayes":          GaussianNB(),
    "KNN (k=5)":            KNeighborsClassifier(
                                n_neighbors=5,
                                n_jobs=-1),
    "SVM":                  SVC(
                                kernel="rbf",
                                probability=True,       # needed for PR curve
                                random_state=RANDOM_STATE),
    # ── Tree-based ────────────────────────────────────────────────────────
    "Gradient Boosting":    GradientBoostingClassifier(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=4,
                                random_state=RANDOM_STATE),
    # ── Deep Learning (MLP) ───────────────────────────────────────────────
    "DNN (MLP)":            MLPClassifier(
                                hidden_layer_sizes=(64, 32),  # two hidden layers
                                activation="relu",
                                solver="adam",
                                max_iter=300,
                                early_stopping=True,
                                random_state=RANDOM_STATE),
}

trained_models  = {}   # name -> fitted model  (kept for PR curves)
all_predictions = {}   # name -> y_pred array

for name, model in all_models.items():
    print(f"  Training {name} ...")
    model.fit(X_train_scaled, y_train)
    trained_models[name]  = model
    all_predictions[name] = model.predict(X_test_scaled)
    print(f"    Done.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP iv — ADVANCED EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP iv — Advanced Evaluation")
print("=" * 60)

# ── Build full metrics table ───────────────────────────────────────────────
full_results = []
for name, y_pred in all_predictions.items():
    full_results.append({
        "Model":      name,
        "Accuracy":   accuracy_score (y_test, y_pred),
        "Precision":  precision_score(y_test, y_pred, zero_division=0),
        "Recall":     recall_score   (y_test, y_pred),
        "F2 Score":   fbeta_score    (y_test, y_pred, beta=2),
        "F2 Macro":   fbeta_score    (y_test, y_pred, beta=2, average="macro"),
    })

metrics_df = pd.DataFrame(full_results).set_index("Model")

# ── Print comparative table ────────────────────────────────────────────────
print("\n" + "-" * 75)
print(f"{'MODEL COMPARISON':^75}")
print("-" * 75)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
      f"{'F2 Score':>10} {'F2 Macro':>10}")
print("-" * 75)
for name, row in metrics_df.iterrows():
    print(f"{name:<22} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>10.4f} {row['F2 Score']:>10.4f} "
          f"{row['F2 Macro']:>10.4f}")
print("-" * 75)

# Best model by F2 Score (recall-weighted — missing an attack is costly)
best_model_name = metrics_df["F2 Score"].idxmax()
print(f"\n  Best model by F2 Score : {best_model_name} "
      f"({metrics_df.loc[best_model_name, 'F2 Score']:.4f})\n")

# ── Confusion Matrix + Precision-Recall Curve ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step iv — Advanced Evaluation", fontsize=14, fontweight="bold")

# Confusion Matrix — best model
ax_cm      = axes[0]
best_preds = all_predictions[best_model_name]
cm         = confusion_matrix(y_test, best_preds)
disp       = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=["Normal", "Attack"])
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(f"Confusion Matrix\n{best_model_name}", fontweight="bold")

# Annotate quadrant labels (TN / FP / FN / TP)
for i, j, label in [(0,0,"TN"), (0,1,"FP"), (1,0,"FN"), (1,1,"TP")]:
    ax_cm.text(j, i - 0.28, label,
               ha="center", va="center",
               fontsize=9, color="grey", style="italic")

# Precision-Recall Curve — top 3 models by F2 Score
ax_pr      = axes[1]
top3_names = metrics_df["F2 Score"].nlargest(3).index.tolist()
pr_colors  = ["#E8694C", "#4C9BE8", "#2ECC71"]

for color, name in zip(pr_colors, top3_names):
    model  = trained_models[name]
    scores = (model.predict_proba(X_test_scaled)[:, 1]
              if hasattr(model, "predict_proba")
              else model.decision_function(X_test_scaled))
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, scores)
    f2_val = metrics_df.loc[name, "F2 Score"]
    ax_pr.plot(rec_vals, prec_vals, color=color, lw=2,
               label=f"{name}  (F2={f2_val:.3f})")

# Random-classifier baseline
baseline = y_test.sum() / len(y_test)
ax_pr.axhline(y=baseline, color="grey", linestyle="--", lw=1,
              label=f"Random baseline ({baseline:.2f})")

ax_pr.set_title("Precision-Recall Curve — Top 3 Models", fontweight="bold")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(loc="lower left", fontsize=9)
ax_pr.set_xlim([0, 1])
ax_pr.set_ylim([0, 1.05])
ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\hp\Desktop\AI in cyber\step4_evaluation.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Evaluation plots saved -> step4_evaluation.png")
print("\nAll steps complete.")