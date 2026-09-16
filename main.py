import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, learning_curve, GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42

# ------------------------------------------------------------
# LOAD & INSPECT DATASET
# ------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")   # 0 = malignant, 1 = benign

print("Shape:", X.shape)
print(X.head())
print(X.describe())
print("Missing values:", X.isnull().sum().sum())
print("Target distribution:\n", y.value_counts())


# ------------------------------------------------------------
# TRAIN/TEST SPLIT (then lock train/test set)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ------------------------------------------------------------
# MODEL OVERFITTING
# ------------------------------------------------------------
overfit_model = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)
overfit_model.fit(X_train, y_train)

def report(model, X_, y_true, name):
    pred = model.predict(X_)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    print(f"{name:12s} | Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return acc, prec, rec, f1

print("\n--- Overfitting model ---")
report(overfit_model, X_train, y_train, "Train")
report(overfit_model, X_test, y_test, "Test")
print("Depth:", overfit_model.get_depth(), "| Leaves:", overfit_model.get_n_leaves())

# ------------------------------------------------------------
# K-CROSS VALIDATION: Compare K = 3, 5, 7
# ------------------------------------------------------------
print("\n--- K-Cross comparison (StratifiedKCross, accuracy) ---")
for k in [3, 5, 7]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(overfit_model, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"K={k:2d} | mean={scores.mean():.4f} std={scores.std():.4f} | folds={np.round(scores,3)}")

# ------------------------------------------------------------
# max_depth (bias-variance sweep) with 5-fold CV
# ------------------------------------------------------------
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
depths = list(range(1, 16))
train_scores, cv_means, cv_stds = [], [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    m.fit(X_train, y_train)
    train_scores.append(m.score(X_train, y_train))
    s = cross_val_score(m, X_train, y_train, cv=skf5, scoring="accuracy")
    cv_means.append(s.mean()); cv_stds.append(s.std())

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker='o', label="Training Accuracy")
plt.errorbar(depths, cv_means, yerr=cv_stds, marker='s', capsize=3, label="CV Mean (+/- std)")
plt.xlabel("max_depth"); plt.ylabel("Accuracy")
plt.title("Model Complexity vs Performance")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("complexity_vs_performance.png", dpi=130)
plt.close()

# ------------------------------------------------------------
# LEARNING CURVE (model overfitting)
# ------------------------------------------------------------
train_sizes, train_sc, val_sc = learning_curve(
    overfit_model, X_train, y_train, cv=skf5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE
)
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_sc.mean(axis=1), marker='o', label="Training score")
plt.plot(train_sizes, val_sc.mean(axis=1), marker='s', label="CV score")
plt.xlabel("Training set size"); plt.ylabel("Accuracy")
plt.title("Learning Curve - Overfit Decision Tree")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("learning_curve.png", dpi=130)
plt.close()

# ------------------------------------------------------------
# PIPELINE + GRIDSEARCHCV (hyperparameter tuning, Not Test Set)
# ------------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),   # fit chi tren Train, ben trong moi fold cua CV
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])
param_grid = {
    "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(pipe, param_grid, cv=skf5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
tuned_model = grid.best_estimator_

# ------------------------------------------------------------
# FINAL TEST ON TEST SET (only once)
# ------------------------------------------------------------
print("\n--- FINAL TEST EVALUATION ---")
print("Overfit model:"); report(overfit_model, X_test, y_test, "Test")
print("Tuned model:");   report(tuned_model, X_test, y_test, "Test")

results_table = pd.DataFrame({
    "Model": ["Overfit (max_depth=None)", "Tuned (GridSearchCV)"],
    "Train Acc": [overfit_model.score(X_train, y_train), tuned_model.score(X_train, y_train)],
    "CV Mean (5-fold)": [
        cross_val_score(overfit_model, X_train, y_train, cv=skf5).mean(),
        cross_val_score(tuned_model, X_train, y_train, cv=skf5).mean()
    ],
    "Test Acc": [overfit_model.score(X_test, y_test), tuned_model.score(X_test, y_test)],
})
print(results_table)

# ------------------------------------------------------------
# CONFUSION MATRIX + TREE PLOT
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ConfusionMatrixDisplay.from_estimator(overfit_model, X_test, y_test,
                                       display_labels=data.target_names, ax=axes[0], colorbar=False)
axes[0].set_title("Overfit Model (Test Set)")
ConfusionMatrixDisplay.from_estimator(tuned_model, X_test, y_test,
                                       display_labels=data.target_names, ax=axes[1], colorbar=False)
axes[1].set_title("Tuned Model (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=130)
plt.close()

plt.figure(figsize=(14, 7))
plot_tree(tuned_model.named_steps["model"], max_depth=3, feature_names=data.feature_names,
          class_names=data.target_names, filled=True, fontsize=7)
plt.title("Tuned Decision Tree - top 3 levels (full depth=%d)"
          % tuned_model.named_steps["model"].get_depth())
plt.tight_layout()
plt.savefig("tree_tuned.png", dpi=130)
plt.close()

print("\nDone. Plots saved: complexity_vs_performance.png, learning_curve.png, "
      "confusion_matrices.png, tree_tuned.png")