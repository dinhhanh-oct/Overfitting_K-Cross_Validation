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

