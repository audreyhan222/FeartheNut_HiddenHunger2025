
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay, precision_recall_curve
)
import matplotlib.pyplot as plt
import joblib
from data_cleaning import clean_data

# ---------------------------
# 1) Load data
# ---------------------------

df, preprocess = clean_data()


X = df.drop(columns=['Hidden_Hunger_Flag'])
y = df['Hidden_Hunger_Flag']


mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # try (128, 64) or (64, 64, 32) as variants
    activation="relu",
    solver="adam",
    alpha=1e-4,                    # L2 regularization
    batch_size=32,
    learning_rate="adaptive",
    max_iter=2000,
    early_stopping=True,           # uses 10% of training set as validation
    n_iter_no_change=20,
    random_state=42,
)

clf = Pipeline(steps=[("prep", preprocess), ("mlp", mlp)])

# ---------------------------
# 3) Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# Fit on train and evaluate on test
# ---------------------------
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nTest metrics:")
print(f"  Accuracy: {acc:.3f}")
print(f"  F1:       {f1:.3f}")
print(f"  ROC AUC:  {auc:.3f}\n")

print("Confusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))


prec, rec, thr = precision_recall_curve(y_test, y_prob)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_thr = thr[max(best_idx - 1, 0)] if len(thr) > 0 else 0.5

print(f"\nBest F1 threshold (test set): {best_thr:.3f}")
y_pred_opt = (y_prob >= best_thr).astype(int)
print("Metrics at optimized threshold:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_opt):.3f}")
print(f"  F1:       {f1_score(y_test, y_pred_opt):.3f}")
print(f"  ROC AUC:  {roc_auc_score(y_test, y_prob):.3f}")  # unchanged by threshold

# ---------------------------
# Plot ROC curve
# ---------------------------
# RocCurveDisplay.from_predictions(y_test, y_prob)
# plt.title("Neural Network ROC Curve")
# plt.tight_layout()
# plt.show()

# ---------------------------
# 8) Save the trained pipeline
# ---------------------------
# joblib.dump(clf, "mlp_hidden_hunger_pipeline.joblib")
# print("\nSaved model to mlp_hidden_hunger_pipeline.joblib")