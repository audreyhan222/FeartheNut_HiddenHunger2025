
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
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
from data_cleaning import clean_data
import joblib


df, preprocess = clean_data()


X = df.drop(columns=['Hidden_Hunger_Flag'])
y = df['Hidden_Hunger_Flag']


mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   
    activation="relu",
    solver="adam",
    alpha=1e-4,                    
    batch_size=32,
    learning_rate="adaptive",
    max_iter=2000,
    early_stopping=True,           
    n_iter_no_change=20,
    random_state=42,
)

clf = Pipeline(steps=[("prep", preprocess), ("mlp", mlp)])

# ---------------------------
# Train/Test split
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
print(f"  ROC AUC:  {roc_auc_score(y_test, y_prob):.3f}")  


# # Create Predictions csv
predictions_df = pd.DataFrame(clf.predict(X))
predictions_df.to_csv('outputs/risk_predictions.csv', index=False)


# # Plot ROC curve

# # RocCurveDisplay.from_predictions(y_test, y_prob)
# # plt.title("Neural Network ROC Curve")
# # plt.tight_layout()
# # plt.show()





# # SHAP Plot
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt

# # Expect that clf (Pipeline with 'prep' + 'mlp'), X_train, X_test, y_test already exist from prior steps.
# # If not, reload data and rebuild the same pipeline quickly:
# try:
#     clf
#     X_test
#     X_train
# except NameError:
#     df = pd.read_csv("/mnt/data/hidden_hunger.csv")
#     target_col = "Hidden_Hunger_Flag"
#     categorical_cols = ["Gender", "Income_Bracket", "Education_Level"]
#     numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]
#     X = df[categorical_cols + numeric_cols]
#     y = df[target_col]

#     from sklearn.model_selection import train_test_split
#     from sklearn.compose import ColumnTransformer
#     from sklearn.preprocessing import OneHotEncoder, StandardScaler
#     from sklearn.pipeline import Pipeline
#     from sklearn.neural_network import MLPClassifier

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     preprocess = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
#             ("num", StandardScaler(), numeric_cols),
#         ]
#     )

#     mlp = MLPClassifier(
#         hidden_layer_sizes=(64, 32),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         batch_size=32,
#         learning_rate="adaptive",
#         max_iter=2000,
#         early_stopping=True,
#         n_iter_no_change=20,
#         random_state=42,
#     )

#     clf = Pipeline(steps=[("prep", preprocess), ("mlp", mlp)])
#     clf.fit(X_train, y_train)

# # Define a prediction function that returns probability for class 1
# def predict_proba_class1(Xinput):
#     # Ensure we maintain column names if Xinput is a numpy array
#     if isinstance(Xinput, np.ndarray):
#         X_df = pd.DataFrame(Xinput, columns=X_train.columns)
#     else:
#         X_df = Xinput
#     return clf.predict_proba(X_df)[:, 1]

# # Select a small background sample for KernelExplainer speed
# background = X_train.sample(min(100, len(X_train)), random_state=42)

# # Select a manageable subset of test data for explanation
# X_explain = X_test.sample(min(200, len(X_test)), random_state=7)

# # Initialize KernelExplainer
# explainer = shap.KernelExplainer(predict_proba_class1, background, link="logit")

# # Compute SHAP values (limit nsamples for speed)
# shap_values = explainer.shap_values(X_explain, nsamples=100)

# # Summary plot
# shap.summary_plot(shap_values, X_explain, show=False)
# plt.title("SHAP Summary Plot — Neural Network (MLP)")
# plt.tight_layout()
# plt.show()


# Feature Importance
# model_df = clean_data()[0]
# features_to_use = ['Age', 'Gender', 'Income_Bracket', 'Education_Level', 'Vitamin_A_Intake_ug', 'Vitamin_D_Intake_IU', 'Zinc_Intake_mg', 'Iron_Intake_mg', 'Folate_Intake_ug']

# X, y = model_df[features_to_use], model_df['Hidden_Hunger_Flag']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # feature_names = model_df.feature_names

# model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True, max_features='sqrt')
# model.fit(X_train, y_train)

# importances = model.feature_importances_

#     # Sort features by importance for better visualization
# indices = np.argsort(importances)[::-1]
# sorted_importances = importances[indices]
# sorted_feature_names = [features_to_use[i] for i in indices]

# plt.figure(figsize=(10, 6))
# plt.bar(range(X.shape[1]), sorted_importances, align='center')
# plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=90)
# plt.title("Random Forest Feature Importance")
# plt.xlabel("Feature")
# plt.ylabel("Importance (Mean Decrease in Impurity)")
# plt.tight_layout()
# plt.show()

filename = 'my_model.sav'
joblib.dump(clf, filename)