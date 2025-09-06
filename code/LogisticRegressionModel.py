import shap
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from data_cleaning import clean_data

# Create sample dataset
model_df = clean_data()
features_to_use = ['Age', 'Gender', 'Income_Bracket', 'Education_Level', 'Vitamin_A_Intake_ug', 'Vitamin_D_Intake_IU', 'Zinc_Intake_mg', 'Iron_Intake_mg', 'Folate_Intake_ug']

X, y = model_df[features_to_use], model_df['Hidden_Hunger_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))
print("Model coefficients:")
for name, coef in zip(features_to_use, model.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# Create SHAP explainer
explainer = shap.LinearExplainer(model, X_train)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test)

print(f"\nSHAP values shape: {shap_values.shape}")
print(f"First prediction SHAP values:")
for name, shap_val in zip(features_to_use, shap_values[0]):
    print(f"  {name}: {shap_val:.4f}")

# SHAP values should sum to: prediction - expected_value
expected_value = explainer.expected_value
prediction_prob = model.predict_proba(X_test.iloc[[0]])[:, 1][0]
prediction_logit = np.log(prediction_prob / (1 - prediction_prob))

print(f"\nVerification for first prediction:")
print(f"Expected value (baseline): {expected_value:.4f}")
print(f"Sum of SHAP values: {np.sum(shap_values[0]):.4f}")
print(f"Expected + SHAP sum: {expected_value + np.sum(shap_values[0]):.4f}")
print(f"Actual prediction (logit): {prediction_logit:.4f}")

# Create summary of SHAP values
shap_df = pd.DataFrame(shap_values, columns=features_to_use)
print(f"\nSHAP values summary:")
print(shap_df.describe())