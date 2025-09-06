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

# Feature importance based on mean absolute SHAP values
feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\nFeature importance (mean |SHAP|):")
print(importance_df)

# Example: Explain a single prediction in detail
sample_idx = 0
sample_data = X_test.iloc[sample_idx]
sample_shap = shap_values[sample_idx]
sample_pred = model.predict_proba(X_test.iloc[[sample_idx]])[0, 1]

print(f"\n" + "="*50)
print(f"DETAILED EXPLANATION FOR SAMPLE {sample_idx}")
print(f"="*50)
print(f"Predicted probability: {sample_pred:.4f}")
print(f"Baseline probability: {1/(1+np.exp(-expected_value)):.4f}")
print(f"\nFeature contributions:")

for name, value, shap_val in zip(features_to_use, sample_data, sample_shap):
    direction = "increases" if shap_val > 0 else "decreases"
    print(f"  {name}: {value:.2f} -> {direction} prediction by {abs(shap_val):.4f}")


def plot_shap_summary():
    """Create SHAP summary plot"""
    shap.summary_plot(shap_values, X_test, feature_names=features_to_use, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

def plot_shap_waterfall(sample_idx=0):
    """Create waterfall plot for a single prediction"""
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx], 
            base_values=expected_value,
            data=X_test.iloc[sample_idx].values,
            feature_names=features_to_use
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}")
    plt.tight_layout()
    plt.show()

def plot_shap_bar():
    """Create bar plot of feature importance"""
    shap.bar_plot(
        shap.Explanation(
            values=shap_values,
            feature_names=features_to_use
        ),
        show=False
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

print(f"\nTo create visualizations, call:")
print(f"plot_shap_summary()    # Summary plot")
print(f"plot_shap_waterfall()  # Waterfall plot for single prediction")
print(f"plot_shap_bar()        # Bar plot of importance")

plot_shap_summary()