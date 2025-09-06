import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from data_cleaning import clean_data
warnings.filterwarnings('ignore')

model_df = clean_data()
features_to_use = ['Age', 'Gender', 'Income_Bracket', 'Education_Level', 'Vitamin_A_Intake_ug', 'Vitamin_D_Intake_IU', 'Zinc_Intake_mg', 'Iron_Intake_mg', 'Folate_Intake_ug']

X, y = model_df[features_to_use], model_df['Hidden_Hunger_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Learning rate shrinks contribution of each tree
    max_depth=3,             # Maximum depth of individual trees
    min_samples_split=2,     # Minimum samples required to split internal node
    min_samples_leaf=1,      # Minimum samples required to be at leaf node
    subsample=1.0,           # Fraction of samples used for fitting trees
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_accuracy_gb = gb_model.score(X_train, y_train)
test_accuracy_gb = gb_model.score(X_test, y_test)
auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

print(f"Training Accuracy: {train_accuracy_gb:.4f}")
print(f"Test Accuracy: {test_accuracy_gb:.4f}")
print(f"AUC Score: {auc_gb:.4f}")
print(f"Overfitting Check (train - test): {train_accuracy_gb - test_accuracy_gb:.4f}")

# Detailed results
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# Feature importance
print(f"\nFeature Importance:")
feature_importance_gb = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in feature_importance_gb.iterrows():
    print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")


def plot_feature_importance():
    """Plot feature importance from final model"""
    plt.figure(figsize=(10, 6))
    final_importance_sorted = final_importance.sort_values('Importance', ascending=True)
    plt.barh(range(len(final_importance_sorted)), final_importance_sorted['Importance'], 
             color='skyblue', alpha=0.8)
    plt.yticks(range(len(final_importance_sorted)), final_importance_sorted['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Gradient Boosting Feature Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_comparison():
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    gb_fpr, gb_tpr, _ = roc_curve(y_test, final_gb_model.predict_proba(X_test)[:, 1])
    
    # Plot ROC curves
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {final_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


print(f"\nTo create visualizations, call:")
print(f"plot_learning_curve()        # Learning curve analysis")
print(f"plot_feature_importance()    # Feature importance plot")
print(f"plot_roc_comparison()        # ROC curves for all models")

plot_feature_importance()
