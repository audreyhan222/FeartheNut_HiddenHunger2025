from sklearn.ensemble import RandomForestClassifier # or RandomForestRegressor
from sklearn.datasets import load_iris # or your dataset
import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split

model_df = clean_data()
model_df['VitA/Age'] = model_df['Vitamin_A_Intake_ug'] + model_df['Age']
features_to_use = ['Age', 'Gender', 'Income_Bracket', 'Education_Level', 'Vitamin_D_Intake_IU', 'Zinc_Intake_mg', 'Iron_Intake_mg', 'Folate_Intake_ug', 'VitA/Age']

X, y = model_df[features_to_use], model_df['Hidden_Hunger_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature_names = model_df.feature_names

model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True, max_features='sqrt')
model.fit(X_train, y_train)

importances = model.feature_importances_

    # Sort features by importance for better visualization
indices = np.argsort(importances)[::-1]
sorted_importances = importances[indices]
sorted_feature_names = [features_to_use[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), sorted_importances, align='center')
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=90)
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance (Mean Decrease in Impurity)")
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
# Alternative: test_accuracy = accuracy_score(y_test, y_pred)

print("RANDOM FOREST MODEL RESULTS")
print("=" * 40)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference (overfitting check): {train_accuracy - test_accuracy:.4f}")