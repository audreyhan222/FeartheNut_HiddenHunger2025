import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from data_cleaning import clean_data


model_df = clean_data()
features_to_use = ['Age', 'Gender', 'Income_Bracket', 'Education_Level', 'Vitamin_A_Intake_ug', 'Vitamin_D_Intake_IU', 'Zinc_Intake_mg', 'Iron_Intake_mg', 'Folate_Intake_ug']

X, y = model_df[features_to_use], model_df['Hidden_Hunger_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Baseline performance
baseline_accuracy = model.score(X_test, y_test)
baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline AUC: {baseline_auc:.4f}")

