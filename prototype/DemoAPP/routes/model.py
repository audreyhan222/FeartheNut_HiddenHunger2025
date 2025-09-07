import joblib
import pandas as pd

filename = 'my_model.sav'
loaded_model = joblib.load(filename)

def predict_risk(feature_inputs):
    values_df = pd.DataFrame([feature_inputs], columns=[
        'Age',
        'Gender', 
        'Income_Bracket',
        'Education_Level',
        'Vitamin_A_Intake_ug',
        'Vitamin_D_Intake_IU',
        'Zinc_Intake_mg',
        'Iron_Intake_mg',
        'Folate_Intake_ug'
    ])
    return loaded_model.predict(values_df)
