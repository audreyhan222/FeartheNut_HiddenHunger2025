import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def clean_data():
    df = pd.read_csv('code/hidden_hunger.csv')

    target = "Hidden_Hunger_Flag"
    categorical = ["Gender", "Income_Bracket", "Education_Level"]
    numeric = [c for c in df.columns if c not in categorical + [target]]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    return df, preprocess



    # df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    # df['Income_Bracket'] = df['Income_Bracket'].map({'low': 0, 'lower_middle': 1, 'upper_middle': 2, 'high': 3})
    # df['Education_Level'] = df['Education_Level'].map({'primary': 0, 'secondary': 1, 'tertiary': 2})
    # df['Vitamin_A_Intake_ug'] = pd.to_numeric(df['Vitamin_A_Intake_ug'], errors='coerce')
    # df['Vitamin_D_Intake_IU'] = pd.to_numeric(df['Vitamin_D_Intake_IU'], errors='coerce')
    # df['Zinc_Intake_mg'] = pd.to_numeric(df['Zinc_Intake_mg'], errors='coerce')
    # df['Iron_Intake_mg'] = pd.to_numeric(df['Iron_Intake_mg'], errors='coerce')
    # df['Folate_Intake_ug'] = pd.to_numeric(df['Folate_Intake_ug'], errors='coerce')
    # df['Hidden_Hunger_Flag'] = df['Hidden_Hunger_Flag'].map({0: False, 1: True})

    # return df