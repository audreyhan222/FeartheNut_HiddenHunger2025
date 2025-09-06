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

    print(df.head())
    return df, preprocess

