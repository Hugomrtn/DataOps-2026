from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


app = FastAPI()
model = joblib.load("model.joblib")


def transform_data(df):
    def vowel_consonant_ratio(name):
        vowels = sum(1 for c in name.lower() if c in "aeiouy")
        cons = sum(1 for c in name.lower() if c in "bcdfghjklmnpqrstvwxz")
        return vowels / cons if cons > 0 else 0

    encoder = OrdinalEncoder()

    df["vowel_cons_ratio"] = df["preusuel"].apply(vowel_consonant_ratio)
    df['last_letter'] = df['preusuel'].str[-1].str.lower()
    df["last_two"] = df["preusuel"].str[-2:].str.lower()

    df["last_letter"] = encoder.fit_transform(
        df["last_letter"].values.reshape(-1, 1))
    df["last_two"] = encoder.fit_transform(
        df["last_two"].values.reshape(-1, 1))

    print("Data transformed successfully")
    return df


@app.get("/")
def root():
    return {"message": "Gender Prediction API is running"}


@app.get("/predict")
def predict(name: str):
    df = pd.DataFrame({"preusuel": [name]})
    df = transform_data(df)
    features = ["vowel_cons_ratio", "last_letter", "last_two"]
    X = df[features]
    prediction = model.predict(X)[0]

    return {
        "name": name,
        "gender": prediction,
    }
