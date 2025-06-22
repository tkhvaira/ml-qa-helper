import joblib
import pandas as pd
from src.data_preparation import encode_features

def predict(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df = encode_features(df)
    predictions = model.predict(df)
    return predictions