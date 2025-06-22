import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data_preparation import load_and_clean_data, encode_features

def train_model(data_path, model_path):
    df = load_and_clean_data(data_path)
    df = encode_features(df)

    X = df.drop(["test_result"], axis=1)
    y = df["test_result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")