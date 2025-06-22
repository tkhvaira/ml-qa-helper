import pandas as pd
from src.predict import predict
from sklearn.ensemble import RandomForestClassifier

def test_predict(tmp_path):
    data = pd.DataFrame({
        "module": ["test_module_1", "test_module_2", "test_module_3"],
        "test_name": ["test_1", "test_2", "test_3"],
        "test_result": ["pass", "failed", "skipped"],
        "error_message": ["-", "Timeout", ""]
    })
    data_path = tmp_path / "test_data.csv"
    model_path = tmp_path / "test_model.joblib"
    data.to_csv(data_path, index=False)

    from src.train_model import train_model
    train_model(data_path, model_path)

    predictions = predict(model_path, data_path)
    assert len(predictions) == len(data)
    assert all(isinstance(p, int) for p in predictions)
    assert all(p in [0, 1] for p in predictions)

    from src.evaluate_model import evaluate_model
    evaluate_model(model_path, data_path, predictions)


