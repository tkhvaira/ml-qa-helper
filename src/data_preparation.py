import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['test_result', 'error_message'], inplace=True)
    df['test_result'] = df['test_result'].replace({'pass': 1, 'failed': 0, 'skipped': 0})
    return df

def encode_features(df):
    enc = LabelEncoder()
    df['module'] = enc.fit_transform(df['module'])
    df['test_name'] = enc.fit_transform(df['test_name'])
    return df