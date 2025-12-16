import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys
import warnings

warnings.filterwarnings("ignore")
np.random.seed(40)

if __name__ == "__main__":

    file_path = sys.argv[1] if len(sys.argv) > 1 else "weather_dataset_preprocessing.csv"

    df = pd.read_csv(file_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("rf", RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=X_train.head(5)
    )